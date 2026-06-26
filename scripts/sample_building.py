import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
import math
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from omegaconf import OmegaConf

import torch
from torchvision import transforms
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import str2bool
from scripts.sample_utils import (
    load_model,
    save_args,
    make_model_input_shape,
    sample_batch,
    save_results,
    torch_to_np,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--building_id", type=str, default=None)
    parser.add_argument("--building_json", type=str, default=None)
    parser.add_argument("--building_root", type=str, default="/home/plj/buildings")
    parser.add_argument("--view_condition_dir", type=str, default=None)
    parser.add_argument("--view_condition_root", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--decode_frame_bs", type=int, default=1)
    parser.add_argument("--sample_type", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--low_vram", type=str2bool, default=True)
    parser.add_argument("--save_mp4", type=str2bool, default=True)
    parser.add_argument("--save_mp4_sheet", action="store_true", default=False)
    parser.add_argument("--save_npz", action="store_true", default=False)
    parser.add_argument("--save_jpg", action="store_true", default=False)
    parser.add_argument("--save_frames", action="store_true", default=False)
    parser.add_argument("--save_fps", type=int, default=8)
    return parser


def render_building_condition(building_json, num_frames, resolution=256, view_condition_dir=None):
    with open(building_json, "r") as f:
        buildings = json.load(f)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    frames = []
    for height_idx in range(1, num_frames + 1):
        img = Image.new("L", (resolution, resolution), 0)
        draw = ImageDraw.Draw(img)
        for polygon, height_values in buildings:
            height = float(height_values[0])
            if height < height_idx:
                continue
            draw.polygon([tuple(point) for point in polygon], fill=255)
        frame = to_tensor(img.convert("L").convert("RGB"))[:1]
        frame = frame.view(1, 1, resolution, resolution)
        if view_condition_dir is not None:
            view_path = os.path.join(view_condition_dir, f"h{height_idx}.png")
            if not os.path.isfile(view_path):
                raise FileNotFoundError(f"Missing view condition frame: {view_path}")
            view_frame = to_tensor(Image.open(view_path).convert("L").convert("RGB"))[:1]
            view_frame = view_frame.view(1, 1, resolution, resolution)
            frame = torch.cat([frame, view_frame], dim=0)
        frames.append(frame)
    return torch.cat(frames, dim=1)


@torch.no_grad()
def get_building_condition(model, opt):
    if opt.building_json is not None:
        building_json = opt.building_json
    elif opt.building_id is not None:
        building_json = os.path.join(opt.building_root, f"{opt.building_id}.json")
    else:
        raise ValueError("Provide --building_id or --building_json")

    view_condition_dir = opt.view_condition_dir
    if view_condition_dir is None and opt.view_condition_root is not None and opt.building_id is not None:
        view_condition_dir = os.path.join(opt.view_condition_root, opt.building_id)

    condition = render_building_condition(building_json, opt.num_frames, view_condition_dir=view_condition_dir)
    condition = condition.unsqueeze(0).repeat(opt.batch_size, 1, 1, 1, 1).to(model.device)
    condition = condition.to(next(model.cond_stage_model.parameters()).dtype)
    c = model.get_learned_conditioning(condition)
    return {"c_crossattn": [c]}


@torch.no_grad()
def sample_and_save_low_vram(model, noise_shape, n_iters, opt, condition, sampler=None, save_name="results"):
    saved = 0
    for _ in range(n_iters):
        samples_latent = sample_batch(
            model,
            noise_shape,
            condition=condition,
            sample_type=opt.sample_type,
            sampler=sampler,
            ddim_steps=opt.ddim_steps,
            eta=opt.eta,
            store_intermediates=False,
        )
        samples = model.decode_first_stage(samples_latent, decode_bs=opt.decode_frame_bs, return_cpu=True)
        videos = torch_to_np(samples)
        del samples_latent, samples
        torch.cuda.empty_cache()

        for local_idx in range(videos.shape[0]):
            if saved >= opt.n_samples:
                break
            save_results(
                videos[local_idx:local_idx + 1],
                opt.save_dir,
                save_name=f"{save_name}_{saved:03d}",
                save_fps=opt.save_fps,
                save_mp4=opt.save_mp4,
                save_npz=opt.save_npz,
                save_mp4_sheet=opt.save_mp4_sheet,
                save_jpg=opt.save_jpg,
                save_frames=opt.save_frames,
            )
            saved += 1


def main():
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    os.makedirs(opt.save_dir, exist_ok=True)
    save_args(opt.save_dir, opt)
    if opt.seed is not None:
        seed_everything(opt.seed)

    config = OmegaConf.load(opt.config_path)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    model, _, _ = load_model(config, opt.ckpt_path, gpu_id=opt.gpu_id)
    model.half()
    sampler = DDIMSampler(model) if opt.sample_type == "ddim" else None

    start = time.time()
    noise_shape = make_model_input_shape(model, opt.batch_size, T=opt.num_frames)
    n_iters = math.ceil(opt.n_samples / opt.batch_size)
    save_name = f"building{opt.building_id}_seed{opt.seed:05d}" if opt.seed is not None else f"building{opt.building_id}"

    with torch.no_grad():
        with autocast():
            condition = get_building_condition(model, opt)
            sample_and_save_low_vram(model, noise_shape, n_iters, opt, condition, sampler=sampler, save_name=save_name)

    print("Finish sampling!")
    print(f"Run time = {(time.time() - start):.2f} seconds")


if __name__ == "__main__":
    main()
