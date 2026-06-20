import torch
from tqdm import tqdm
import numpy as np

def calculate_latent_stats(model, dataloader, device="cuda"):
    model.to(device)
    model.eval()
    
    all_latents = []
    
    print("正在提取隐空间特征...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # 这里的 batch['image'] 取决于你 Dataset 的返回格式
            x = batch["image"].to(device) 
            
            # 1. 通过 VAE Encoder 获得分布参数
            # 注意：AutoencoderKL 通常返回的是个分布对象或矩（moments）
            posterior = model.encode(x)
            
            # 2. 取分布的均值作为隐空间表征 (z)
            if hasattr(posterior, 'mode'):
                z = posterior.mode()
            else:
                # 某些实现中 encode 直接返回 tensor 或者 samples
                z = posterior 
            
            # 将 tensor 移到 CPU 释放显存
            all_latents.append(z.cpu().numpy())
            
            # 如果样本太多，可以提前停止
            if len(all_latents) > 100: 
                break

    # 合并所有特征 [N, C, T, H, W]
    all_latents = np.concatenate(all_latents, axis=0)
    
    # 3. 计算全局统计量
    # 针对所有维度计算均值和标准差
    mean = np.mean(all_latents)
    std = np.std(all_latents)
    
    print(f"\n计算结果:")
    print(f"建议 shift_factor (mean): {mean:.8f}")
    print(f"建议 scale_factor (1/std): {1/std:.8f}")
    
    return mean, 1/std

config.model['ckptdir'] = ckptdir
        config.model.params['logdir'] = logdir
        model = instantiate_from_config(config.model)
        # --- 强制加载权重，防止 Trainer 自动去读 checkpoint ---
        ckpt_path = "/home/plj/LVDM-main/models/2026-04-01T20-01-16_ae/checkpoints/last.ckpt"
        if os.path.exists(ckpt_path):
            print(f"正在手动加载模型权重: {ckpt_path}")
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            # 处理 state_dict 嵌套
            if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
                model.load_state_dict(ckpt_data["state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt_data, strict=False)
        
stats = calculate_latent_stats(vae_model, train_dataloader)