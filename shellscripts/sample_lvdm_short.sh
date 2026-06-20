
CONFIG_PATH="configs/lvdm_short/sky.yaml"
BASE_PATH="/home/plj/LVDM-main/logs/lvdm_short/lvdm_short_rm/checkpoints/epoch=0002-step=041324.ckpt"
AEPATH="/home/plj/LVDM-main/models/2026-04-01T20-01-16_ae/checkpoints/last.ckpt"
OUTDIR="results/rm/"
PYTHON="/home/plj/anaconda3/envs/lvdm/bin/python"

$PYTHON scripts/sample_uncond.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --num_frames 20 \
    --decode_frame_bs 1 \
    --sample_type ddim \
    --ddim_steps 50 \
    --low_vram true \
    --seed 1000 \
    --save_mp4 false \
    --save_jpg \
    --save_frames \
    model.params.first_stage_config.params.ckpt_path=$AEPATH

# if use DDIM： add: `--sample_type ddim --ddim_steps 50`
