
CONFIG_PATH="configs/lvdm_short/sky.yaml"
BASE_PATH="/home/plj/LVDM-main/logs/lvdm_short/lvdm_short_rm/checkpoints/epoch=0002-step=041324.ckpt"
AEPATH="/home/plj/LVDM-main/models/2026-04-01T20-01-16_ae/checkpoints/last.ckpt"
OUTDIR="results/rm/"

python scripts/sample_uncond.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --save_dir $OUTDIR \
    -t --gpus 0, \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --precision 16  \
    --show_denoising_progress \
    model.params.first_stage_config.params.ckpt_path=$AEPATH

# if use DDIM： add: `--sample_type ddim --ddim_steps 50`
