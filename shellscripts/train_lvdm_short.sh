
PROJ_ROOT="logs/lvdm_short "                      # root directory for saving experiment logs
EXPNAME="lvdm_short_rm"          # experiment name 
DATADIR="/home/plj/home/DataDisk/plj/lvdm"  # dataset directory
AEPATH="/home/plj/LVDM-main/models/2026-04-01T20-01-16_ae/checkpoints/last.ckpt"    # pretrained video autoencoder checkpoint

CONFIG="configs/lvdm_short/sky.yaml"
# OR CONFIG="configs/videoae/ucf.yaml"
# OR CONFIG="configs/videoae/taichi.yaml"
CKPTPATH="/home/plj/LVDM-main/logs/lvdm_short/lvdm_short_rm/checkpoints/epoch=0002-step=041324.ckpt" 
# run
export TOKENIZERS_PARALLELISM=false
python main.py \
--base $CONFIG \
-v   --gpu ,7 \
--name $EXPNAME \
--logdir $PROJ_ROOT \
-load_from_checkpoint $CKPTPATH \
--auto_resume True \
lightning.trainer.num_nodes=1 \
data.params.train.params.data_root=$DATADIR \
data.params.validation.params.data_root=$DATADIR \
model.params.first_stage_config.params.ckpt_path=$AEPATH

# -------------------------------------------------------------------------------------------------
# commands for multi nodes training
# - use torch.distributed.run to launch main.py
# - set `gpus` and `lightning.trainer.num_nodes`

# For example:

# python -m torch.distributed.run \
#     --nproc_per_node=8 --nnodes=$NHOST --master_addr=$MASTER_ADDR --master_port=1234 --node_rank=$INDEX \
#     main.py \
#     --base $CONFIG \
#     -t --gpus 0,1,2,3,4,5,6,7 \
#     --name $EXPNAME \
#     --logdir $PROJ_ROOT \
#     --auto_resume True \
#     lightning.trainer.num_nodes=$NHOST \
#     data.params.train.params.data_root=$DATADIR \
#     data.params.validation.params.data_root=$DATADIR
