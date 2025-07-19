# main_dir=DEXONOMY_7k_pcdcentric
main_dir=DEXONOMY_70k_pcdcentric

DATA_PATH="/data/user_data/austinz/Robots/manipulation"

# train_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_7000/train.zarr
# eval_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_7000/val.zarr
train_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_70k/train.zarr
eval_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_70k/val.zarr
train_instructions=instructions/calvin/train_keypose_instructions.json
val_instructions=instructions/calvin/val_keypose_instructions.json

dataset=Dexonomy
num_workers=4
memory_limit=6
B=128
B_val=8
chunk_size=1

# Training/testing arguments, change these for HPT
val_freq=100
eval_only=true
eval_overfit=false
lr=3e-4
lr_scheduler=constant
wd=5e-10
train_iters=600000
use_compile=true

# Model arguments, change (some of) these for new architectures
model_type=grasp_denoiser
bimanual=false
keypose_only=false

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=3

C=192
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=1

workspace_normalizer_buffer=0.01
quaternion_format=wxyz
relative_action=true
denoise_timesteps=10
denoise_model=rectified_flow

num_shared_attn_layers=20
embedding_dim=256
visualize_denoising_steps=true
accurate_joint_pos=true

run_log_dir=overfit_is_good_$model_type-$dataset-lr$lr-$lr_scheduler-$denoise_model-B$B-Bval$B_val-DT$denoise_timesteps

checkpoint=train_logs/DEXONOMY_70k_pcdcentric/run_Jul16_grasp_denoiser-Dexonomy-lr1e-4-constant-rectified_flow-B32-Bval8-DT10-ajptrue-embed256-C192-n_attn_layers20-frozenpe/last.pth

ngpus=1

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --checkpoint $checkpoint \
    --model_type $model_type \
    --dataset $dataset \
    --lr $lr \
    --lr_scheduler $lr_scheduler \
    --denoise_model $denoise_model \
    --denoise_timesteps $denoise_timesteps \
    --batch_size $B \
    --batch_size_val $B_val \
    --val_freq $val_freq \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --num_shared_attn_layers $num_shared_attn_layers \
    --chunk_size $chunk_size \
    --embedding_dim $embedding_dim \
    --num_workers $num_workers \
    --eval_only $eval_only \
    --eval_overfit $eval_overfit \
    --visualize_denoising_steps $visualize_denoising_steps \
    --accurate_joint_pos $accurate_joint_pos \