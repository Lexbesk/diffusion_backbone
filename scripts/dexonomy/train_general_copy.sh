# main_dir=DEXONOMY_70k_pcdcentric
main_dir=DEXONOMY_type1_pcdcentric


DATA_PATH="/data/user_data/austinz/Robots/manipulation"

train_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_type1/train.zarr
eval_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_type1/val.zarr
# train_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_70k/train.zarr
# eval_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_70k/val.zarr
train_instructions=instructions/calvin/train_keypose_instructions.json
val_instructions=instructions/calvin/val_keypose_instructions.json

dataset=Dexonomy
num_workers=4
memory_limit=6
B=64
B_val=64
chunk_size=1

# Training/testing arguments, change these for HPT
val_freq=1000
eval_only=false
lr=1e-4
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
accurate_joint_pos=true
test_mujoco=true

run_log_dir=run_Jul16_$model_type-$dataset-lr$lr-$lr_scheduler-$denoise_model-B$B-Bval$B_val-DT$denoise_timesteps-ajp$accurate_joint_pos-embed$embedding_dim-C$C-n_attn_layers$num_shared_attn_layers-frozenpe
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth


ngpus=4

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
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --accurate_joint_pos $accurate_joint_pos \
    --test_mujoco $test_mujoco \