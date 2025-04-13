main_dir=Mobaloha

train_data_dir=/ws/train_data/3cams_traj_zarr/train.zarr
eval_data_dir=/ws/train_data/3cams_traj_zarr/val.zarr
train_instructions=instructions/mobaloha/instructions.json
val_instructions=instructions/mobaloha/instructions.json

dataset=Mobaloha
num_workers=4
B=2
B_val=4

# Training/testing arguments, change these for HPT
val_freq=100
eval_only=false
lr=1e-4
lr_scheduler=constant
wd=5e-3
train_iters=600000

# Model arguments, change (some of) these for new architectures
model_type=denoise3d
bimanual=true
keypose_only=false

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=5

C=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3

workspace_normalizer_buffer=0.05
quaternion_format=xyzw
relative_action=false
denoise_timesteps=30
denoise_model=rectified_flow

run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-DT$denoise_timesteps
checkpoint=None
#checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth
#checkpoint=peract2_front_wrist3d_2.pth

ngpus=1

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --batch_size $B \
    --batch_size_val $B_val \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
    --val_freq $val_freq \
    --eval_only $eval_only \
    --lr $lr \
    --lr_scheduler $lr_scheduler \
    --wd $wd \
    --train_iters $train_iters \
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --quaternion_format $quaternion_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model
