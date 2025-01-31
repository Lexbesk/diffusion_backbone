# rm -r /scratch/CALVIN_zarr
# cp -r /data/user_data/ngkanats/CALVIN_zarr /scratch/

main_dir=CalvinFast

train_data_dir=/lustre/fsw/portfolios/nvr/users/ngkanatsios/CALVIN_zarr/train.zarr
eval_data_dir=/lustre/fsw/portfolios/nvr/users/ngkanatsios/CALVIN_zarr/val.zarr
train_instructions=instructions/calvin_task_ABC_D/training.pkl
val_instructions=instructions/calvin_task_ABC_D/validation.pkl

lr=3e-4
wd=5e-3
lr_scheduler=constant
num_history=3
denoise_timesteps=25  # 10
denoise_model=ddpm
keypose_only=false
quaternion_format=wxyz
rotation_parametrization=6D
fps_subsampling_factor=3
backbone=clip
use_instruction=true
workspace_normalizer_buffer=0.01  # 0.05
relative_action=true
B=256
B_val=64
C=192
num_attn_heads=8
num_vis_ins_attn_layers=3
train_iters=600000
val_freq=4000
precompute_instruction_encodings=true
num_workers=4
dataset=ABC_D
ngpus=4

run_log_dir=C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-DT$denoise_timesteps
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth
# checkpoint=none
eval_only=false

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_fast.py \
    --dataset $dataset \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --precompute_instruction_encodings $precompute_instruction_encodings \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --num_workers $num_workers \
    --train_iters $train_iters \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_ins_attn_layers $num_vis_ins_attn_layers \
    --use_instruction $use_instruction \
    --rotation_parametrization $rotation_parametrization \
    --fps_subsampling_factor $fps_subsampling_factor \
    --backbone $backbone \
    --quaternion_format $quaternion_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --val_freq $val_freq \
    --batch_size $B \
    --batch_size_val $B_val \
    --keypose_only $keypose_only \
    --lr $lr\
    --wd $wd \
    --lr_scheduler $lr_scheduler \
    --num_history $num_history \
    --relative_action $relative_action \
    --eval_only $eval_only \
    --checkpoint $checkpoint \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir}
