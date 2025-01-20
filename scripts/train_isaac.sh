# rm -r /scratch/GNFactor_zarr
# cp -r /data/user_data/ngkanats/GNFactor_zarr /scratch/

main_dir=Isaac

train_data_dir=/data/user_data/ngkanats/isaac_zarr.zarr
eval_data_dir=/data/user_data/ngkanats/isaac_zarr.zarr/

lr=1e-4
lr_scheduler=constant
num_history=1
denoise_timesteps=10
denoise_model=rectified_flow
keypose_only=false
quaternion_format=wxyz
rotation_parametrization=6D
fps_subsampling_factor=8
backbone=clip
use_instruction=false
relative_action=true
workspace_normalizer_buffer=0.05
B=128
B_val=64
C=144
train_iters=200000
val_freq=4000
precompute_instruction_encodings=true
num_workers=4
dataset=Isaac
ngpus=4
ngpus=1

run_log_dir=all_ca_C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-DT$denoise_timesteps
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth
# checkpoint=none
eval_only=true

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_fast_isaac.py \
    --dataset $dataset \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --precompute_instruction_encodings $precompute_instruction_encodings \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --num_workers $num_workers \
    --train_iters $train_iters \
    --embedding_dim $C \
    --use_instruction $use_instruction \
    --relative_action $relative_action \
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
    --lr_scheduler $lr_scheduler \
    --num_history $num_history \
    --eval_only $eval_only \
    --checkpoint $checkpoint \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir}
