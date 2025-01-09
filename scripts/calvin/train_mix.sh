main_dir=CalvinMix

train_data_dir=data/calvin/packaged_ABC_D/training
eval_data_dir=data/calvin/packaged_ABC_D/validation
instructions=instructions/calvin_task_ABC_D

lr=3e-4
wd=5e-3
lr_scheduler=constant
dense_interpolation=1
interpolation_length=20
num_history=1
denoise_timesteps=5
denoise_model=rectified_flow
keypose_only=false
quaternion_format=wxyz
rotation_parametrization=6D
use_instruction=true
workspace_normalizer_buffer=0.01
B=72
B_val=24
C=216
train_iters=70000
val_freq=4000
workspace_normalizer_iter=128
precompute_instruction_encodings=true
fps_subsampling_factor=3
num_workers=4
dataset=TrainABCTestD_CalvinDataset
relative_action=true
backbone=resnet50
ngpus=1

run_log_dir=C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-DT$denoise_timesteps-BK$backbone
# checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth
checkpoint=none
eval_only=false

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_mix.py \
    --dataset $dataset \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --instructions $instructions \
    --precompute_instruction_encodings $precompute_instruction_encodings \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --workspace_normalizer_iter $workspace_normalizer_iter \
    --num_workers $num_workers \
    --train_iters $train_iters \
    --embedding_dim $C \
    --use_instruction $use_instruction \
    --rotation_parametrization $rotation_parametrization \
    --quaternion_format $quaternion_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --val_freq $val_freq \
    --fps_subsampling_factor $fps_subsampling_factor \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --batch_size $B \
    --batch_size_val $B_val \
    --keypose_only $keypose_only \
    --backbone $backbone \
    --lr $lr\
    --wd $wd \
    --lr_scheduler $lr_scheduler \
    --num_history $num_history \
    --relative_action $relative_action \
    --eval_only $eval_only \
    --checkpoint $checkpoint \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir}
