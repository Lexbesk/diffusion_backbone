# You probably don't need to change these
merged_config_file=online_evaluation_calvin/configs/merged_config_val_abc_d.yaml
task_config_file=online_evaluation_calvin/configs/new_playtable_tasks.yaml
ann_config_file=online_evaluation_calvin/configs/new_playtable_validation.yaml

# Things you may want to change if you train/eval a new model
seed=0
checkpoint=calvin_new.pth
base_log_dir=eval_logs/Planner_Calvin/
save_video=0

# Things you can change if you customize the model architecture
backbone=clip
embedding_dim=192
num_attn_heads=8
num_vis_ins_attn_layers=3
rotation_parametrization=6D
quaternion_format=wxyz
denoise_timesteps=25
denoise_model=ddpm
fps_subsampling_factor=3
interpolation_length=20
relative_action=1

ngpus=1


torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy.py \
    --merged_config_file $merged_config_file \
    --task_config_file $task_config_file \
    --ann_config_file $ann_config_file \
    --seed $seed \
    --checkpoint $checkpoint \
    --base_log_dir $base_log_dir \
    --save_video $save_video \
    --backbone $backbone \
    --embedding_dim $embedding_dim \
    --num_attn_heads $num_attn_heads \
    --num_vis_ins_attn_layers $num_vis_ins_attn_layers \
    --rotation_parametrization $rotation_parametrization \
    --quaternion_format $quaternion_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --fps_subsampling_factor $fps_subsampling_factor \
    --interpolation_length $interpolation_length \
    --relative_action $relative_action
