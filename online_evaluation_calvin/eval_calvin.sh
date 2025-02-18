main_dir=Planner_Calvin

calvin_dataset_path=calvin/dataset/task_ABC_D
calvin_model_path=/data/user_data/ngkanats/calvin/calvin_models
calvin_gripper_loc_bounds=calvin/dataset/task_ABC_D/validation/statistics.yaml

interpolation_length=20
num_history=3
diffusion_timesteps=25
denoise_model=ddpm
C=192
num_attn_heads=8
num_vis_ins_attn_layers=3
ngpus=1
backbone=clip
relative_action=1
fps_subsampling_factor=3
lang_enhanced=1
quaternion_format=wxyz  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint
rotation_parametrization=6D


torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy.py \
    --calvin_dataset_path $calvin_dataset_path \
    --calvin_model_path $calvin_model_path \
    --text_encoder clip \
    --text_max_length 16 \
    --tasks A B C D\
    --backbone $backbone \
    --calvin_gripper_loc_bounds $calvin_gripper_loc_bounds \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_ins_attn_layers $num_vis_ins_attn_layers \
    --use_instruction 1 \
    --rotation_parametrization $rotation_parametrization \
    --diffusion_timesteps $diffusion_timesteps \
    --denoise_model $denoise_model \
    --interpolation_length $interpolation_length \
    --num_history $num_history \
    --relative_action $relative_action \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --save_video 0 \
    --base_log_dir train_logs/${main_dir}/pretrained/eval_logs/ \
    --quaternion_format $quaternion_format \
    --checkpoint calvin_new.pth
