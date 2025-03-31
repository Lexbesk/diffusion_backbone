# You probably don't need to change these
merged_config_file=online_evaluation_calvin/configs/merged_config_val_abc_d.yaml
task_config_file=online_evaluation_calvin/configs/new_playtable_tasks.yaml
ann_config_file=online_evaluation_calvin/configs/new_playtable_validation.yaml

# Things you may want to change if you train/eval a new model
seed=0
checkpoint=train_logs/CALVIN/denoise3dsa-Calvin-C192-B64-lr3e-4-constant-H3-rectified_flow-DT10-clip-finetuned_false/best.pth
base_log_dir=eval_logs/CALVIN/denoise3dsa-Calvin-C192-B64-lr3e-4-constant-H3-rectified_flow-DT10-clip-finetuned_false/
save_video=false

# Things you can change if you customize the model architecture
model_type=denoise3dsa
pred_len=16
backbone=clip
fps_subsampling_factor=3
embedding_dim=192
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3
relative_action=true
quaternion_format=wxyz
denoise_timesteps=10
denoise_model=rectified_flow

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
    --model_type $model_type \
    --pred_len $pred_len \
    --backbone $backbone \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $embedding_dim \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --relative_action $relative_action \
    --quaternion_format $quaternion_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model
