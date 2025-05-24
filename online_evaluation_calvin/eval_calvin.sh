# You probably don't need to change these
merged_config_file=online_evaluation_calvin/configs/merged_config_val_abc_d.yaml
# merged_config_file=online_evaluation_calvin/configs/merged_config_train_abc_d.yaml
task_config_file=online_evaluation_calvin/configs/new_playtable_tasks.yaml
ann_config_file=online_evaluation_calvin/configs/new_playtable_validation.yaml
# ann_config_file=online_evaluation_calvin/configs/new_playtable.yaml

# Things you may want to change if you train/eval a new model
seed=0
checkpoint=flower.pth
checkpoint=train_logs/CALVIN/flower-Calvin-C192-B8-lr2e-5-tristage_flower-H1-rectified_flow-clip-finetuned_false/last.pth
base_log_dir=eval_logs/CALVIN/flower_retrained_50k/
save_video=false

# Things you can change if you customize the model architecture
model_type=flower # denoise3dle
pred_len=10
pre_tokenize=false
custom_img_size=224
backbone=clip
fps_subsampling_factor=3
embedding_dim=192
num_attn_heads=8
num_vis_instr_attn_layers=3
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
    --pre_tokenize $pre_tokenize \
    --custom_img_size $custom_img_size \
    --backbone $backbone \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $embedding_dim \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --relative_action $relative_action \
    --quaternion_format $quaternion_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model
