main_dir=BiManualActor_MobileAloha

dataset=/home/tsungwek/data/mobile_aloha
valset=/home/tsungwek/data/mobile_aloha

lr=1e-4
wd=5e-3
dense_interpolation=1
interpolation_length=26
num_history=1
diffusion_timesteps=50
B=8
C=120
ngpus=1
quaternion_format=xyzw
bimanual=1
relative_action=1
gripper_loc_bounds_buffer=0.08
keypose_only=1
run_log_dir=diffusion_multitask-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-R$relative_action-rgbfix-cleanbg-KP$keypose_only-correct_data


CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    mobaloha.py \
    --tasks close_pen pick_up_plate pouring_into_bowl put_block_into_bowl stack_block \
    --current_task close_pen \
    --dataset $dataset \
    --valset $valset \
    --gripper_loc_bounds tasks/mobaloha_multitasks_rel_keypose_location_bounds.json \
    --gripper_loc_bounds_buffer $gripper_loc_bounds_buffer \
    --num_workers 4 \
    --train_iters 200000 \
    --embedding_dim $C \
    --instructions instructions/mobaloha/instructions.pkl \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq 4000 \
    --val_iters 8 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 14 \
    --cache_size 0 \
    --cache_size_val 0 \
    --keypose_only 0 \
    --variations {0..0} \
    --lr $lr\
    --wd $wd \
    --bimanual $bimanual \
    --num_history $num_history \
    --cameras front\
    --max_episodes_per_task -1 \
    --max_episode_length 20 \
    --relative_action $relative_action \
    --quaternion_format $quaternion_format \
    --eval_only 1 \
    --keypose_only $keypose_only \
    --checkpoint train_logs/$main_dir/$run_log_dir/last.pth \
    --run_log_dir ${run_log_dir}

