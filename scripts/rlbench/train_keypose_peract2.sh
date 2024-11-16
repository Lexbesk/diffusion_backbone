main_dir=BiManualActor_13Peract2_100Demo_multitask

dataset=/scratch/peract2/train
valset=/scratch/peract2/test
dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/peract2/train
valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/peract2/test
dataset=/lustre/fsw/portfolios/nvr/users/ngkanatsios/packaged/peract2/train
valset=/lustre/fsw/portfolios/nvr/users/ngkanatsios/packaged/peract2/test

lr=1e-4
dense_interpolation=1
interpolation_length=2
num_history=1
diffusion_timesteps=100
B=8
C=120
ngpus=6
quaternion_format=wxyz
bimanual=1
gripper_loc_bounds_buffer=0.08
camera_calibration_aug=1
run_log_dir=diffusion_multitask_new-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-CA$camera_calibration_aug


CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory.py \
    --tasks bimanual_pick_laptop bimanual_pick_plate bimanual_straighten_rope bimanual_sweep_to_dustpan coordinated_lift_ball coordinated_lift_tray coordinated_push_box coordinated_put_bottle_in_fridge coordinated_put_item_in_drawer coordinated_take_tray_out_of_oven dual_push_buttons handover_item_easy handover_item \
    --dataset $dataset \
    --valset $valset \
    --instructions instructions/peract2/instructions.pkl \
    --gripper_loc_bounds tasks/13_peract2_tasks_location_bounds.json \
    --gripper_loc_bounds_buffer $gripper_loc_bounds_buffer \
    --num_workers 4 \
    --train_iters 600000 \
    --embedding_dim $C \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --camera_calibration_aug $camera_calibration_aug \
    --val_freq 1000 \
    --val_iters 8 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 14 \
    --cache_size 0 \
    --cache_size_val 0 \
    --keypose_only 1 \
    --variations {0..199} \
    --lr $lr\
    --bimanual $bimanual \
    --num_history $num_history \
    --cameras over_shoulder_left over_shoulder_right wrist_left wrist_right front\
    --max_episodes_per_task -1 \
    --quaternion_format $quaternion_format \
    --run_log_dir ${run_log_dir} \
    --checkpoint /lustre/fsw/portfolios/nvr/users/ngkanatsios/analogical_manipulation/train_logs/BiManualActor_13Peract2_100Demo_multitask/diffusion_multitask_new-C120-B8-lr1e-4-DI1-2-H1-DT100-CA1/last.pth

