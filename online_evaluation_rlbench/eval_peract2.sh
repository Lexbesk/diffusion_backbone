exp=flow_bimanual_3dda_train
tasks=(
    bimanual_push_box
    bimanual_lift_ball
    bimanual_dual_push_buttons
    bimanual_pick_plate
    bimanual_put_item_in_drawer
    bimanual_put_bottle_in_fridge
    bimanual_handover_item
    bimanual_pick_laptop
    bimanual_straighten_rope
    bimanual_sweep_to_dustpan
    bimanual_lift_tray
    bimanual_handover_item_easy
    bimanual_take_tray_out_of_oven
)

data_dir=/data/group_data/katefgroup/VLA/peract2_raw_squash/test/
dataset=Peract2
instructions=instructions/peract2/instructions.json
num_episodes=100
max_tries=2
max_steps=25
verbose=false
interpolation_length=2
num_history=3
denoise_timesteps=10
denoise_model=rectified_flow
quaternion_format=xyzw  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint
rotation_parametrization=6D
use_instruction=true
C=120
num_attn_heads=8
num_vis_ins_attn_layers=3
fps_subsampling_factor=5
relative_action=false
seed=0
checkpoint=peract2_front.pth
headless=true

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
        --task ${tasks[$i]} \
        --checkpoint $checkpoint \
        --dataset $dataset \
        --denoise_timesteps $denoise_timesteps \
        --denoise_model $denoise_model \
        --fps_subsampling_factor $fps_subsampling_factor \
        --relative_action $relative_action \
        --num_history $num_history \
        --verbose $verbose \
        --collision_checking false \
        --embedding_dim $C \
        --num_attn_heads $num_attn_heads \
        --num_vis_ins_attn_layers $num_vis_ins_attn_layers \
        --rotation_parametrization $rotation_parametrization \
        --data_dir $data_dir \
        --num_episodes $num_episodes \
        --output_file eval_logs/$exp/$checkpoint/seed$seed/${tasks[$i]}/eval.json  \
        --use_instruction $use_instruction \
        --instructions $instructions \
        --max_tries $max_tries \
        --max_steps $max_steps \
        --seed $seed \
        --quaternion_format $quaternion_format \
        --interpolation_length $interpolation_length \
        --headless $headless \
        --bimanual true
done

