exp=refactored_better_3dda_peract_sc20
tasks=(
    close_jar
    insert_onto_square_peg
    light_bulb_in
    meat_off_grill
    open_drawer
    place_shape_in_shape_sorter
    place_wine_at_rack_location
    push_buttons
    put_groceries_in_cupboard
    put_item_in_drawer
    put_money_in_safe
    reach_and_drag
    slide_block_to_color_target
    stack_blocks
    stack_cups
    sweep_to_dustpan_of_size
    turn_tap
    place_cups
)

# Testing arguments
checkpoint=train_logs/Peract/denoise3d-Peract-C120-B64-lr1e-4-constant-H3-rectified_flow-DT10/best.pth
exp=$exp/denoise3d-Peract-C120-B64-lr1e-4-constant-H3-rectified_flow-DT10
num_episodes=25
max_tries=2
max_steps=20
headless=true
collision_checking=false
seed=0
# Dataset arguments
data_dir=/data/group_data/katefgroup/VLA/peract_raw/test
instructions=instructions/peract/instructions.json
dataset=Peract
image_size=128,128
# Logging arguments
verbose=false
# Model arguments
model_type=denoise3d
bimanual=false
prediction_len=1
fps_subsampling_factor=5
embedding_dim=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3
relative_action=false
quaternion_format=xyzw
denoise_timesteps=10
denoise_model=rectified_flow

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
        --checkpoint $checkpoint \
        --task ${tasks[$i]} \
        --num_episodes $num_episodes \
        --max_tries $max_tries \
        --max_steps $max_steps \
        --headless $headless \
        --collision_checking $collision_checking \
        --seed $seed \
        --data_dir $data_dir \
        --test_instructions $instructions \
        --dataset $dataset \
        --image_size $image_size \
        --output_file eval_logs/$exp/seed$seed/${tasks[$i]}/eval.json  \
        --verbose $verbose \
        --model_type $model_type \
        --bimanual $bimanual \
        --prediction_len $prediction_len \
        --fps_subsampling_factor $fps_subsampling_factor \
        --embedding_dim $embedding_dim \
        --num_attn_heads $num_attn_heads \
        --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
        --num_history $num_history \
        --relative_action $relative_action \
        --quaternion_format $quaternion_format \
        --denoise_timesteps $denoise_timesteps \
        --denoise_model $denoise_model
done

python online_evaluation_rlbench/collect_results.py \
    --folder eval_logs/$exp/$checkpoint/seed$seed/
