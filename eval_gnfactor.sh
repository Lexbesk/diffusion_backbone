# You may need to run the following beforehard:
# export PYTHONPATH=$PYTHONPATH:$(pwd)



exp=flow_3dda_gnfactor
tasks=(
    close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups
)

data_dir=/data/group_data/katefgroup/VLA/peract_test
dataset=PeractSingleCam
instructions=instructions/peract/instructions.pkl
num_episodes=100
max_tries=2
max_steps=20
verbose=true
dense_interpolation=true
interpolation_length=2
num_history=3
denoise_timesteps=100
denoise_model=ddpm
quaternion_format=xyzw  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint
rotation_parametrization=6D
use_instruction=true
C=120
num_attn_heads=8
num_vis_ins_attn_layers=3
fps_subsampling_factor=5
relative_action=false
seed=0
checkpoint=ddpm.pth
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
        --output_file eval_logs/$exp/seed$seed/eval.json  \
        --use_instruction $use_instruction \
        --instructions $instructions \
        --max_tries $max_tries \
        --max_steps $max_steps \
        --seed $seed \
        --quaternion_format $quaternion_format \
        --interpolation_length $interpolation_length \
        --dense_interpolation $dense_interpolation \
        --headless $headless
done
