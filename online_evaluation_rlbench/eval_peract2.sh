exp=flow_bimanual_debug_microstep
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
# tasks=(
#     bimanual_handover_item
#     bimanual_handover_item_easy
#     # bimanual_take_tray_out_of_oven
# )

# Testing arguments
checkpoint=/home/ngkanats/repos/lbs/analogical_manipulation/train_logs/Peract2/fivecam_denoise3d-Peract2All-C120-B64-lr1e-4-constant-H3-ddpm-DT100/last.pth
checkpoint_alias=fivecam_denoise3d-Peract2All-C120-B64-lr1e-4-constant-H3-ddpm-DT100
# checkpoint=peract2_front_wrist3d_2.pth
checkpoint_alias=test_3dda
num_episodes=100
max_tries=2
max_steps=25
headless=true
collision_checking=false
seed=0
replay=false
# Dataset arguments
data_dir=/data/group_data/katefgroup/VLA/peract2_raw_squash/test/
dataset=Peract2All
image_size=256,256
# Model arguments
model_type=denoise3d
bimanual=true
prediction_len=1
fps_subsampling_factor=4
embedding_dim=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3
relative_action=false
quaternion_format=xyzw
denoise_timesteps=100
denoise_model=ddpm

# num_ckpts=${#tasks[@]}
# for ((i=0; i<$num_ckpts; i++)); do
#     xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
#         --checkpoint $checkpoint \
#         --task ${tasks[$i]} \
#         --num_episodes $num_episodes \
#         --max_tries $max_tries \
#         --max_steps $max_steps \
#         --headless $headless \
#         --collision_checking $collision_checking \
#         --seed $seed \
#         --replay $replay \
#         --data_dir $data_dir \
#         --dataset $dataset \
#         --image_size $image_size \
#         --output_file eval_logs/$exp/$checkpoint_alias/seed$seed/${tasks[$i]}/eval.json  \
#         --model_type $model_type \
#         --bimanual $bimanual \
#         --prediction_len $prediction_len \
#         --fps_subsampling_factor $fps_subsampling_factor \
#         --embedding_dim $embedding_dim \
#         --num_attn_heads $num_attn_heads \
#         --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
#         --num_history $num_history \
#         --relative_action $relative_action \
#         --quaternion_format $quaternion_format \
#         --denoise_timesteps $denoise_timesteps \
#         --denoise_model $denoise_model
# done

python online_evaluation_rlbench/collect_results.py \
    --folder eval_logs/$exp/$checkpoint_alias/seed$seed/
