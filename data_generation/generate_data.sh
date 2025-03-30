source /katefgroup/rlbench_user/miniconda3/bin/activate 3d_diffuser_actor
#export PYTHONPATH=$PYTHONPATH:$(pwd)
#export PYTHONPATH=$PYTHONPATH:/katefgroup/rlbench_user/miniconda3/envs/3d_diffuser_actor/lib/python3.8/site-packages:$(pwd)

seed=0
task=take_umbrella_out_of_umbrella_stand
variation=0
variation_count=1
tasks=(
    take_umbrella_out_of_umbrella_stand
    insert_usb_in_computer
    straighten_rope
    pick_and_lift_small
    put_knife_on_chopping_board
    place_shape_in_shape_sorter
    take_toilet_roll_off_stand
    put_umbrella_in_umbrella_stand
    setup_checkers
    wipe_desk
    stack_block
    take_shoes_out
    slide_cabinet_and_place_cups
    beat_the_buzz
    lift_numbered_block
    turn_tap
    pick_up_cup
    close_microwave
    close_fridge
    close_grill
    open_grill
    unplug_charger
    press_switch
    take_money_out_safe
    open_microwave
    put_money_in_safe
    open_door
    close_door
    open_fridge
    open_oven
    plug_charger_in_power_supply
    slide_block_to_target
    reach_and_drag
    take_frame_off_hanger
    water_plants
    hang_frame_on_hanger
    scoop_with_spatula
    place_hanger_on_rack
    move_hanger
    sweep_to_dustpan
    take_plate_off_colored_dish_rack
    screw_nail
    basketball_in_hoop
    put_rubbish_in_bin
    meat_off_grill
    meat_on_grill
    change_channel
    tv_on
    tower3
    push_buttons
    stack_wine
    turn_oven_on
    change_clock
    open_window
    open_wine_bottle
)

# 1. generate microstep demonstrations
num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
     xvfb-run -a python generate_dataset_microsteps.py \
          --save_path /data/user_data/mbronars/data/RLBench/decomp/seed{seed} \
          --all_task_file assets/all_tasks.json \
          --image_size 256,256 --renderer opengl \
          --episodes_per_task 100 \
          --tasks ${tasks[$i]} --variations ${variation_count} --offset ${variation} \
          --processes 1 --seed ${seed}
done
