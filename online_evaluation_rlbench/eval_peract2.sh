exp=flow_3dda

data_dir=./data/peract2/raw/test/
dataset=Peract2
instructions=instructions/peract2/instructions.pkl
num_episodes=100
max_tries=2
verbose=true
dense_interpolation=true
interpolation_length=2
num_history=1
denoise_timesteps=10
denoise_model=rectified_flow
quaternion_format=wxyz  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint
rotation_parametrization=6D
use_instruction=true
C=144
fps_subsampling_factor=5
relative_action=false
seed=0
checkpoint=train_logs/Peract/C144-B12-lr1e-4-constant-H1-rectified_flow-DT10/last.pth

python online_evaluation_rlbench/evaluate_policy.py \
    --checkpoint $checkpoint \
    --dataset $dataset \
    --denoise_timesteps $denoise_timesteps \
    --fps_subsampling_factor $fps_subsampling_factor \
    --relative_action $relative_action \
    --num_history $num_history \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking false \
    --embedding_dim $C \
    --rotation_parametrization $rotation_parametrization \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/eval.json  \
    --use_instruction $use_instruction \
    --instructions $instructions \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --quaternion_format $quaternion_format \
    --interpolation_length $interpolation_length \
    --dense_interpolation $dense_interpolation

