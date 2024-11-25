exp=flow_actor_gnfactor

data_dir=./data/peract/raw/test/
instructions=instructions/peract/instructions.pkl
num_episodes=100
use_instruction=True
max_tries=2
max_steps=20
verbose=True
interpolation_length=2
embedding_dim=144
fps_subsampling_factor=5
relative_action=False
collision_checking=False
predict_trajectory=True
seed=0
denoise_model=rectified_flow
denoise_timesteps=10
rotation_parametrization=6D
checkpoint=train_logs/diffuser_actor_gnfactor.pth
quaternion_format=wxyz  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint

python online_evaluation_rlbench/evaluate_policy.py \
    --checkpoint $checkpoint \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --fps_subsampling_factor $fps_subsampling_factor \
    --relative_action $relative_action \
    --num_history 1 \
    --verbose $verbose \
    --collision_checking $collision_checking \
    --predict_trajectory $predict_trajectory \
    --embedding_dim $embedding_dim \
    --rotation_parametrization $rotation_parametrization \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions $instructions \
    --max_tries $max_tries \
    --max_steps $max_steps \
    --seed $seed \
    --quaternion_format $quaternion_format \
    --interpolation_length $interpolation_length \
    --dense_interpolation True