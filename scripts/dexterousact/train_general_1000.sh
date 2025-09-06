# main_dir=DEXONOMY_70k_pcdcentric
main_dir=DexterousAct_debug


DATA_PATH="/data/user_data/austinz/Robots/manipulation"

# train_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_type1new/train.zarr
# eval_data_dir=$DATA_PATH/zarr_datasets/Dexonomy_zarr_type1/val.zarr
train_data_dir=$DATA_PATH/zarr_datasets/dexterousact_1000/train.zarr
# train_data_dir=/data/group_data/katefgroup/datasets/austinz/zarr_datasets/Dexonomy_zarr_all/train.zarr
eval_data_dir=$DATA_PATH/zarr_datasets/dexterousact_1000/train.zarr
train_instructions=instructions/calvin/train_keypose_instructions.json
val_instructions=instructions/calvin/val_keypose_instructions.json

dataset=DexterousAct
memory_limit=6
lv2_batch_size=1 # equally divides the batch size B=64
B=32 # actual batch size is B * lv2_batch_size * K
B_val=64
chunk_size=1

# Training/testing arguments, change these for HPT
eval_only=false
lr=1e-4
lr_scheduler=constant
wd=5e-10
train_iters=600000
use_compile=true

# Model arguments, change (some of) these for new architectures
model_type=dexterousactor
bimanual=false
keypose_only=false

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=3

C=192
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=1

workspace_normalizer_buffer=0.01
quaternion_format=wxyz
relative_action=true
denoise_timesteps=10
denoise_model=rectified_flow

num_shared_attn_layers=10
embedding_dim=256
accurate_joint_pos=true
test_mujoco=true
condition_on_grasp_type_id=true
guidance_weight=1.5  # e.g., 1.5 for classifier-free guidance
val_set_all_anchor=true

nhist=4
nfuture=4
K=2 # number of timesteps sampled per episode during training

# # debugging choice
# val_freq=10
# vis_freq=10
# ngpus=2
# num_workers=2

# # training choice
val_freq=1000
vis_freq=10000000000
ngpus=4
num_workers=4


run_log_dir=run_Sep6_1000-B$B-lv2bs$lv2_batch_size-Bval$B_val-DT$denoise_timesteps-nhist$nhist-nfuture$nfuture-K$K-numlayers${num_shared_attn_layers}-embedding_dim$embedding_dim
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

# run_log_dir=run_alltypes_Jul20-B32-lv2bs4-Bval64-DT10-ajptrue-embed256-C192-nlayers30-visfreq100-typecondtrue
# checkpoint=train_logs/Dexonomy_zarr_prime/run_alltypes_Jul20-B32-lv2bs4-Bval64-DT10-ajptrue-embed256-C192-nlayers30-visfreq100-typecondtrue/last.pth




export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR="tmp"

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --checkpoint $checkpoint \
    --model_type $model_type \
    --dataset $dataset \
    --lr $lr \
    --lr_scheduler $lr_scheduler \
    --denoise_model $denoise_model \
    --denoise_timesteps $denoise_timesteps \
    --batch_size $B \
    --batch_size_val $B_val \
    --val_freq $val_freq \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --num_shared_attn_layers $num_shared_attn_layers \
    --chunk_size $chunk_size \
    --embedding_dim $embedding_dim \
    --num_workers $num_workers \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --accurate_joint_pos $accurate_joint_pos \
    --test_mujoco $test_mujoco \
    --lv2_batch_size $lv2_batch_size \
    --vis_freq $vis_freq \
    --condition_on_grasp_type_id $condition_on_grasp_type_id \
    --guidance_weight $guidance_weight \
    --val_set_all_anchor $val_set_all_anchor \
    --nhist $nhist \
    --nfuture $nfuture \
    --K $K \