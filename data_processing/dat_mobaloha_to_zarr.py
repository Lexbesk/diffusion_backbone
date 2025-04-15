import json
from pathlib import Path
import pickle
import os
import blosc
from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm
from rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    store_instructions
)

ROOT = '/media/jiahe/data/data/aloha/keypose_3cams/'
STORE_PATH = '/media/jiahe/data/data/aloha/keypose_3cams/3cams_keypose_zarr/'
# ROOT = '/media/jiahe/data/data/aloha/trajectory_3cams/'
# STORE_PATH = '/media/jiahe/data/data/aloha/trajectory_3cams/zarr/'
STORE_EVERY = 1  # in keyposes
NCAM = 3
IM_SIZE = 256


# left_gripper_camera_extrinsics = np.array([
#   [ 0.03844001, -0.42007775,  0.90667361, -0.085],
#   [-0.99888832, -0.04092953,  0.02338624,  0.009],
#   [ 0.02728569, -0.90656464, -0.42118409,  0.075],
#   [ 0.        ,  0.        ,  0.        ,  1.   ],
# ])

left_gripper_camera_intrinsics = np.array([
    [ 256.,    0.       ,  128.       ],
    [   0.      ,   256.,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])


# right_gripper_camera_extrinsics = np.array([
#   [ 0.03844001, -0.42007775,  0.90667361, -0.085],
#   [-0.99888832, -0.04092953,  0.02338624,  0.009],
#   [ 0.02728569, -0.90656464, -0.42118409,  0.075],
#   [ 0.        ,  0.        ,  0.        ,  1.   ],
# ])
right_gripper_camera_intrinsics = np.array([
    [ 256.,    0.       ,  128.       ],
    [   0.      ,   256.,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])

front_camera_extrinsics = np.array([
  [-0.06624313, -0.6334383,   0.7709525,  -0.1514688 ],
  [-0.99780121,  0.04371347, -0.0498185,   0.06114338],
  [-0.00214406, -0.77255747, -0.63494122,  0.43910462],
  [ 0.,          0.,          0.,          1.        ],
])
front_camera_intrinsics = np.array([
    [ 256.,    0.       ,  128.       ],
    [   0.      ,   256.,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])
# cameras = {
#     'left': {
#         'extrinsics': left_gripper_camera_extrinsics,
#         'intrinsics': left_gripper_camera_intrinsics
#     },
#     'right': {
#         'extrinsics': right_gripper_camera_extrinsics,
#         'intrinsics': right_gripper_camera_intrinsics
#     },
#     'front': {
#         'extrinsics': front_camera_extrinsics,
#         'intrinsics': front_camera_intrinsics
#     }
# }



def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def all_tasks_main(split, tasks):

    camera_order = ['front', 'left', 'right']
    task2id = {task: t for t, task in enumerate(tasks)}
    variations = [0]

    # Collect all episodes
    episodes = []
    for task in tasks:
        for var in variations:
            _path = Path(f'{ROOT}{split}/{task}+{var}/')
            print("_path: ", _path)
            if not _path.is_dir():
                continue
            episodes.extend([
                (task, var, ep) for ep in sorted(_path.glob("*.npy"))
            ])

    # Read once to get the number of keyposes
    n_keyposes = 0
    for _, _, ep in tqdm(episodes):
        # print("ep: ", ep)
        ep_data = np.load(ep, allow_pickle=True)
        ep_data = ep_data.item()
        n_keyposes += len(ep_data['frames'])

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"{STORE_PATH}{split}.zarr", mode="w") as zarr_file:
        zarr_file.create_dataset(
            "rgb",
            shape=(0, NCAM, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, NCAM, 3, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "depth",
            shape=(0, NCAM, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, NCAM, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "proprioception",
            shape=(0, 3, 2, 8),
            chunks=(STORE_EVERY, 3, 2, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "action",
            shape=(0, 1, 2, 8),
            chunks=(STORE_EVERY, 1, 2, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "extrinsics",
            shape=(0, NCAM, 4, 4),
            chunks=(STORE_EVERY, NCAM, 4, 4),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "intrinsics",
            shape=(0, NCAM, 3, 3),
            chunks=(STORE_EVERY, NCAM, 3, 3),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "task_id", shape=(0,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "variation", shape=(0,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="uint8"
        )
        # Loop through episodes
        start = 0
        for task, var, ep in episodes: #tqdm(episodes):
            # Read
            print("task, var, ep: ", task, var, ep)
            content = np.load(ep, allow_pickle=True).item()
            
            # content['frames'] = content['frames'][:-1]                
            # Map [-1, 1] to [0, 255] uint8
            # rgb = (255 * (content['obs'][:, -NCAM:, 0].numpy())).astype(np.uint8)

            obs = torch.stack(content['obs']).numpy()
            rgb = ( 255 * obs[ :, :, 0]).astype(np.uint8)
            
            # print("min, max: ", np.min(rgb), " ", np.max(rgb))
            # Extract depth from point cloud, faster loading
            
            depth = (torch.stack(content['depth']).numpy()).astype(np.float16)
            depth = depth / 1000.
            # print("min, max: ", np.min(depth), " ", np.max(depth))

            # Store current eef pose as well as two previous ones
            prop = np.stack([
                to_numpy(tens).astype(np.float32) for tens in content['gripper']
            ])
            prop_1 = np.concatenate([prop[:1], prop[:-1]])
            prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
            # prop = np.concatenate([prop_2, prop_1, prop], 1)
            prop = np.stack([prop_2, prop_1, prop], 1)
            # Next keypose (concatenate curr eef to form a "trajectory")
            actions = np.stack([
                to_numpy(tens).astype(np.float32) for tens in content['action']
            ])


            # Extrinsics (keyframes, cameras, 4, 4)
            extrinsics = np.stack([
                content['extrinsic'][k]
                for k in content['frames']
            ])

            # Intrinsics (keyframes, cameras, 3, 3)
            intrinsics = np.stack([
                np.stack([
                    front_camera_intrinsics,
                    front_camera_intrinsics,
                    front_camera_intrinsics
                ])
                for k in content['frames']
            ])

            actions = np.expand_dims(actions, axis = 1)
            print("actions: ", actions.shape)
            # print("prop[:, -1:]: ", prop[:, -1:].shape )
            # actions = np.concatenate([prop[:, -1:], actions], 1)
            # Task ids and variation ids
            task_id = np.array([task2id[task]] * len(content['frames'])).astype(np.uint8)
            var_ = np.array([var] * len(content['frames'])).astype(np.uint8)
            print()
            print("actions: ", actions.shape)
            print("intrinsics: ", intrinsics.shape)
            print("n_keyposes: ", len(content['frames']))
            # Write
            zarr_file['rgb'].append(rgb)
            zarr_file['depth'].append(depth)
            zarr_file['proprioception'].append(prop)
            zarr_file['action'].append(actions)
            zarr_file['extrinsics'].append(extrinsics)
            zarr_file['intrinsics'].append(intrinsics)
            zarr_file['task_id'].append(task_id)
            zarr_file['variation'].append(var_)



if __name__ == "__main__":

    tasks = [
        'open_marker','handover_block','insert_battery','insert_marker_into_cup','lift_ball',
        'pickup_plate','stack_blocks','stack_bowls','straighten_rope','ziploc'


        # easy TASKS
        #"insert_marker_into_cup",
        #"insert_marker_into_cup", "lift_ball", "pickup_plate", "stack_bowls", "straighten_rope",
        # HARD TASKS 
        #"handover_block", "open_marker", "stack_blocks", "ziploc", "insert_battery",
    ]

    # for split in ['train', 'val']:
    #     all_tasks_main(split, tasks)

    # Store instructions as json (can be run independently)
    os.makedirs('../instructions/mobaloha', exist_ok=True)
    # instr_dict = store_instructions(ROOT, tasks)
    instr_dict = {}
    for task in tasks:
        tmp = {"0":[task]}
        instr_dict[task] = tmp
        # print("tmp: ", tmp)
    print("instr_dict: ", instr_dict)
    with open('../instructions/mobaloha/instructions.json', 'w') as fid:
        json.dump(instr_dict, fid)