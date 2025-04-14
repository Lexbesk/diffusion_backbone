import json
import os
import pickle

from numcodecs import Blosc
import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    store_instructions
)


ROOT = '/data/group_data/katefgroup/VLA/voxactb_raw/'
STORE_PATH = '/data/group_data/katefgroup/VLA/zarr_datasets/voxactb_zarr/'
STORE_EVERY = 1  # in keyposes
NCAM = 3
IM_SIZE = 128
DEPTH_SCALE = 2**24 - 1


class Arm:

    def __init__(self, gripper_pose, gripper_open, joint_velocities):
        self.gripper_pose = gripper_pose
        self.gripper_open = gripper_open
        self.joint_velocities = joint_velocities


def all_tasks_main(split, tasks):
    cameras = ["front", "wrist", "wrist2"]
    task2id = {task: t for t, task in enumerate(tasks)}

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
        for task in tasks:
            print(task)
            task_folder = f'{ROOT}/{split}/{task}/all_variations/episodes'
            episodes = sorted(os.listdir(task_folder))
            for ep in tqdm(episodes):
                # Read low-dim file from RLBench
                ld_file = f"{task_folder}/{ep}/low_dim_obs.pkl"
                with open(ld_file, 'rb') as f:
                    demo = pickle.load(f)
                # Voxact-b format to peract2
                for i in range(len(demo)):
                    demo[i].left = Arm(
                        demo[i].gripper_left_pose,
                        demo[i].gripper_left_open,
                        demo[i].joint_positions_left
                    )
                    demo[i].right = Arm(
                        demo[i].gripper_right_pose,
                        demo[i].gripper_right_open,
                        demo[i].joint_positions_right
                    )

                # Keypose discovery
                key_frames = keypoint_discovery(demo, bimanual=True)
                key_frames.insert(0, 0)

                # Loop through keyposes and store:
                # RGB (keyframes, cameras, 3, 256, 256)
                rgb = np.stack([
                    np.stack([
                        np.array(Image.open(
                            f"{task_folder}/{ep}/{cam}_rgb/{k}.png"
                        ))
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])
                rgb = rgb.transpose(0, 1, 4, 2, 3)

                # Depth (keyframes, cameras, 256, 256)
                depth_list = []
                for k in key_frames[:-1]:
                    cam_d = []
                    for cam in cameras:
                        depth = image_to_float_array(Image.open(
                            f"{task_folder}/{ep}/{cam}_depth/{k}.png"
                        ), DEPTH_SCALE)
                        near = demo[k].misc[f'{cam}_camera_near']
                        far = demo[k].misc[f'{cam}_camera_far']
                        depth = near + depth * (far - near)
                        cam_d.append(depth)
                    depth_list.append(np.stack(cam_d).astype(np.float16))
                depth = np.stack(depth_list)

                # Proprioception (keyframes, 3, 2, 8)
                states = np.stack([np.concatenate([
                    demo[k].left.gripper_pose,
                    [demo[k].left.gripper_open],
                    demo[k].right.gripper_pose,
                    [demo[k].right.gripper_open]
                ]) for k in key_frames]).astype(np.float32)
                # Store current eef pose as well as two previous ones
                prop = states[:-1]
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1)
                prop = prop.reshape(len(prop), 3, 2, 8)

                # Action (keyframes, 3, 2, 8)
                actions = states[1:].reshape(len(states[1:]), 1, 2, 8)

                # Extrinsics (keyframes, cameras, 4, 4)
                extrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_extrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])

                # Intrinsics (keyframes, cameras, 3, 3)
                intrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_intrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])

                # Task id (keyframes,)
                task_id = np.array([task2id[task]] * len(key_frames[:-1]))
                task_id = task_id.astype(np.uint8)

                # Variation (keyframes,)
                with open(f"{task_folder}/{ep}/variation_number.pkl", 'rb') as f:
                    var_ = pickle.load(f)
                var_ = np.array([int(var_)] * len(key_frames[:-1]))
                var_ = var_.astype(np.uint8)

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
        'hand_over_item',
        'open_drawer',
        'open_jar',
        'put_item_in_drawer'
    ]
    # Create zarr data
    for split in ['train', 'val']:
        all_tasks_main(split, tasks)
    # Store instructions as json (can be run independently)
    os.makedirs('instructions/voxactb', exist_ok=True)
    instr_dict = store_instructions(ROOT, tasks)
    with open('instructions/voxactb/instructions.json', 'w') as fid:
        json.dump(instr_dict, fid)
