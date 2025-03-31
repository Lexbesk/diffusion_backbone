import json
import os

from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm

import utils.pytorch3d_transforms as pytorch3d_transforms


ROOT = '/data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D'
STORE_PATH = '/data/group_data/katefgroup/VLA/CALVIN_zarr'
STORE_EVERY = 1  # in keyposes
SUBSAMPLE = 3
IM_SIZE = 160


def convert_rotation(action):
    """Convert Euler angles to wxyz Quarternion."""
    rot = action[..., 3:6]
    rot = torch.as_tensor(rot)
    mat = pytorch3d_transforms.euler_angles_to_matrix(rot, "XYZ")
    quat = pytorch3d_transforms.matrix_to_quaternion(mat)
    quat = quat.numpy()

    return np.concatenate((action[..., :3], quat, action[..., -1:]), -1)


def all_tasks_main(split):
    # All CALVIN episodes
    suffix = 'training' if split == 'train' else 'validation'
    annos = np.load(
        f'{ROOT}/{suffix}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"{STORE_PATH}/{split}.zarr", mode="w") as zarr_file:
        zarr_file.create_dataset(
            "rgb_front",
            shape=(0, 1, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, 1, 3, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "rgb_wrist",
            shape=(0, 1, 3, 84, 84),
            chunks=(STORE_EVERY, 1, 3, 84, 84),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "depth_front",
            shape=(0, 1, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, 1, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "instr_id", shape=(0,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="int"
        )
        zarr_file.create_dataset(
            "proprioception",
            shape=(0, 1, 8),
            chunks=(STORE_EVERY, 1, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "action",
            shape=(0, 16, 8),
            chunks=(STORE_EVERY, 16, 8),
            compressor=compressor,
            dtype="float32"
        )

        # Loop through episodes
        for ann_id, (start, end) in tqdm(enumerate(annos['info']['indx'])):
            rgb_front, rgb_wrist, depth_front, prop = [], [], [], []
            # Each episode is split in multiple files
            for ep_id in range(start, end + 1):
                episode = 'episode_{:07d}.npz'.format(ep_id)
                data = np.load(f'{ROOT}/{suffix}/{episode}')
                rgb_front.append(data['rgb_static'].transpose(2, 0, 1).astype(np.uint8))
                rgb_wrist.append(data['rgb_gripper'].transpose(2, 0, 1).astype(np.uint8))
                depth_front.append(data['depth_static'].astype(np.float16))
                prop.append(convert_rotation(np.concatenate([
                    data['robot_obs'][:3],
                    data['robot_obs'][3:6],  # Euler to quat
                    (data['robot_obs'][[-1]] > 0).astype(np.float32)  # [0, 1]
                ], axis=-1)))

            # Merge
            rgb_front = np.stack(rgb_front)[:, None, :, 20:180, 20:180]
            rgb_wrist = np.stack(rgb_wrist)[:, None]
            depth_front = np.stack(depth_front)[:, None, 20:180, 20:180]
            instr_id = np.array([ann_id] * len(rgb_front))
            prop = np.stack(prop).astype(np.float32)
            actions16 = np.concatenate((prop[1:], np.stack([prop[-1]] * 16)))
            actions16 = np.array([
                actions16[i:i+16] for i in range(len(actions16) - 16 + 1)
            ])[:len(prop)]

            # Write
            zarr_file['rgb_front'].append(rgb_front[:-1][::SUBSAMPLE])
            zarr_file['rgb_wrist'].append(rgb_wrist[:-1][::SUBSAMPLE])
            zarr_file['depth_front'].append(depth_front[:-1][::SUBSAMPLE])
            zarr_file['instr_id'].append(instr_id[:-1][::SUBSAMPLE])
            zarr_file['proprioception'].append(prop[:-1, None][::SUBSAMPLE])
            zarr_file['action'].append(actions16[:-1][::SUBSAMPLE])


def store_instructions(split):
    # All CALVIN episodes
    suffix = 'training' if split == 'train' else 'validation'
    annos = np.load(
        f'{ROOT}/{suffix}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()
    return annos['language']['ann']


if __name__ == "__main__":
    all_tasks_main('train')
    all_tasks_main('val')
    # Store instructions as json (can be run independently)
    os.makedirs('instructions/calvin', exist_ok=True)
    instr = store_instructions('train')
    with open('instructions/calvin/train_instructions.json', 'w') as fid:
        json.dump(instr, fid)
    instr = store_instructions('val')
    with open('instructions/calvin/val_instructions.json', 'w') as fid:
        json.dump(instr, fid)
