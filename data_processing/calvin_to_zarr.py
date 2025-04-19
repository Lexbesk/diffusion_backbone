import json
import os

from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm

import utils.pytorch3d_transforms as pytorch3d_transforms
from online_evaluation_calvin.utils_with_calvin import get_env


ROOT = '/data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D'
STORE_PATH = '/data/user_data/ngkanats/zarr_datasets/CALVIN_zarr'
# ROOT = '/lustre/fsw/portfolios/nvr/users/ngkanatsios/task_ABC_D'
# STORE_PATH = '/lustre/fsw/portfolios/nvr/users/ngkanatsios/zarr_datasets/CALVIN_zarr'
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


def to_relative_action(actions, anchor_actions, qform='xyzw'):
    # actions: (..., N, 8), anchor_actions: (..., 1, 8)
    assert actions.shape[-1] == 8

    prev = torch.cat([anchor_actions, actions], dim=-2)[..., :-1, :]
    rel_pos = actions[..., :3] - prev[..., :3]  # (N, 3)

    if qform == 'xyzw':
        # pytorch3d takes wxyz quaternion, the input is xyzw
        rel_orn = pytorch3d_transforms.quaternion_multiply(
            actions[..., [6, 3, 4, 5]],
            pytorch3d_transforms.quaternion_invert(prev[..., [6,3,4,5]])
        )[..., [1, 2, 3, 0]]
    elif qform == 'wxyz':
        # pytorch3d takes wxyz quaternion, the input is xyzw
        rel_orn = pytorch3d_transforms.quaternion_multiply(
            actions[..., 3:7],
            pytorch3d_transforms.quaternion_invert(prev[..., 3:7])
        )
    else:
        assert False

    gripper = actions[..., -1:]
    rel_actions = torch.concat([rel_pos, rel_orn, gripper], dim=-1)

    return rel_actions.numpy()


def all_tasks_main(env, split):
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
            "depth_wrist",
            shape=(0, 1, 84, 84),
            chunks=(STORE_EVERY, 1, 84, 84),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "extrinsics_wrist",
            shape=(0, 4, 4),
            chunks=(STORE_EVERY, 4, 4),
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
            shape=(0, 12, 8),
            chunks=(STORE_EVERY, 12, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "rel_action",
            shape=(0, 12, 8),
            chunks=(STORE_EVERY, 12, 8),
            compressor=compressor,
            dtype="float32"
        )

        # Loop through episodes
        for ann_id, (start, end) in tqdm(enumerate(annos['info']['indx'])):
            rgb_front, rgb_wrist, depth_front, depth_wrist, ext_wrist = [], [], [], [], []
            prop = []
            # Each episode is split in multiple files
            for ep_id in range(start, end + 1):
                episode = 'episode_{:07d}.npz'.format(ep_id)
                data = np.load(f'{ROOT}/{suffix}/{episode}')
                # Every other SUBSAMPLE, store observations
                if (ep_id - int(start)) % SUBSAMPLE == 0:
                    rgb_front.append(data['rgb_static'].transpose(2, 0, 1).astype(np.uint8))
                    rgb_wrist.append(data['rgb_gripper'].transpose(2, 0, 1).astype(np.uint8))
                    depth_front.append(data['depth_static'].astype(np.float16))
                    depth_wrist.append(data['depth_gripper'].astype(np.float16))
                    # additional logic for storing wrist camera extrinsics
                    env.reset(
                        robot_obs=data['robot_obs'],
                        scene_obs=data['scene_obs']
                    )
                    ext_wrist.append(np.linalg.inv(
                        np.array(env.cameras[1].view_matrix).reshape((4, 4)).T
                    ).astype(np.float16))
                # Store proprio in EVERY timestep
                prop.append(convert_rotation(np.concatenate([
                    data['robot_obs'][:3],
                    data['robot_obs'][3:6],  # Euler to quat
                    (data['robot_obs'][[-1]] > 0).astype(np.float32)  # [0, 1]
                ], axis=-1)))

            # Merge
            rgb_front = np.stack(rgb_front)[:, None, :, 20:180, 20:180]
            rgb_wrist = np.stack(rgb_wrist)[:, None]
            depth_front = np.stack(depth_front)[:, None, 20:180, 20:180]
            depth_wrist = np.stack(depth_wrist)[:, None]
            ext_wrist = np.stack(ext_wrist)
            instr_id = np.array([ann_id] * len(rgb_front))
            prop = np.stack(prop).astype(np.float32)
            actions12 = np.concatenate((prop[1:], np.stack([prop[-1]] * 12)))
            actions12 = np.array([
                actions12[i:i+12] for i in range(len(actions12) - 12 + 1)
            ])[:len(prop)]
            # Relative actions
            rel_actions = to_relative_action(
                torch.from_numpy(actions12),
                torch.from_numpy(prop)[:, None],
                qform='wxyz'
            )

            # Write
            zarr_file['rgb_front'].append(rgb_front[:-1])
            zarr_file['rgb_wrist'].append(rgb_wrist[:-1])
            zarr_file['depth_front'].append(depth_front[:-1])
            zarr_file['depth_wrist'].append(depth_wrist[:-1])
            zarr_file['extrinsics_wrist'].append(ext_wrist[:-1])
            zarr_file['instr_id'].append(instr_id[:-1])
            zarr_file['proprioception'].append(prop[::SUBSAMPLE][:-1, None])
            zarr_file['action'].append(actions12[::SUBSAMPLE][:-1])
            zarr_file['rel_action'].append(rel_actions[::SUBSAMPLE][:-1])
            assert all(
                zarr_file['action'].shape[0] == zarr_file[key].shape[0]
                for key in zarr_file.keys()
            )


def store_instructions(split):
    # All CALVIN episodes
    suffix = 'training' if split == 'train' else 'validation'
    annos = np.load(
        f'{ROOT}/{suffix}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()
    return annos['language']['ann']


if __name__ == "__main__":
    env = get_env(
        "online_evaluation_calvin/configs/merged_config_val_abc_d.yaml",
        show_gui=False
    )
    all_tasks_main(env, 'train')
    env.close()
    del env
    env = get_env(
        "online_evaluation_calvin/configs/merged_config_val_abc_d.yaml",
        show_gui=False
    )
    all_tasks_main(env, 'val')
    env.close()
    del env
    # Store instructions as json (can be run independently)
    os.makedirs('instructions/calvin', exist_ok=True)
    instr = store_instructions('train')
    with open('instructions/calvin/train_instructions.json', 'w') as fid:
        json.dump(instr, fid)
    instr = store_instructions('val')
    with open('instructions/calvin/val_instructions.json', 'w') as fid:
        json.dump(instr, fid)
