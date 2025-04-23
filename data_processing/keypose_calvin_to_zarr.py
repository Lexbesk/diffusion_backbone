import json
import os

from numcodecs import Blosc
import zarr
import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation as R, Slerp
import torch
from tqdm import tqdm

import utils.pytorch3d_transforms as pytorch3d_transforms
from online_evaluation_calvin.utils_with_calvin import get_env


ROOT = '/data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D'
STORE_PATH = '/data/user_data/ngkanats/zarr_datasets/CALVIN_keypose_zarr'
# ROOT = '/lustre/fsw/portfolios/nvr/users/ngkanatsios/task_ABC_D'
# STORE_PATH = '/lustre/fsw/portfolios/nvr/users/ngkanatsios/zarr_datasets/CALVIN_zarr'
STORE_EVERY = 1  # in keyposes
IM_SIZE = 160


def get_eef_velocity_from_trajectories(trajectories):
    trajectories = np.stack([trajectories[0]] + trajectories, axis=0)
    velocities = trajectories[1:] - trajectories[:-1]

    V = np.linalg.norm(velocities[:, :3], axis=-1)
    W = np.linalg.norm(velocities[:, 3:6], axis=-1)

    velocities = np.concatenate(
        [velocities, [velocities[-1]]],
        # [velocities[[0]], velocities],
        axis=0
    )
    accelerations = velocities[1:] - velocities[:-1]

    A = np.linalg.norm(accelerations[:, :3], axis=-1)

    return V, W, A


def gripper_state_changed(trajectories):
    trajectories = np.stack(
        [trajectories[0]] + trajectories, axis=0
    )
    openess = trajectories[:, -1]
    changed = openess[:-1] != openess[1:]

    return np.where(changed)[0]


def keypoint_discovery(trajectories, buffer_size=5):
    """Determine way point from the trajectories.

    Args:
        trajectories: a list of 1-D np arrays.  Each array is
            7-dimensional (x, y, z, euler_x, euler_y, euler_z, opene).
        stopping_delta: the minimum velocity to determine if the
            end effector is stopped.

    Returns:
        an Integer array indicates the indices of waypoints
    """
    V, W, A = get_eef_velocity_from_trajectories(trajectories)

    # waypoints are local minima of gripper movement
    _local_max_A = argrelextrema(A, np.greater)[0]
    topK = np.sort(A)[::-1][int(A.shape[0] * 0.2)]
    large_A = A[_local_max_A] >= topK
    _local_max_A = _local_max_A[large_A].tolist()

    local_max_A = [_local_max_A.pop(0)]
    for i in _local_max_A:
        if i - local_max_A[-1] >= buffer_size:
            local_max_A.append(i)

    # waypoints are frames with changing gripper states
    gripper_changed = gripper_state_changed(trajectories)
    one_frame_before_gripper_changed = (
        gripper_changed[gripper_changed > 1] - 1
    )

    keyframe_inds = (
        local_max_A +
        gripper_changed.tolist()
        # one_frame_before_gripper_changed.tolist()
    )
    keyframe_inds = np.unique(keyframe_inds)

    keyframes = [trajectories[i] for i in keyframe_inds]

    return keyframes, keyframe_inds


class TrajectoryInterpolator:
    """Interpolate a trajectory to a fixed number of steps."""

    def __init__(self, interpolation_length=50):
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        old_steps = np.linspace(0, 1, len(trajectory))
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Extract components
        pos = trajectory[:, :3]        # (N, 3)
        quat = trajectory[:, 3:7]      # (N, 4), xyzw
        grip = trajectory[:, 7]        # (N,)

        # Interpolate position and gripper linearly or with spline
        interp_pos = CubicSpline(old_steps, pos)(new_steps)
        interp_grip = interp1d(old_steps, grip)(new_steps)

        # SLERP for quaternions
        key_rots = R.from_quat(quat, scalar_first=True)  # (w, x, y, z)
        slerp = Slerp(old_steps, key_rots)
        interp_quat = slerp(new_steps).as_quat()  # shape (L, 4)

        # Convert to tensor
        interpolated = torch.cat([
            torch.tensor(interp_pos, dtype=torch.float32),
            torch.tensor(interp_quat, dtype=torch.float32),
            torch.tensor(interp_grip[:, None], dtype=torch.float32)
        ], dim=-1)

        return interpolated.numpy()


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
    interp = TrajectoryInterpolator(interpolation_length=20)
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
            shape=(0, 20, 8),
            chunks=(STORE_EVERY, 20, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "rel_action",
            shape=(0, 20, 8),
            chunks=(STORE_EVERY, 20, 8),
            compressor=compressor,
            dtype="float32"
        )

        # Loop through episodes
        for ann_id, (start, end) in tqdm(enumerate(annos['info']['indx'])):
            rgb_front, rgb_wrist, depth_front, depth_wrist, ext_wrist = [], [], [], [], []
            prop = []
            # We'll read the data twice, once for proprio and once for images
            for ep_id in range(start, end + 1):
                episode = 'episode_{:07d}.npz'.format(ep_id)
                data = np.load(f'{ROOT}/{suffix}/{episode}')
                # Store proprio in EVERY timestep
                prop.append(convert_rotation(np.concatenate([
                    data['robot_obs'][:3],
                    data['robot_obs'][3:6],  # Euler to quat
                    (data['robot_obs'][-1:] > 0).astype(np.float32)  # [0, 1]
                ], axis=-1)))
            # Keypose discovery
            _, keyframe_inds = keypoint_discovery(prop)
            keyframe_inds = np.concatenate([[0], keyframe_inds, [len(prop) - 1]])
            keyframe_inds = np.array([
                kid for k, kid in enumerate(keyframe_inds)
                if k == len(keyframe_inds) - 1 or keyframe_inds[k + 1] - kid > 3
            ])[:-1]
            # Re-read the data for observations
            for ep_id in keyframe_inds:
                episode = 'episode_{:07d}.npz'.format(start + ep_id)
                data = np.load(f'{ROOT}/{suffix}/{episode}')
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

            # Merge
            rgb_front = np.stack(rgb_front)[:, None, :, 20:180, 20:180]
            rgb_wrist = np.stack(rgb_wrist)[:, None]
            depth_front = np.stack(depth_front)[:, None, 20:180, 20:180]
            depth_wrist = np.stack(depth_wrist)[:, None]
            ext_wrist = np.stack(ext_wrist)
            instr_id = np.array([ann_id] * len(rgb_front))

            # Interpolate proprioception
            prop = np.stack(prop).astype(np.float32)
            next_keyframe_inds = np.concatenate(
                [keyframe_inds[1:], [len(prop) - 1]]
            )
            actions = np.stack([
                interp(prop[i + 1:j + 1])
                for i, j in zip(keyframe_inds, next_keyframe_inds)
            ])
            prop = prop[keyframe_inds]
            # Relative actions
            rel_actions = to_relative_action(
                torch.from_numpy(actions),
                torch.from_numpy(prop)[:, None],
                qform='wxyz'
            )

            # Write
            zarr_file['rgb_front'].append(rgb_front)
            zarr_file['rgb_wrist'].append(rgb_wrist)
            zarr_file['depth_front'].append(depth_front)
            zarr_file['depth_wrist'].append(depth_wrist)
            zarr_file['extrinsics_wrist'].append(ext_wrist)
            zarr_file['instr_id'].append(instr_id)
            zarr_file['proprioception'].append(prop[:, None])
            zarr_file['action'].append(actions)
            zarr_file['rel_action'].append(rel_actions)
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
    with open('instructions/calvin/train_keypose_instructions.json', 'w') as fid:
        json.dump(instr, fid)
    instr = store_instructions('val')
    with open('instructions/calvin/val_keypose_instructions.json', 'w') as fid:
        json.dump(instr, fid)
