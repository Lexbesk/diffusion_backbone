from pathlib import Path
import pickle

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm

from datasets.utils import TrajectoryInterpolator
import utils.pytorch3d_transforms as pytorch3d_transforms


ROOT = '/data/user_data/ngkanats/calvin/packaged_ABC_D/'
STORE_PATH = '/data/user_data/ngkanats/CALVIN_zarr'
STORE_EVERY = 1  # in keyposes
IM_SIZE = 160


def inverse_depth_batched(pcds, viewMatrix):
    """Convert point cloud in world frame to depth."""
    # pcds is (b, 3, h, w)
    T_cam_world = np.array(viewMatrix).reshape((4, 4)).T

    b, _, h, w = pcds.shape
    pcds = pcds.transpose(1, 0, 2, 3).reshape(3, -1)  # 3 B*H*W
    pcds = np.concatenate((pcds, np.ones_like(pcds)[:1]))

    pcds = T_cam_world @ pcds  # 4 B*H*W`
    return -pcds[2].reshape(b, h, w)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def convert_rotation(action):
    """Convert Euler angles to wxyz Quarternion."""
    rot = action[..., 3:6]
    rot = torch.as_tensor(rot)
    mat = pytorch3d_transforms.euler_angles_to_matrix(rot, "XYZ")
    quat = pytorch3d_transforms.matrix_to_quaternion(mat)
    quat = quat.numpy()

    return np.concatenate((action[..., :3], quat, action[..., -1:]), -1)


def all_tasks_main(split):
    tasks = ["A", "B", "C"] if split == 'train' else ['D']
    suffix = 'training' if split == 'train' else 'validation'
    camera_order = ['front', 'wrist']
    _interpolate_traj = TrajectoryInterpolator(True, 20)

    # Collect all episodes
    episodes = []
    for task in tasks:
        _path = Path(f'{ROOT}/{suffix}/{task}/')
        episodes.extend(_path.glob("*.dat"))

    # Read once to get the number of keyposes
    n_keyposes = 0
    for ep in tqdm(episodes):
        with open(ep, "rb") as f:
            content = pickle.loads(blosc.decompress(f.read()))
        n_keyposes += len(content[0])

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"{STORE_PATH}/{split}.zarr", mode="w") as zarr_file:
        ncam = 2
        zarr_file.create_dataset(
            "rgb",
            shape=(n_keyposes, ncam, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, ncam, 3, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "pcd",
            shape=(n_keyposes, ncam, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, ncam, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "instr_id", shape=(n_keyposes,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="int"
        )
        zarr_file.create_dataset(
            "proprioception",
            shape=(n_keyposes, 3, 8),
            chunks=(STORE_EVERY, 3, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "action",
            shape=(n_keyposes, 20, 8),
            chunks=(STORE_EVERY, 20, 8),
            compressor=compressor,
            dtype="float32"
        )

        # Loop through episodes
        start = 0
        for ep in tqdm(episodes):
            with open(ep, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            # Map [-1, 1] to [0, 255] uint8 and crop
            rgb = (
                127.5 * (content[1][:, :, 0, :, 20:180, 20:180] + 1)
            ).astype(np.uint8)
            # Point cloud
            pcd = content[1][:, :, 1, :, 20:180, 20:180].astype(np.float16)
            # Store current eef pose as well as two previous ones
            prop = np.stack([
                convert_rotation(to_numpy(tens)).astype(np.float32)
                for tens in content[4]
            ])
            prop_1 = np.concatenate([prop[:1], prop[:-1]])
            prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
            prop = np.concatenate([prop_2, prop_1, prop], 1)
            # Trajectories
            actions = np.stack([
                convert_rotation(_interpolate_traj(torch.tensor(item)).numpy())
                for item in content[5]
            ]).astype(np.float32)
            # Language indices
            instr_ids = np.array([content[6][0]] * len(rgb)).astype(int)

            # write
            end = start + len(rgb)
            zarr_file['rgb'][start:end] = rgb
            zarr_file['pcd'][start:end] = pcd
            zarr_file['proprioception'][start:end] = prop
            zarr_file['action'][start:end] = actions
            zarr_file['instr_id'][start:end] = instr_ids
            start = end


if __name__ == "__main__":
    all_tasks_main('train')
    all_tasks_main('val')
