from pathlib import Path
import pickle
import random

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import torch
from tqdm import tqdm


ROOT = 'observations'
STORE_EVERY = 1  # in keyposes
IM_SIZE = 128
ILEN = 10  # trajectory interpolation length


def backproject_gym_depth(depth_map, proj_matrix):
    fx = 2.0 / proj_matrix[0, 0]
    fy = 2.0 / proj_matrix[1, 1]
    x, y = np.meshgrid(
        [x for x in range(depth_map.shape[1])],
        [y for y in range(depth_map.shape[0])],
    )
    input_x = x.astype(np.float32)
    input_y = y.astype(np.float32)
    z = depth_map
    
    input_x -= depth_map.shape[1] // 2
    input_y -= depth_map.shape[0] // 2
    input_x /= depth_map.shape[1]
    input_y /= depth_map.shape[0]

    output_x = z * fx * input_x
    output_y = z * fy * input_y

    return np.stack((output_x, output_y, z), -1)


def backproject_gym_depth_torch(depth_map, proj_matrix):
    fx = 2.0 / proj_matrix[0, 0]
    fy = 2.0 / proj_matrix[1, 1]
    x, y = torch.meshgrid(
        torch.arange(depth_map.shape[1]),
        torch.arange(depth_map.shape[0])
    )
    input_x = x.T.float()
    input_y = y.T.float()
    z = depth_map
    
    input_x -= depth_map.shape[1] // 2
    input_y -= depth_map.shape[0] // 2
    input_x /= depth_map.shape[1]
    input_y /= depth_map.shape[0]

    output_x = z * fx * input_x
    output_y = z * fy * input_y

    return torch.stack((output_x, output_y, z), -1)


def normalise_quat(x):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)


class TrajectoryInterpolator:
    """Interpolate a trajectory to have fixed length."""

    def __init__(self, use=False, interpolation_length=50):
        self._use = use
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        if not self._use:
            return trajectory

        # Calculate the current number of steps
        old_num_steps = len(trajectory)

        # Create a 1D array for the old and new steps
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Interpolate
        resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        interpolator = CubicSpline(old_steps, trajectory[:, :-1])
        resampled[:, :-1] = interpolator(new_steps)
        last_interpolator = interp1d(old_steps, trajectory[:, -1])
        resampled[:, -1] = last_interpolator(new_steps)

        resampled = torch.tensor(resampled)
        if trajectory.shape[1] == 8:
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        elif trajectory.shape[1] == 16:
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
            resampled[:, 11:15] = normalise_quat(resampled[:, 11:15])
        return resampled.numpy()


class Isaac_D2C:

    def __init__(self):
        pass

    @staticmethod
    def backproject_gym_depth(depth_map, proj_matrix):
        fx = 2.0 / proj_matrix[..., 0, 0]  # (B, nc)
        fy = 2.0 / proj_matrix[..., 1, 1]  # (B, nc)
        x, y = torch.meshgrid(
            torch.arange(depth_map.shape[-1]),
            torch.arange(depth_map.shape[-2])
        )
        input_x = x.T.to(depth_map.device).half()
        input_y = y.T.to(depth_map.device).half()
        z = depth_map
        
        input_x -= depth_map.shape[-1] // 2
        input_y -= depth_map.shape[-2] // 2
        input_x /= depth_map.shape[-1]
        input_y /= depth_map.shape[-2]

        output_x = z * fx[..., None, None] * input_x[None, None]
        output_y = z * fy[..., None, None] * input_y[None, None]

        return torch.stack((output_x, output_y, z), -1)

    def __call__(self, gym_depth, proj_matrix, extrinsics, xyz_image=None):
        """
        gym_depth: (B, 2, H, W),
        proj_matrix: (B, 2, 4, 4),
        extrinsics: (B, 2, 4, 4),
        xyz_image: (B, 2, H, W, 3)
        """
        # From gym_depth to camera frame
        pcd = self.backproject_gym_depth(gym_depth, proj_matrix)  # B nc H W 3
        if xyz_image is not None:
            assert torch.allclose(pcd, xyz_image)
        # From camera frame to world frame
        _, _, h, w, _ = pcd.shape
        pcd = pcd.reshape(len(pcd), 2, -1, 3)  # (B, nc, H*W, 3)
        pcd = torch.cat((pcd, torch.ones_like(pcd)[..., :1]), -1)
        rz = torch.tensor([  # rotation matrix 90 degrees around z-axis
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).to(pcd.device)[None, None].half()
        ry = torch.tensor([  # rotation matrix 90 degrees around y-axis
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ]).to(pcd.device)[None, None].half()
        result = torch.matmul(
            extrinsics.half() @ ry @ rz,
            pcd.permute(0, 1, 3, 2).half()
        )
        result = result.permute(0, 1, 3, 2)[..., :3]
        result = result.reshape(len(result), 2, h, w, 3)
        return result.permute(0, 1, 4, 2, 3)


def all_tasks_main():
    """
    [
        rgb: (N, 2, H, W, 3),
        gym_depth: (N, 2, H, W),
        segmentation: (N, 2, H, W),
        proj_matrix: (N, 2, 4, 4),
        xyz_image: (N, 2, H, W, 3),
        extrinsics: (N, 2, 4, 4),
        label_dict: dict {obj_id: int},
        rigid_poses: dict,
        eef_pose: (N, 3+4+1),
        success: bool
    ]
    """
    ncam = 2
    camera_order = ['wrist', 'front']
    traj_interp = TrajectoryInterpolator(use=True, interpolation_length=ILEN)
    # Collect all episodes
    episodes = list(Path(ROOT).glob("*.dat"))
    random.shuffle(episodes)

    # Read once to get the number of keyposes
    n_keyposes = 0
    for ep in tqdm(episodes):
        with open(ep, "rb") as f:
            content = pickle.loads(blosc.decompress(f.read()))
        if content[-1]:
            n_keyposes += len(content[0])

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"train.zarr", mode="w") as zarr_file:
        zarr_file.create_dataset(
            "rgb",
            shape=(n_keyposes, ncam, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, ncam, 3, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "depth",
            shape=(n_keyposes, ncam, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, ncam, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "seg",
            shape=(n_keyposes, ncam, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, ncam, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="bool"
        )
        zarr_file.create_dataset(
            "proj_matrix",
            shape=(n_keyposes, ncam, 4, 4),
            chunks=(STORE_EVERY, ncam, 4, 4),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "extrinsics",
            shape=(n_keyposes, ncam, 4, 4),
            chunks=(STORE_EVERY, ncam, 4, 4),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "proprioception",
            shape=(n_keyposes, 1, 8),
            chunks=(STORE_EVERY, 1, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "action",
            shape=(n_keyposes, ILEN, 8),
            chunks=(STORE_EVERY, ILEN, 8),
            compressor=compressor,
            dtype="float32"
        )

        # Write
        start = 0
        for ep in tqdm(episodes):
            with open(ep, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            if not content[-1]:
                continue
            end = start + len(content[0])
            zarr_file['rgb'][start:end] = content[0].astype(np.uint8).transpose(0, 1, 4, 2, 3)
            zarr_file['depth'][start:end] = content[1].astype(np.float16)
            zarr_file['seg'][start:end] = content[2] == content[6]['/world/obj0']
            zarr_file['proj_matrix'][start:end] = content[3].astype(np.float16)
            zarr_file['extrinsics'][start:end] = content[5].astype(np.float16)
            zarr_file['proprioception'][start:end] = np.stack([
                _t[0][None].astype(np.float32) for _t in content[8]
            ])
            zarr_file['action'][start:end] = np.stack([
                traj_interp(_t).astype(np.float32) for _t in content[8]
            ])
            start = end


if __name__ == "__main__":
    all_tasks_main()
