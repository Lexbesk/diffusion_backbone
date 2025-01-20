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
STORE_PATH = '/data/user_data/ngkanats/isaac_zarr'
READ_EVERY = 1000  # in episodes
STORE_EVERY = 1  # in keyposes
RAND_STORE_EVERY = 1  # in keyposes
IM_SIZE = 128
ILEN = 25  # trajectory interpolation length


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


def all_tasks_main():
    """
    [
        img (kp, nc, 128, 128, 3),
        seg (kp, nc, 128, 128),
        pcd (kp, nc, 128, 128, 3),
        dict {obj_name: seg_id},
        content[4]['robot']['panda_hand'] (kp, 4, 4)
        list of trajectories [(n_i, 8)],
        boolean (must be True)
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
    tlen = 0
    for ep in tqdm(episodes):
        with open(ep, "rb") as f:
            content = pickle.loads(blosc.decompress(f.read()))
        if content[-1] and len(content[0]) == 2:
            n_keyposes += len(content[0])
            tlen += sum(len(it) for it in content[5])

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"{STORE_PATH}.zarr", mode="w") as zarr_file:
        zarr_file.create_dataset(
            "rgb",
            shape=(n_keyposes, ncam, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, ncam, 3, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "seg",
            shape=(n_keyposes, ncam, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, ncam, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="bool"
        )
        zarr_file.create_dataset(
            "pcd",
            shape=(n_keyposes, ncam, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, ncam, 3, IM_SIZE, IM_SIZE),
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

        # Read every READ_EVERY
        start = 0
        for s in range(0, len(episodes), READ_EVERY):
            rgb, seg, pcd, prop, actions = [], [], [], [], []
            # collect data
            for ep in tqdm(episodes[s:s + READ_EVERY]):
                with open(ep, "rb") as f:
                    content = pickle.loads(blosc.decompress(f.read()))
                if not content[-1] or len(content[0]) != 2:
                    continue
                rgb.append(content[0].astype(np.uint8).transpose(0, 1, 4, 2, 3))
                seg.append(content[1] == content[3]['/world/obj0'])
                pcd.append(content[2].astype(np.float16).transpose(0, 1, 4, 2, 3))
                prop.extend([_t[0][None].astype(np.float32) for _t in content[5]])
                actions.extend([traj_interp(_t).astype(np.float32) for _t in content[5]])
            # write
            end = start + len(actions)
            zarr_file['rgb'][start:end] = np.concatenate(rgb)
            zarr_file['seg'][start:end] = np.concatenate(seg)
            zarr_file['pcd'][start:end] = np.concatenate(pcd)
            zarr_file['proprioception'][start:end] = np.stack(prop)
            zarr_file['action'][start:end] = np.stack(actions)
            start = end


if __name__ == "__main__":
    all_tasks_main()
