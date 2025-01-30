from pathlib import Path
import pickle

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm


ROOT = '/lustre/fsw/portfolios/nvr/users/ngkanatsios/Peract_packaged'
STORE_PATH = '/lustre/fsw/portfolios/nvr/users/ngkanatsios/GNFactor_zarr_custom'
STORE_EVERY = 1  # in keyposes
IM_SIZE = 256


front_camera_extrinsics = np.array([
    [ 1.19209290e-07, -4.22617942e-01, -9.06307936e-01, 1.34999919e+00],
    [-1.00000000e+00, -5.96046448e-07,  1.49011612e-07, 3.71546562e-08],
    [-5.66244125e-07,  9.06307936e-01, -4.22617912e-01, 1.57999933e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])
front_camera_intrinsics = np.array([
    [-351.6771208,    0.       ,  128.       ],
    [   0.       , -351.6771208,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])
cameras = {
    'front': {
        'extrinsics': front_camera_extrinsics,
        'intrinsics': front_camera_intrinsics
    }
}


def inverse_depth_batched(xyz, extrinsics, intrinsics):
    """Convert point cloud in world frame to depth."""
    # Construct camera projection
    C = extrinsics[:3, 3][None].T  # (3, 1)
    R = extrinsics[:3, :3]  # (3, 3)
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)  # (3, 1)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)  # (3, 4)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)  # (3, 4)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]]
    )  # (4, 4)
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)  # (4, 4)

    # World space to pixel space
    b, h, w = xyz.shape[:3]
    xyz = np.concatenate([xyz, np.ones((b, h, w, 1))], -1)
    xyz = np.reshape(xyz, (b * h * w, -1)).T
    xyz = np.matmul(np.linalg.inv(cam_proj_mat_inv), xyz)
    xyz = np.reshape(xyz.T, (b, h, w, -1))[..., :3]
    
    # Isolate the influence of depth
    return xyz[..., -1]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def all_tasks_main(split):
    tasks = [
        "close_jar", "open_drawer", "sweep_to_dustpan_of_size",
        "meat_off_grill", "turn_tap", "slide_block_to_color_target",
        "put_item_in_drawer", "reach_and_drag", "push_buttons",
        "stack_blocks"
    ]
    camera_order = ['left', 'right', 'wrist', 'front']
    task2id = {task: t for t, task in enumerate(tasks)}
    eps_per_task = 20
    variations = range(0, 199)

    # Collect all episodes
    with open(f'eps_{split}.pkl', 'rb') as fid:
        episodes = pickle.load(fid)
    episodes = [
        (ep0, ep1, Path(str(ep2).replace('/data/user_data/ngkanats/', '/lustre/fsw/portfolios/nvr/users/ngkanatsios/')))
        for ep0, ep1, ep2 in episodes
    ]

    # Read once to get the number of keyposes
    n_keyposes = 0
    for _, _, ep in tqdm(episodes):
        with open(ep, "rb") as f:
            content = pickle.loads(blosc.decompress(f.read()))
        n_keyposes += len(content[0])

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"{STORE_PATH}/{split}.zarr", mode="w") as zarr_file:
        zarr_file.create_dataset(
            "rgb",
            shape=(n_keyposes, 1, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, 1, 3, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "depth",
            shape=(n_keyposes, 1, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, 1, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "task_id", shape=(n_keyposes,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "variation", shape=(n_keyposes,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="uint8"
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
            shape=(n_keyposes, 2, 8),
            chunks=(STORE_EVERY, 2, 8),
            compressor=compressor,
            dtype="float32"
        )

        # Loop through episodes
        start = 0
        for task, var, ep in tqdm(episodes):
            # Read
            with open(ep, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            # Map [-1, 1] to [0, 255] uint8
            rgb = (127.5 * (content[1][:, -1:, 0] + 1)).astype(np.uint8)
            # Extract depth from point cloud, faster loading
            depth = np.stack([
                inverse_depth_batched(
                    content[1][:, c, 1].transpose(0, 2, 3, 1),
                    cameras[cam]['extrinsics'],
                    cameras[cam]['intrinsics']
                )
                for c, cam in enumerate(camera_order)
                if cam == 'front'
            ], 1).astype(np.float16)
            # Store current eef pose as well as two previous ones
            prop = np.stack([
                to_numpy(tens).astype(np.float32) for tens in content[4]
            ])
            prop_1 = np.concatenate([prop[:1], prop[:-1]])
            prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
            prop = np.concatenate([prop_2, prop_1, prop], 1)
            # Next keypose (concatenate curr eef to form a "trajectory")
            actions = np.stack([
                to_numpy(tens).astype(np.float32) for tens in content[2]
            ])
            actions = np.concatenate([prop[:, -1:], actions], 1)
            # Task ids and variation ids
            tids = np.array([task2id[task]] * len(content[0])).astype(np.uint8)
            _vars = np.array([var] * len(content[0])).astype(np.uint8)

            # write
            end = start + len(_vars)
            zarr_file['rgb'][start:end] = rgb
            zarr_file['depth'][start:end] = depth
            zarr_file['proprioception'][start:end] = prop
            zarr_file['action'][start:end] = actions
            zarr_file['task_id'][start:end] = tids
            zarr_file['variation'][start:end] = _vars
            start = end


if __name__ == "__main__":
    for split in ['train', 'val']:
        all_tasks_main(split)
