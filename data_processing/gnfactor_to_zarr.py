from pathlib import Path
import pickle
import random

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm


ROOT = '/scratch/Peract_packaged/val'
STORE_PATH = '/data/user_data/ngkanats/GNFactor_zarr/val'
READ_EVERY = 100  # in episodes
STORE_EVERY = 1  # in keyposes
RAND_STORE_EVERY = 1  # in keyposes
IM_SIZE = 256


left_shoulder_camera_extrinsics = np.array([
    [ 1.73648179e-01,  8.92538846e-01,  4.16198105e-01, -1.74999714e-01],
    [ 9.84807789e-01, -1.57378674e-01, -7.33871460e-02, 2.00000003e-01],
    [-1.78813934e-07,  4.22618657e-01, -9.06307697e-01, 1.97999895e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])
left_shoulder_camera_intrinsics = np.array([
    [-351.6771208,    0.       ,  128.       ],
    [   0.       , -351.6771208,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])
right_shoulder_camera_extrinsics = np.array([
    [-1.73648357e-01,  8.92538846e-01,  4.16198105e-01, -1.74997091e-01],
    [ 9.84807789e-01,  1.57378793e-01,  7.33869076e-02, -2.00000003e-01],
    [-1.19209290e-07,  4.22618628e-01, -9.06307697e-01, 1.97999227e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])
right_shoulder_camera_intrinsics = np.array([
    [-351.6771208,    0.       ,  128.       ],
    [   0.       , -351.6771208,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])
wrist_camera_extrinsics = np.array([
    [ 8.34465027e-07,  9.87690389e-01,  1.56421274e-01, 3.04353595e-01],
    [ 9.99999940e-01, -7.15255737e-07,  1.86264515e-07, -6.17044233e-03],
    [ 3.05473804e-07,  1.56421274e-01, -9.87690210e-01, 1.57466102e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])
wrist_camera_intrinsics = np.array([
    [-221.70249591,    0.        ,  128.        ],
    [   0.        , -221.70249591,  128.        ],
    [   0.        ,    0.        ,    1.        ]
])
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
    'left': {
        'extrinsics': left_shoulder_camera_extrinsics,
        'intrinsics': left_shoulder_camera_intrinsics
    },
    'right': {
        'extrinsics': right_shoulder_camera_extrinsics,
        'intrinsics': right_shoulder_camera_intrinsics
    },
    'wrist': {
        'extrinsics': wrist_camera_extrinsics,
        'intrinsics': wrist_camera_intrinsics
    },
    'front': {
        'extrinsics': front_camera_extrinsics,
        'intrinsics': front_camera_intrinsics
    }
}


def inverse_depth(xyz, extrinsics, intrinsics):
    """Convert point cloud in world frame to depth."""
    # Construct camera projection
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]]
    )
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)

    # World space to pixel space
    h, w = xyz.shape[:2]
    xyz = np.concatenate([xyz, np.ones((h, w, 1))], -1)
    xyz = np.reshape(xyz, (h * w, -1)).T
    xyz = np.matmul(np.linalg.inv(cam_proj_mat_inv), xyz)
    xyz = np.reshape(xyz.T, (h, w, -1))[..., :3]
    
    # Isolate the influence of depth
    return xyz[..., -1]


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


def all_tasks_main():
    tasks = [
        "close_jar", "open_drawer", "sweep_to_dustpan_of_size",
        "meat_off_grill", "turn_tap", "slide_block_to_color_target",
        "put_item_in_drawer", "reach_and_drag", "push_buttons",
        "stack_blocks"
    ]
    camera_order = ['left', 'right', 'wrist', 'front']
    task2id = {task: t for t, task in enumerate(tasks)}
    max_episodes_per_task = 20
    variations = range(0, 199)
    # Collect all episodes
    episodes = []
    for task in tasks:
        num_vars = 0
        for var in variations:
            _path = Path(f'{ROOT}/{task}+{var}/')
            if not _path.is_dir():
                continue
            else:
                num_vars += 1

        num_episode_per_var = max_episodes_per_task // num_vars
        task_episodes = []
        for var in variations:
            _path = Path(f'{ROOT}/{task}+{var}/')
            if not _path.is_dir():
                continue
            task_episodes.extend(
                [(task, var, ep) for ep in _path.glob("*.dat")][:num_episode_per_var+1]
            )
        if len(task_episodes) > max_episodes_per_task:
            task_episodes = random.sample(task_episodes, max_episodes_per_task)
        episodes.extend(task_episodes)
    random.shuffle(episodes)

    # Read once to get the number of keyposes
    n_keyposes = 0
    for _, _, ep in tqdm(episodes):
        with open(ep, "rb") as f:
            content = pickle.loads(blosc.decompress(f.read()))
        n_keyposes += len(content[0])

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"{STORE_PATH}.zarr", mode="w") as zarr_file:
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
            shape=(n_keyposes, 1, 8),
            chunks=(STORE_EVERY, 1, 8),
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

        # Read every READ_EVERY
        start = 0
        for s in range(0, len(episodes), READ_EVERY):
            rgb, depth, prop, actions, tids, _vars = [], [], [], [], [], []
            # collect data
            for task, var, ep in tqdm(episodes[s:s + READ_EVERY]):
                with open(ep, "rb") as f:
                    content = pickle.loads(blosc.decompress(f.read()))
                rgb.append((127.5 * (content[1][:, -1:, 0] + 1)).astype(np.uint8))
                batch_depth = np.stack([
                    inverse_depth_batched(
                        content[1][:, c, 1].transpose(0, 2, 3, 1),
                        cameras[cam]['extrinsics'],
                        cameras[cam]['intrinsics']
                    )
                    for c, cam in enumerate(camera_order)
                    if cam == 'front'
                ], 1)
                depth.append(batch_depth.astype(np.float16))
                prop.extend([
                    to_numpy(tens).astype(np.float32) for tens in content[4]
                ])
                actions.extend([
                    to_numpy(tens).astype(np.float32) for tens in content[2]
                ])
                tids.extend([task2id[task]] * len(content[0]))
                _vars.extend([var] * len(content[0]))
            # shuffle
            inds = np.random.permutation(len(_vars))
            # write
            end = start + len(inds)
            zarr_file['rgb'][start:end] = np.concatenate(rgb)[inds]
            zarr_file['depth'][start:end] = np.concatenate(depth)[inds]
            zarr_file['proprioception'][start:end] = np.stack(prop)[inds]
            zarr_file['action'][start:end] = np.stack(
                (np.concatenate(prop), np.concatenate(actions)), 1
            )[inds]
            zarr_file['task_id'][start:end] = np.array(tids)[inds].astype(np.uint8)
            zarr_file['variation'][start:end] = np.array(_vars)[inds].astype(np.uint8)
            start = end


def randomize_order():
    print('Randomizing order')
    with zarr.open_group(f"{STORE_PATH}.zarr", mode="r") as read_file:
        n_keyposes = read_file['variation'].shape[0]
        print(f'found {n_keyposes} keyposes')
        compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
        with zarr.open_group(f"{STORE_PATH}_randomized.zarr", mode="w") as fid:
            fid.create_dataset(
                "rgb",
                shape=(n_keyposes, 1, 3, IM_SIZE, IM_SIZE),
                chunks=(RAND_STORE_EVERY, 1, 3, IM_SIZE, IM_SIZE),
                compressor=compressor,
                dtype="uint8"
            )
            fid.create_dataset(
                "depth",
                shape=(n_keyposes, 1, IM_SIZE, IM_SIZE),
                chunks=(RAND_STORE_EVERY, 1, IM_SIZE, IM_SIZE),
                compressor=compressor,
                dtype="float16"
            )
            fid.create_dataset(
                "task_id", shape=(n_keyposes,), chunks=(RAND_STORE_EVERY,),
                compressor=compressor,
                dtype="uint8"
            )
            fid.create_dataset(
                "variation", shape=(n_keyposes,), chunks=(RAND_STORE_EVERY,),
                compressor=compressor,
                dtype="uint8"
            )
            fid.create_dataset(
                "proprioception",
                shape=(n_keyposes, 1, 8),
                chunks=(RAND_STORE_EVERY, 1, 8),
                compressor=compressor,
                dtype="float32"
            )
            fid.create_dataset(
                "action",
                shape=(n_keyposes, 2, 8),
                chunks=(RAND_STORE_EVERY, 2, 8),
                compressor=compressor,
                dtype="float32"
            )
            # shuffle
            inds = np.random.permutation(n_keyposes)
            # write
            fields = [
                'rgb', 'depth', 'task_id', 'variation',
                'proprioception', 'action'
            ]
            for i, ind in tqdm(enumerate(inds)):
                for field in fields:
                    fid[field][i] = read_file[field][ind]


if __name__ == "__main__":
    all_tasks_main()
    randomize_order()
