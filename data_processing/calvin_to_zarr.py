from pathlib import Path
import pickle
import random

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm

import utils.pytorch3d_transforms as pytorch3d_transforms


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
    """Convert Euler angles to Quarternion."""
    rot = action[..., 3:6]
    rot = torch.as_tensor(rot)
    mat = pytorch3d_transforms.euler_angles_to_matrix(rot, "XYZ")
    quat = pytorch3d_transforms.matrix_to_quaternion(mat)
    quat = quat.numpy()

    return np.concatenate((action[..., :3], quat, action[..., -1:]), -1)


def all_tasks_main():
    tasks = ["A", "B", "C"]
    camera_order = ['left', 'right', 'wrist', 'front']
    task2id = {task: t for t, task in enumerate(tasks)}
    # Collect all episodes
    episodes = []
    for task in tasks:
        _path = Path(f'{ROOT}/{task}/')
        episodes.extend([
            (task, _id, ep) for _id, ep in enumerate(_path.glob("*.dat"))
        ])
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
        ncam = 2
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
            "task_id", shape=(n_keyposes,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "episode_id", shape=(n_keyposes,), chunks=(STORE_EVERY,),
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
                rgb.append((
                    127.5 * (content[1][:, :, 0, :, 20:180, 20:180] + 1)
                ).astype(np.uint8))
                batch_depth = np.stack([
                    inverse_depth_batched(
                        content[1][:, c, 1, :, 20:180, 20:180],
                        cameras[cam]
                    )
                    for c, cam in enumerate(camera_order)
                ], 1)
                depth.append(batch_depth.astype(np.float16))
                prop.extend([
                    convert_rotation(to_numpy(tens)).astype(np.float32)
                    for tens in content[4]
                ])
                actions.extend([
                    convert_rotation(to_numpy(tens)).astype(np.float32)
                    for tens in content[2]
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
            zarr_file['episode_id'][start:end] = np.array(_vars)[inds].astype(np.uint8)
            start = end


def randomize_order():
    print('Randomizing order')
    with zarr.open_group(f"{STORE_PATH}.zarr", mode="r") as read_file:
        n_keyposes = read_file['variation'].shape[0]
        print(f'found {n_keyposes} keyposes')
        compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
        with zarr.open_group(f"{STORE_PATH}_randomized.zarr", mode="w") as fid:
            ncam = 2
            fid.create_dataset(
                "rgb",
                shape=(n_keyposes, ncam, 3, IM_SIZE, IM_SIZE),
                chunks=(RAND_STORE_EVERY, ncam, 3, IM_SIZE, IM_SIZE),
                compressor=compressor,
                dtype="uint8"
            )
            fid.create_dataset(
                "depth",
                shape=(n_keyposes, ncam, IM_SIZE, IM_SIZE),
                chunks=(RAND_STORE_EVERY, ncam, IM_SIZE, IM_SIZE),
                compressor=compressor,
                dtype="float16"
            )
            fid.create_dataset(
                "task_id", shape=(n_keyposes,), chunks=(RAND_STORE_EVERY,),
                compressor=compressor,
                dtype="uint8"
            )
            fid.create_dataset(
                "episode_id", shape=(n_keyposes,), chunks=(RAND_STORE_EVERY,),
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
                'rgb', 'depth', 'task_id', 'episode_id',
                'proprioception', 'action'
            ]
            for i, ind in tqdm(enumerate(inds)):
                for field in fields:
                    fid[field][i] = read_file[field][ind]


if __name__ == "__main__":
    all_tasks_main()
    randomize_order()
