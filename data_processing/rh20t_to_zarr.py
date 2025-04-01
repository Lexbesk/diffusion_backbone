import json
import os

from numcodecs import Blosc
import zarr
import numpy as np
from tqdm import tqdm
from ipdb import set_trace as st


ROOT = '/data/group_data/katefgroup/VLA/rh20t/processed/'
STORE_PATH = '/data/user_data/ngkanats/RH20T_zarr/'
STORE_EVERY = 1  # in keyposes
SUBSAMPLE = 5
NCAM = 2  # +1 2D camera
IM_SIZE = 256


def all_tasks_main(split):
    cameras_3d = ['shoulder_left', 'front']
    cam_dict = {
        'shoulder_left': '036422060909',
        'front': '038522062288',
        'hand': '045322071843'
    }
    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"{STORE_PATH}/{split}.zarr", mode="w") as zarr_file:
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
            "rgb2d",
            shape=(0, 1, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, 1, 3, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "proprioception",
            shape=(0, 1, 1, 8),
            chunks=(STORE_EVERY, 1, 1, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "action",
            shape=(0, 16, 1, 8),
            chunks=(STORE_EVERY, 16, 1, 8),
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
            "instr_id", shape=(0,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="uint8"
        )

        # Loop through episodes
        episodes = sorted(os.listdir(os.path.join(ROOT, split)))
        for ep_id, ep in tqdm(enumerate(episodes)):
            data = np.load(f'{ROOT}/{split}/{ep}', allow_pickle=True).item()

            # Loop through keyposes and store:
            # RGB (frames, 3d_cameras, 3, 256, 256)
            rgb = np.stack([
                np.stack([
                    data['rgbds'][i][cam_dict[cam]]['rgb'].astype(np.uint8)
                    if cam_dict[cam] in data['rgbds'][i]
                    else np.zeros((IM_SIZE, IM_SIZE, 3)).astype(np.uint8)
                    for i in np.arange(len(data['rgbds']))[::SUBSAMPLE]
                ])
                for cam in cameras_3d
            ], 1)
            rgb = rgb.transpose(0, 1, 4, 2, 3)

            # Depth (frames, 3d_cameras, 256, 256)
            depth = np.stack([
                np.stack([
                    data['rgbds'][i][cam_dict[cam]]['depth'].astype(np.float16)
                    if cam_dict[cam] in data['rgbds'][i]
                    else np.zeros((IM_SIZE, IM_SIZE)).astype(np.float16)
                    for i in np.arange(len(data['rgbds']))[::SUBSAMPLE]
                ])
                for cam in cameras_3d
            ], 1)

            # RGB (frames, 3d_cameras, 3, 256, 256)
            rgb2d = np.stack([
                np.stack([
                    data['rgbds'][i][cam_dict[cam]]['rgb'].astype(np.uint8)
                    if cam_dict[cam] in data['rgbds'][i]
                    else np.zeros((IM_SIZE, IM_SIZE, 3)).astype(np.uint8)
                    for i in np.arange(len(data['rgbds']))[::SUBSAMPLE]
                ])
                for cam in ['hand',]
            ], 1)
            rgb2d = rgb2d.transpose(0, 1, 4, 2, 3)

            # Proprioception (frames, 1, 1, 8)
            prop = np.concatenate([
                np.stack(data['ee_pose']),
                (np.stack(data['cmds']) > 50.)[:, None].astype(float)
            ], 1).astype(np.float32)  # (frames, 8)
            if len(data['ee_pose']) == 2 * len(data['rgbds']):
                prop = prop[:len(data['rgbds'])]
            prop = prop[:, None, None]
            
            # Action (frames, 16, 1, 8)
            actions16 = np.concatenate((prop[1:], np.stack([prop[-1]] * 16)))
            actions16 = np.stack([
                actions16[i:i+16] for i in range(len(actions16) - 16 + 1)
            ])[:len(prop)]  # (frames, 16, 1, 1, 8)
            actions16 = actions16[:, :, 0]

            # Extrinsics (frames, cameras, 4, 4)
            extrinsics = np.stack([
                data['extrinsics'].get(cam_dict[cam], np.eye(4)).astype(np.float16)
                for cam in cameras_3d
            ])
            extrinsics = np.stack([extrinsics] * len(rgb))

            # Intrinsics (frames, cameras, 3, 3)
            intrinsics = np.stack([
                data['intrinsics'].get(cam_dict[cam], np.eye(3)).astype(np.float16)
                for cam in cameras_3d
            ])
            intrinsics = np.stack([intrinsics] * len(rgb))

            # Task id (frames,)
            instr_id = np.array([ep_id] * len(rgb))
            instr_id = instr_id.astype(np.uint8)

            # Write
            zarr_file['rgb'].append(rgb[:-1])
            zarr_file['depth'].append(depth[:-1])
            zarr_file['rgb2d'].append(rgb2d[:-1])
            zarr_file['proprioception'].append(prop[::SUBSAMPLE][:-1])
            zarr_file['action'].append(actions16[::SUBSAMPLE][:-1])
            zarr_file['extrinsics'].append(extrinsics[:-1])
            zarr_file['intrinsics'].append(intrinsics[:-1])
            zarr_file['instr_id'].append(instr_id[:-1])

            # Checks
            len_ = len(zarr_file['rgb'])
            for key in zarr_file:
                if len(zarr_file[key]) != len_:  # , f'bad key {key}'
                    st()


def store_instructions(split):
    episodes = sorted(os.listdir(os.path.join(ROOT, split)))
    return [
        np.load(f'{ROOT}/{split}/{ep}', allow_pickle=True).item()['language']
        for ep in tqdm(episodes)
    ]


if __name__ == "__main__":
    for split in ['train', 'eval']:
        all_tasks_main(split)
    
    # Store instructions as json (can be run independently)
    os.makedirs('instructions/rh20t', exist_ok=True)
    instr = store_instructions('train')
    with open('instructions/rh20t/train_instructions.json', 'w') as fid:
        json.dump(instr, fid)
    instr = store_instructions('eval')
    with open('instructions/rh20t/val_instructions.json', 'w') as fid:
        json.dump(instr, fid)
