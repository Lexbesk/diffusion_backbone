from collections import defaultdict
import json
import pickle

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from ipdb import set_trace as st


# ROOT = '/scratch/Peract_packaged/'
STORE_PATH = '/data/group_data/katefgroup/VLA/data/training_data'
READ_EVERY = 50  # in episodes
STORE_EVERY = 1  # in keyposes
NCAM = 1
IM_SIZE = 256


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def all_tasks_main(split):
    annotation_files = [
        "/data/group_data/katefgroup/VLA/data/tower3_full_annotations.json"
    ]
    split_ratio = 0.8

    n_examples = 0
    all_annos = defaultdict(list)
    for fname in annotation_files:
        with open(fname, 'r') as fid:
            _annos = json.load(fid)
        all_demos = _annos.keys()
        # sort by demo number where name is 'name_X'
        all_demos = sorted(all_demos, key=lambda x: int(x.split('_')[-1]))
        split_point = int(split_ratio * len(all_demos))
        if split == 'train':
            target_demos = all_demos[:split_point]
        else:
            target_demos = all_demos[split_point:]
        for task in target_demos:
            all_annos['images'].extend([
                f'{_annos[task]["image_path"]}/{_n}'
                for _n in _annos[task]['images']
            ])
            all_annos['action_path'].extend(
                [_annos[task]['action_path']] * len(_annos[task]['images'])
            )
            all_annos['action_intervals'].extend(_annos[task]['action_intervals'])
            all_annos['subgoals'].extend(_annos[task]['subgoals'])
            n_examples += len(_annos[task]['images'])

    # save all_annos['subgoals'] as a list in a seperate file
    np.savez(f'{STORE_PATH}/{split}_subgoals.npz', subgoals=all_annos['subgoals'])

    # lets make it so we are predicting a fixed sized action chunk
    n_keyposes = 16
    
    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"{STORE_PATH}{split}.zarr", mode="w") as zarr_file:
        zarr_file.create_dataset(
            "rgb",
            shape=(n_examples, NCAM, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, NCAM, 3, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "depth",
            shape=(n_examples, NCAM, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, NCAM, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "proprioception",
            shape=(n_examples, 1, 8),
            chunks=(STORE_EVERY, 1, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "action",
            shape=(n_examples, n_keyposes, 8),
            chunks=(STORE_EVERY, n_keyposes, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "lang_id", shape=(n_examples,),
            chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="int"
        )

        # Read every READ_EVERY
        for s in range(n_examples):
            # collect data
            # rgb
            img_path = all_annos['images'][s]
            rgb = Image.open(img_path).convert('RGB').resize((IM_SIZE, IM_SIZE))
            rgb = np.array(rgb).astype(np.uint8).transpose(2, 0, 1)[None]
            zarr_file['rgb'][s] = rgb
            # depth
            depth_path = img_path.replace('front_rgb', 'front_depth')
            depth = Image.open(depth_path).convert('L').resize((IM_SIZE, IM_SIZE))
            depth = np.array(depth).astype(np.float16)[None]
            zarr_file['depth'][s] = depth
            # actions
            data_path = all_annos['action_path'][s].replace(
                '_actions.npz', '.npz'
            )
            actions = np.load(data_path)
            action_interval = all_annos['action_intervals'][s]
            actions = actions[
                action_interval[0]
                :min(action_interval[1], action_interval[0]+16)
            ]
            if len(actions) < 16:
                # pad with repeat of last action
                actions = np.concatenate([
                    actions,
                    np.repeat(actions[-1:], 16-len(actions), axis=0)
                ]).astype(np.float32)
            zarr_file['proprioception'][s] = actions[:1]
            zarr_file['action'][s] = actions
            # annotation id
            zarr_file['lang_id'][s] = s


if __name__ == "__main__":
    for split in ['train', 'val']:
        all_tasks_main(split)
