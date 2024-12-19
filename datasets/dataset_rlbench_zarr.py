import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch
import zarr
from zarr.storage import DirectoryStore
from zarr import LRUStoreCache

from .dataset_base import BaseDataset


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.as_tensor(x)


def read_zarr_with_cache(fname):
    # Configure the underlying store
    store = DirectoryStore(fname)

    # Wrap the store with a cache
    cached_store = LRUStoreCache(store, max_size=16 * 2**30)  # 16 GB cache

    # Open Zarr file with caching
    return zarr.open_group(cached_store, mode="r")


class RLBenchDataset(BaseDataset):
    """RLBench dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,
        precompute_instruction_encodings,
        relative_action=False  # whether to return relative actions
    ):
        if isinstance(root, (Path, str)):
            root = [Path(root)]

        super().__init__(
            root=root,
            training=False,
            image_rescale=(1.0, 1.0),
            relative_action=relative_action,
            color_aug=False
        )

        # Load instructions
        self._precompute_instr_encs = precompute_instruction_encodings
        if instructions:
            if precompute_instruction_encodings:
                self._instructions = pickle.load(open(instructions, "rb"))
            else:
                self._instructions = json.load(open(instructions))
        else:
            instructions = None

        # Load all annotations lazily
        self.annos = read_zarr_with_cache(self._root[0])

    def __getitem__(self, idx):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        # Split RGB and XYZ
        rgbs = to_tensor(self.annos['rgb'][idx])
        pcds = to_tensor(self.annos['depth'][idx])

        if self._color_aug is not None:
            rgbs = self._color_aug(rgbs)

        # Sample one instruction feature
        t_ = int(self.annos['task_id'][idx])
        v_ = int(self.annos['variation'][idx])
        task = self.tasks[t_]
        if self._precompute_instr_encs:
            if self._instructions:
                instr = random.choice(self._instructions[task][v_])
            else:
                instr = torch.zeros((53, 512))
        else:
            if self._instructions:
                instr = [random.choice(self._instructions[task][str(v_)])]
            else:
                instr = [""]

        # Get gripper tensors for respective frame ids
        action = to_tensor(self.annos['action'][idx])
        gripper_history = to_tensor(self.annos['proprioception'][idx])

        ret_dict = {
            "task": [task],
            "instr": instr,  # [str] or tensor(53, 512)
            "rgbs": rgbs,  # tensor(n_cam, 3, H, W)
            "pcds": pcds,  # tensor(n_cam, H, W)
            "proprioception": gripper_history,  # tensor(1, 8)
            "action": action,  # tensor(T, 8)
            "action_mask": torch.zeros(action.shape[:-1]).bool()  # tensor (T,)
        }

        return ret_dict

    def __len__(self):
        return len(self.annos['variation'])


class PeractDataset(RLBenchDataset):
    """RLBench dataset under Peract setup."""
    tasks = [
        "place_cups", "close_jar", "insert_onto_square_peg",
        "light_bulb_in", "meat_off_grill", "open_drawer",
        "place_shape_in_shape_sorter", "place_wine_at_rack_location",
        "push_buttons", "put_groceries_in_cupboard",
        "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
        "slide_block_to_color_target", "stack_blocks", "stack_cups",
        "sweep_to_dustpan_of_size", "turn_tap"
    ]
    variations = range(0, 199)
    cameras = ("left_shoulder", "right_shoulder", "wrist", "front")

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # the path to the instruction file
        precompute_instruction_encodings,  # whether instruction is latent encoded
    ):
        taskvar = [(task, var) for task in self.tasks for var in self.variations]
        cache_size = 0
        max_episode_length = 100
        max_episodes_per_task = -1
        color_aug = False
        bimanual = False
        relative_action = False

        super().__init__(
            root=root,
            instructions=instructions,
            precompute_instruction_encodings=precompute_instruction_encodings
        )
