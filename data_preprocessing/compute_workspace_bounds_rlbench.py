"""
This script computes the minimum and maximum gripper locations for
each task in the training set.
"""
import os
import random
import pickle
import glob

import tap
from typing import List, Tuple, Optional
from pathlib import Path
import torch
import pprint
import json
import numpy as np
from tqdm import tqdm
import blosc

from datasets.dataset_engine import RLBenchDataset


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("wrist", "front")
    image_size: str = "256,256"
    dataset: List[Path] = "data/peract/Peract_packaged/train"
    max_episodes_per_task: int = -1
    cache_size: int = 0
    out_file: str = "location_bounds.json"
    return_trajectory: int = 1
    relative_action: int = 0

    tasks: Tuple[str, ...] = (
        #"bimanual_pick_laptop",
        #"bimanual_pick_plate",
        #"bimanual_straighten_rope",
        #"bimanual_sweep_to_dustpan",
        #"coordinated_lift_ball",
        #"coordinated_lift_tray",
        #"coordinated_push_box",
        #"coordinated_put_bottle_in_fridge",
        #"coordinated_put_item_in_drawer",
        #"coordinated_take_tray_out_of_oven",
        #"dual_push_buttons",
        #"handover_item_easy",
        #"handover_item"
        "close_jar",
        "open_drawer",
        "sweep_to_dustpan_of_size",
        "meat_off_grill",
        "turn_tap",
        "slide_block_to_color_target",
        "put_item_in_drawer",
        "reach_and_drag",
        "push_buttons",
        "stack_blocks"
    )
    variations: Tuple[int, ...] = range(0, 199)
    mode: str = "aggregate"  # channelwise, aggregate
    horizon: int = 26


if __name__ == "__main__":
    args = Arguments().parse_args()

    bounds = {task: [] for task in args.tasks}

    for task in args.tasks:

        taskvar = [
            (task, var)
            for var in args.variations
        ]
        max_episode_length = 200

        dataset = RLBenchDataset(
            root=args.dataset,
            instructions=None,
            taskvar=taskvar,
            max_episode_length=max_episode_length,
            cache_size=0,
            max_episodes_per_task=args.max_episodes_per_task,
            cameras=args.cameras,  # type: ignore
            return_low_lvl_trajectory=True,
            dense_interpolation=True,
            relative_action=bool(args.relative_action),
            interpolation_length=args.horizon,
            training=False
        )

        print(
            f"Computing gripper location bounds for task {task} "
            f"from dataset of length {len(dataset)}"
        )

        stats = []
        for i in tqdm(range(len(dataset))):
            ep = dataset[i]
            # from utils.visualize_keypose_frames import visualize_actions_and_point_clouds_video 
            # visualize_actions_and_point_clouds_video(
            #     ep['pcds'], ep['rgbs'].add(1).mul(0.5),
            #     ep['curr_gripper_history'][:, -1, :8],
            #     ep['curr_gripper_history'][:, -1, 8:16]
            # )
            # import ipdb; ipdb.set_trace()
            bounds[task].append(ep["trajectory"][..., :3].reshape([-1, 3]))

    bounds = {
        task: [
            torch.cat(gripper_locs, dim=0).min(dim=0).values.tolist(),
            torch.cat(gripper_locs, dim=0).max(dim=0).values.tolist()
        ]
        for task, gripper_locs in bounds.items()
        if len(gripper_locs) > 0
    }

    pprint.pprint(bounds)
    json.dump(bounds, open(args.out_file, "w"), indent=4)
