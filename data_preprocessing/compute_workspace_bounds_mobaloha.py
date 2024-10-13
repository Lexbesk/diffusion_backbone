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

from datasets.dataset_mobaloha import MobileAlohaDataset


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("front", )
    image_size: str = "256,256"
    dataset: List[Path] = "/home/tsungwek/data/mobile_aloha/train"
    max_episodes_per_task: int = -1
    cache_size: int = 0
    out_file: str = "location_bounds.json"
    return_trajectory: int = 1
    relative_action: int = 1

    tasks: Tuple[str, ...] = (
        "20241006_plate_keypose",
    )
    variations: Tuple[int, ...] = range(0, 1)
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

        dataset = MobileAlohaDataset(
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
            bimanual=True,
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
            # k = 10
            # visualize_actions_and_point_clouds_video(
            #     ep['pcds'][[k]].expand(26, -1, -1, -1,  -1),
            #     ep['rgbs'][[k]].expand(26, -1, -1, -1, -1),
            #     ep['trajectory'][k, :, 0],
            #     ep['trajectory'][k, :, 1]
            # )
            # visualize_actions_and_point_clouds_video(
            #    ep['pcds'].float(),
            #    ep['rgbs'].float(),
            #    ep['curr_gripper'][:, 0].float(),
            #    ep['curr_gripper'][:, 1].float(),
            #    ep['action'][:, 0].float(),
            #    ep['action'][:, 1].float(),
            # )
            # import ipdb; ipddb.set_trace()
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
