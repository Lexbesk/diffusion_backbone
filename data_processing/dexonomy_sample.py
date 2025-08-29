import argparse
import json
import os
import pickle

import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from data_processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    store_instructions
)

from datasets import create_train_dataloader
from datasets import fetch_dataset_class
from datasets.base_dex import DexDataset
from datasets.sample_dex import DexSampleDataset

from types import SimpleNamespace
from torch.utils.data import DataLoader
from numcodecs import Blosc, VLenUTF8


NCAM = 4
NHAND = 1
IM_SIZE = 128
DEPTH_SCALE = 2**24 - 1
chunk_rows = 1024
grow_factor = 1.3


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/VLA/pe/'),
        # ('tgt', str, '/data/group_data/katefgroup/datasets/austinz/zarr_datasets/Dexonomy_zarr_all')
        ('tgt', str, '/data/user_data/austinz/Robots/manipulation/zarr_datasets/Dexonomy_zarr_prime')
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()

def get_datasets(dataset_cls):
    """Initialize datasets."""

    object_paths = ['assets/object/DGN_5k']
    dataset_config = {
        'num_workers': 0,
        'num_points': 1024,
        'joint_num': 22,
        # 'grasp_type_lst': ["1_Large_Diameter"],
        'grasp_type_lst': [
                            "10_Power_Disk", "11_Power_Sphere", "12_Precision_Disk", "13_Precision_Sphere",
                            "14_Tripod", "15_Fixed_Hook", "16_Lateral", "17_Index_Finger_Extension",
                            "18_Extensior_Type", "1_Large_Diameter", "20_Writing_Tripod", "22_Parallel_Extension",
                            "23_Adduction_Grip", "24_Tip_Pinch", "25_Lateral_Tripod", "26_Sphere_4_Finger",
                            "27_Quadpod", "28_Sphere_3_Finger", "29_Stick", "2_Small_Diameter",
                            "30_Palmar", "31_Ring", "32_Ventral", "33_Inferior_Pincer",
                            "3_Medium_Wrap", "4_Adducted_Thumb", "5_Light_Tool", "6_Prismatic_4_Finger",
                            "7_Prismatic_3_Finger", "8_Prismatic_2_Finger", "9_Palmar_Pinch"
                            ],
        # 'grasp_type_lst': ["1_Large_Diameter", "6_Prismatic_4_Finger", "9_Palmar_Pinch", "18_Extensior_Type", "22_Parallel_Extension",
        #                    "26_Sphere_4_Finger", "31_Ring", "33_Inferior_Pincer"
        #                     ],
        'grasp_path': 'assets/grasp/Dexonomy_GRASP_shadow/succ_collect',
        'object_path': None,
        'split_path': 'valid_split',
        'pc_path': 'vision_data/azure_kinect_dk',  # relative to object_path
        'batch_size': 1
    }
    dataset_config = SimpleNamespace(**dataset_config)
    train_dataset_lst = []
    val_dataset_lst = []
    for p in object_paths:
        object_path = p
        dataset_config.object_path = object_path
        dataset_config.batch_size = 1
        train_dataset_lst.append(dataset_cls(dataset_config, "train"))
        dataset_config.batch_size = 1
        val_dataset_lst.append(dataset_cls(dataset_config, "eval"))
    train_dataset = torch.utils.data.ConcatDataset(train_dataset_lst)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset_lst)
    
    return train_dataset, val_dataset

def create_train_dataloader(train_dataset, val_dataset):

    train_loader = DataLoader(
                    train_dataset,
                    batch_size=128,
                    drop_last=True,
                    num_workers=8,
                    shuffle=False)
    val_loader = DataLoader(
                    val_dataset,
                    batch_size=128,
                    drop_last=True,
                    num_workers=8,
                    shuffle=False)
    return train_loader, val_loader

def all_tasks_main(split):

        train_dataset, val_dataset = get_datasets(DexSampleDataset)
        
        mode = split
        if mode == 'train':
            ds = train_dataset
        else:
            ds = val_dataset
        print(len(ds), 'dataset length')
        dataloader = DataLoader(
            ds,
            batch_size=10,
            drop_last=False,
            num_workers=1,
            shuffle=False
        )
        batch_count = -1
        for batch in tqdm(dataloader, desc=f"Writing {mode}"):
            batch_count += 1
            b = batch["grasp_qpos"].shape[0]
            print(f"Batch {batch_count} with {b} samples")
            

            print(batch["partial_points"].numpy().astype(np.float16).shape, 'partial_points shape')
            print(batch["grasp_qpos"].numpy().astype(np.float32).shape, 'grasp_qpos shape')
            print(batch["pregrasp_qpos"].numpy().astype(np.float32).shape, 'pregrasp_qpos shape')
            print(batch["squeeze_qpos"].numpy().astype(np.float32).shape, 'squeeze_qpos shape')
            print(batch["grasp_type_id"].numpy().astype(np.uint8).shape, 'grasp_type_id shape')
            print(batch["anchor_visible"].numpy().astype(np.uint8).shape, 'anchor_visible shape')
            print(batch["obj_pose"].numpy().astype(np.float32).shape, 'obj_pose shape')
            print(batch["obj_scale"].numpy().astype(np.float32).shape, 'obj_scale shape')
            print(batch["obj_path"], 'obj_path shape')
            
            
            batch_to_save = {}
            batch_to_save['partial_points'] = batch["partial_points"].numpy().astype(np.float16)
            batch_to_save['grasp_qpos'] = batch["grasp_qpos"].numpy().astype(np.float32)
            batch_to_save['pregrasp_qpos'] = batch["pregrasp_qpos"].numpy().astype(np.float32)
            batch_to_save['squeeze_qpos'] = batch["squeeze_qpos"].numpy().astype(np.float32)
            batch_to_save['grasp_type_id'] = batch["grasp_type_id"].numpy().astype(np.uint8)
            batch_to_save['anchor_visible'] = batch["anchor_visible"].numpy().astype(np.uint8)
            batch_to_save['obj_pose'] = batch["obj_pose"].numpy().astype(np.float32)
            batch_to_save['obj_scale'] = batch["obj_scale"].numpy().astype(np.float32)
            batch_to_save['obj_path'] = batch["obj_path"]
            
            save_path_dir = os.path.join('sampled_data', 'sample_object_0')
            os.makedirs(save_path_dir, exist_ok=True)
            save_path = os.path.join(save_path_dir, f"grasp_{batch_count}.npz")
            print(f"Saving sample to {save_path}")
            np.savez(save_path, **batch_to_save)
    

if __name__ == "__main__":
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt
    # Create zarr data
    for split in ['val']:
        all_tasks_main(split)
