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

from types import SimpleNamespace
from torch.utils.data import DataLoader
from numcodecs import Blosc, VLenUTF8


NCAM = 4
NHAND = 1
IM_SIZE = 128
DEPTH_SCALE = 2**24 - 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/VLA/pe/'),
        ('tgt', str, '/data/group_data/katefgroup/datasets/austinz/zarr_datasets/Dexonomy_zarr_1')
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()

def get_datasets(dataset_cls):
    """Initialize datasets."""

    object_paths = ['assets/object/DGN_5k', 'assets/object/objaverse_5k']
    dataset_config = {
        'num_workers': 0,
        'num_points': 1024,
        'joint_num': 22,
        'grasp_type_lst': ["1_Large_Diameter"],
        # 'grasp_type_lst': [
        #                     "10_Power_Disk", "11_Power_Sphere", "12_Precision_Disk", "13_Precision_Sphere",
        #                     "14_Tripod", "15_Fixed_Hook", "16_Lateral", "17_Index_Finger_Extension",
        #                     "18_Extensior_Type", "1_Large_Diameter", "20_Writing_Tripod", "22_Parallel_Extension",
        #                     "23_Adduction_Grip", "24_Tip_Pinch", "25_Lateral_Tripod", "26_Sphere_4_Finger",
        #                     "27_Quadpod", "28_Sphere_3_Finger", "29_Stick", "2_Small_Diameter",
        #                     "30_Palmar", "31_Ring", "32_Ventral", "33_Inferior_Pincer",
        #                     "3_Medium_Wrap", "4_Adducted_Thumb", "5_Light_Tool", "6_Prismatic_4_Finger",
        #                     "7_Prismatic_3_Finger", "8_Prismatic_2_Finger", "9_Palmar_Pinch"
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
    # Check if the zarr already exists
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists. Skipping...")
        return None

    # Initialize zarr
    # compressor = Blosc(cname='lz4', clevel=3, shuffle=Blosc.SHUFFLE)
    compressor = Blosc(
        cname="zstd",      # Zstandard back‑end
        clevel=5,          # 5–7 gives ~2× the ratio of clevel=3 LZ4
        shuffle=Blosc.BITSHUFFLE   # bit‑shuffle filter helps floats a lot
    )
    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(1,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        _create("partial_points", (4096, 3), "float16")   # (N,4096,3)
        _create("pregrasp_qpos",  (29,),     "float32")   # (N,29)
        _create("grasp_qpos",     (29,),     "float32")
        _create("squeeze_qpos",   (29,),     "float32")
        _create("grasp_type_id",  (),        "uint8")     # (N,)
        _create("anchor_visible", (),        "uint8")     # (N,)
        _create("obj_pose",       (7,),      "float32")      # <-- NEW
        _create("obj_scale",      (),        "float32") 
        
        zarr_file.create_dataset(
            "obj_path",
            shape=(0,),
            chunks=(256,),              # tune to taste
            dtype=object,
            object_codec=VLenUTF8()
        )
        
        # dataset_class = fetch_dataset_class("Dexonomy")
        train_dataset, val_dataset = get_datasets(DexDataset)
        
        mode = split
        if mode == 'train':
            ds = train_dataset
        else:
            ds = val_dataset
            
        
        dataloader = DataLoader(
            ds,
            batch_size=256,
            drop_last=False,
            num_workers=32,
            shuffle=False
        )
        batch_count = 0
        for batch in tqdm(dataloader, desc=f"Writing {mode}"):
            # if batch_count >= 70000 // 128 + 1:
            #     break
            batch_count += 1
            # # make sure everything is numpy on CPU
            # pts   = np.asarray(sample["partial_points"],  dtype=np.float32)
            # pre   = np.asarray(sample["pregrasp_qpos"],   dtype=np.float32)
            # grasp = np.asarray(sample["grasp_qpos"],      dtype=np.float32)
            # sqz   = np.asarray(sample["squeeze_qpos"],    dtype=np.float32)
            # gtid  = np.asarray(sample["grasp_type_id"],   dtype=np.uint8)
            
            pts   = batch["partial_points"].numpy().astype(np.float16)
            pre   = batch["pregrasp_qpos"].numpy().astype(np.float32)
            grasp = batch["grasp_qpos"].numpy().astype(np.float32)
            sqz   = batch["squeeze_qpos"].numpy().astype(np.float32)
            gtid  = batch["grasp_type_id"].numpy().astype(np.uint8)
            anchor_visible = batch["anchor_visible"].numpy().astype(np.uint8)
            pose        = batch["obj_pose"].numpy().astype(np.float32)
            scale       = batch["obj_scale"].numpy().astype(np.float32)
            
            paths = np.asarray(batch["obj_path"], dtype=object)

            # append – zarr takes (…, shape_of_field) for each call
            zarr_file["partial_points"].append(pts)   # add leading batch dim
            zarr_file["pregrasp_qpos"].append(pre)
            zarr_file["grasp_qpos"].append(grasp)
            zarr_file["squeeze_qpos"].append(sqz)
            zarr_file["grasp_type_id"].append(gtid)
            zarr_file["anchor_visible"].append(anchor_visible)
            zarr_file["obj_pose"].append(pose)
            zarr_file["obj_scale"].append(scale)
            zarr_file["obj_path"].append(paths)
        


if __name__ == "__main__":
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt
    # Create zarr data
    for split in ['train', 'val']:
        all_tasks_main(split)
    # # Store instructions as json (can be run independently)
    # os.makedirs('instructions/peract', exist_ok=True)
    # instr_dict = store_instructions(ROOT, tasks)
    # with open('instructions/peract/instructions.json', 'w') as fid:
    #     json.dump(instr_dict, fid)
