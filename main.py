"""Main script for training and testing."""

import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from training.depth2cloud import fetch_depth2cloud
from training.trainers import fetch_train_tester
from utils.common_utils import str2bool, str_none


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Dataset/loader arguments
        ('train_data_dir', Path, ''),
        ('eval_data_dir', Path, ''),
        ('train_instructions', Path, ''),
        ('val_instructions', Path, ''),
        ('dataset', str, "Peract"),
        ('num_workers', int, 1),
        ('batch_size', int, 16),
        ('batch_size_val', int, 4),
        ('memory_limit', float, 8),  # cache limit in GB
        # Logging arguments
        ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('exp_log_dir', Path, "exp"),
        ('run_log_dir', Path, "run"),
        # Training and testing arguments
        ('checkpoint', str_none, None),
        ('val_freq', int, 500),
        ('eval_only', str2bool, False),
        ('lr', float, 1e-4),
        ('lr_scheduler', str, "constant"),
        ('wd', float, 5e-3),
        ('train_iters', int, 200000),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, False),
        ('keypose_only', str2bool, True),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('finetune_backbone', str2bool, False),
        ('finetune_text_encoder', str2bool, False),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 144),
        ('num_attn_heads', int, 9),
        ('num_vis_instr_attn_layers', int, 2),
        ('num_history', int, 1),
        # Model arguments: head
        ('workspace_normalizer_buffer', float, 0.04),
        ('relative_action', str2bool, False),
        ('quaternion_format', str, 'wxyz'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow")
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # Seeds
    torch.manual_seed(args.local_rank)
    np.random.seed(args.local_rank)
    random.seed(args.local_rank)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Select dataset and model classes
    dataset_class = fetch_dataset_class(args.dataset)
    model_class = fetch_model_class(args.model_type)
    depth2cloud = fetch_depth2cloud(args.dataset)

    # Run
    TrainTester = fetch_train_tester(args.dataset)
    train_tester = TrainTester(args, dataset_class, model_class, depth2cloud)
    train_tester.main()
