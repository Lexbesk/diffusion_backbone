"""Main script for trajectory optimization."""

import os
from pathlib import Path
import random
from typing import Optional
import pickle

import numpy as np
import torch

from datasets.dataset_mobaloha import MobileAlohaDataset
from diffuser_actor.trajectory_optimization.bimanual_diffuser_actor_mobaloha import BiManualDiffuserActor
from datasets.dataset_mobaloha import SingleArmMobileAlohaDataset
from diffuser_actor.trajectory_optimization.singlearm_diffuser_actor_mobaloha import SingleArmDiffuserActor

from utils.common_utils import (
    count_parameters, get_gripper_loc_bounds
)
from main_trajectory import Arguments as BaseArguments
from main_trajectory import TrainTester as BaseTrainTester
from main_trajectory import traj_collate_fn


class Arguments(BaseArguments):
    instructions: Optional[Path] = None


def load_instructions(instructions):
    instructions = pickle.load(
        open(instructions, "rb")
    )['embeddings']
    return instructions


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        instruction = None
        if(self.args.use_instruction):
            instruction = load_instructions(
                self.args.instructions,
            )
        taskvar = [
            (task, var)
            for task in self.args.tasks
            for var in self.args.variations
        ]

        # Initialize datasets with arguments
        dataset_class = MobileAlohaDataset if self.args.bimanual else SingleArmMobileAlohaDataset
        train_dataset = dataset_class(
            root=self.args.dataset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size,
            max_episodes_per_task=self.args.max_episodes_per_task,
            num_iters=self.args.train_iters,
            cameras=self.args.cameras,
            training=True,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length,
            relative_action=bool(self.args.relative_action),
            # bimanual=bool(self.args.bimanual)
            bimanual=True
        )
        test_dataset = dataset_class(
            root=self.args.valset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size_val,
            max_episodes_per_task=self.args.max_episodes_per_task,
            cameras=self.args.cameras,
            training=False,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length,
            relative_action=bool(self.args.relative_action),
            # bimanual=bool(self.args.bimanual)
            bimanual=True
        )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        model_class = BiManualDiffuserActor if self.args.bimanual else SingleArmDiffuserActor
        _model = model_class(
            backbone=self.args.backbone,
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=bool(self.args.use_instruction),
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            diffusion_timesteps=self.args.diffusion_timesteps,
            nhist=self.args.num_history,
            relative=bool(self.args.relative_action),
            lang_enhanced=bool(self.args.lang_enhanced)
        )
        print("Model parameters:", count_parameters(_model))

        return _model


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    train_tester = TrainTester(args)
    train_tester.main(collate_fn=traj_collate_fn)

