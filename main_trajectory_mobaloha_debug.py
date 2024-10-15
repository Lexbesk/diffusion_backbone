"""Main script for trajectory optimization."""

import os
from pathlib import Path
import random
from typing import Optional

import numpy as np
import torch

from datasets.dataset_mobaloha import MobileAlohaDataset
from diffuser_actor.trajectory_optimization.bimanual_diffuser_actor_mobaloha import BiManualDiffuserActor

from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)
from main_trajectory import Arguments as BaseArguments
from main_trajectory import TrainTester as BaseTrainTester
from main_trajectory import traj_collate_fn


class Arguments(BaseArguments):
    instructions: Optional[Path] = None


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        instruction = load_instructions(
            self.args.instructions,
            tasks=self.args.tasks,
            variations=self.args.variations
        )
        if instruction is None:
            taskvar = [
                (task, var)
                for task in self.args.tasks
                for var in self.args.variations
            ]
        else:
            taskvar = [
                (task, var)
                for task, var_instr in instruction.items()
                for var in var_instr.keys()
            ]

        # Initialize datasets with arguments
        train_dataset = MobileAlohaDataset(
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
            bimanual=bool(self.args.bimanual)
        )
        test_dataset = MobileAlohaDataset(
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
            bimanual=bool(self.args.bimanual)
        )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = BiManualDiffuserActor(
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
    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            if self.args.keypose_only:
                sample["trajectory"] = sample["trajectory"][:, [-1]]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
            else:
                sample["trajectory"] = sample["trajectory"][:, 1:]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

            curr_gripper = (
                sample["curr_gripper"] if self.args.num_history < 1
                else sample["curr_gripper_history"][:, -self.args.num_history:]
            )

            action = model(
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                sample["instr"].to(device),
                curr_gripper.to(device),
                run_inference=True
            )

            debug_data = []
            debug_data.append( sample["trajectory"].cpu() )
            debug_data.append( sample["rgbs"].cpu() )
            debug_data.append( sample["pcds"].cpu() )
            debug_data.append( action.cpu() )
            np.save("debug_{}".format(i), debug_data)

            losses, losses_B = criterion.compute_metrics(
                action,
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device)
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])





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

