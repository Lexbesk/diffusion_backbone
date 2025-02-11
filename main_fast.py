"""Main script for training and testing."""

import io
import os
from pathlib import Path
import random
import argparse

import cv2
from kornia import augmentation as K
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine import BaseTrainTester
from datasets.dataset_rlbench import (
    PeractDataset,
    GNFactorDataset,
    PeractSingleCamDataset
)
from datasets.dataset_comp import RLBenchCompDataset
from datasets.dataset_calvin_zarr import ABC_DDataset, ABC_DSingleCamDataset
from diffuser_actor.encoder.text.clip import ClipTextEncoder
from diffuser_actor.policy import BimanualDenoiseActor, DenoiseActor
from diffuser_actor.policy.denoise_sa_actor import DenoiseActor as DenoiseActorSA
from diffuser_actor.depth2cloud.rlbench import (
    PeractDepth2Cloud,
    GNFactorDepth2Cloud
)
from utils.common_utils import count_parameters, str2bool, str_none


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Trainign and testing
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=str_none, default=None)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=500)
    parser.add_argument('--eval_only', type=str2bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default="constant")
    parser.add_argument('--wd', type=float, default=5e-3)
    parser.add_argument('--train_iters', type=int, default=200000)
    parser.add_argument('--val_iters', type=int, default=-1)
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default="Peract")
    parser.add_argument('--train_data_dir', type=Path, required=True)
    parser.add_argument('--eval_data_dir', type=Path, required=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=4)
    # Logging
    parser.add_argument('--base_log_dir', type=Path,
                        default=Path(__file__).parent / "train_logs")
    parser.add_argument('--exp_log_dir', type=Path, default="exp")
    parser.add_argument('--run_log_dir', type=Path, default="run")
    # Text encoder
    parser.add_argument('--text_max_length', type=int, default=53)
    # Model
    parser.add_argument('--bimanual', type=str2bool, default=False)
    parser.add_argument('--workspace_normalizer_buffer', type=float, default=0.04)
    parser.add_argument('--use_flow_matching', type=str2bool, default=False)
    parser.add_argument('--backbone', type=str, default="clip")
    parser.add_argument('--embedding_dim', type=int, default=144)
    parser.add_argument('--num_attn_heads', type=int, default=9)
    parser.add_argument('--num_vis_ins_attn_layers', type=int, default=2)
    parser.add_argument('--train_instructions', type=Path, default='')
    parser.add_argument('--val_instructions', type=Path, default='')
    parser.add_argument('--precompute_instruction_encodings', type=str2bool, default=True)
    parser.add_argument('--use_instruction', type=str2bool, default=False)
    parser.add_argument('--rotation_parametrization', type=str, default='quat')
    parser.add_argument('--quaternion_format', type=str, default='wxyz')
    parser.add_argument('--denoise_timesteps', type=int, default=10)
    parser.add_argument('--denoise_model', type=str, default="rectified_flow",
                        choices=["ddpm", "rectified_flow"])
    parser.add_argument('--keypose_only', type=str2bool, default=True)
    parser.add_argument('--num_history', type=int, default=0)
    parser.add_argument('--relative_action', type=str2bool, default=False)
    parser.add_argument('--fps_subsampling_factor', type=int, default=5)
    parser.add_argument('--sa_var', type=str2bool, default=False)
    parser.add_argument('--ayush', type=str2bool, default=False)
    parser.add_argument('--not_seed', type=str2bool, default=False)
    parser.add_argument('--memory_limit', type=float, default=8)

    return parser.parse_args()


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        super().__init__(args)
        _cls = {
            "Peract": PeractDepth2Cloud,
            "PeractSingleCam": GNFactorDepth2Cloud,
            # "Peract2": Peract2Dataset,
            "GNFactor": GNFactorDepth2Cloud,
            "RLComp": GNFactorDepth2Cloud,
            "ABC_D": None,
            "ABC_DSingleCam": None
        }[args.dataset]
        im_size = 160 if args.dataset.startswith("ABC") else 256
        if _cls is not None:
            self.depth2cloud = _cls((256, 256))
        else:
            self.depth2cloud = None
        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(
                degrees=0,
                scale=(0.75, 1.25),
                padding_mode="reflection",
                p=1.0
            ),
            K.RandomResizedCrop(
                size=(im_size, im_size),
                scale=(0.7, 1.0)
            )
        ).cuda()

    def get_datasets(self):
        """Initialize datasets."""
        dataset_cls = {
            "Peract": PeractDataset,
            "PeractSingleCam": PeractSingleCamDataset,
            # "Peract2": Peract2Dataset,
            "GNFactor": GNFactorDataset,
            "RLComp": RLBenchCompDataset,
            "ABC_D": ABC_DDataset,
            "ABC_DSingleCam": ABC_DSingleCamDataset
        }[args.dataset]

        # Initialize datasets with arguments
        train_dataset = dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            precompute_instruction_encodings=self.args.precompute_instruction_encodings,
            relative_action=self.args.relative_action,
            mem_limit=self.args.memory_limit
        )
        test_dataset = dataset_cls(
            root=self.args.eval_data_dir,
            instructions=self.args.val_instructions,
            precompute_instruction_encodings=self.args.precompute_instruction_encodings,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1
        )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        if self.args.sa_var:
            model_class = DenoiseActorSA
        elif self.args.bimanual:
            model_class = BimanualDenoiseActor
        else:
            model_class = DenoiseActor
        _model = model_class(
            backbone=self.args.backbone,
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=self.args.use_instruction,
            num_attn_heads=self.args.num_attn_heads,
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model,
            nhist=self.args.num_history,
            relative=self.args.relative_action,
            ayush=self.args.ayush
        )
        print("Model parameters:", count_parameters(_model))

        return _model

    def get_workspace_normalizer(self, data_loader=None):
        print("Computing workspace normalizer...")
        dataset_cls = {
            "Peract": PeractDataset,
            "PeractSingleCam": PeractSingleCamDataset,
            # "Peract2": Peract2Dataset,
            "GNFactor": GNFactorDataset,
            "RLComp": RLBenchCompDataset,
            "ABC_D": ABC_DDataset,
            "ABC_DSingleCam": ABC_DSingleCamDataset
        }[args.dataset]

        # Initialize datasets with arguments
        train_dataset = dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            precompute_instruction_encodings=self.args.precompute_instruction_encodings,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            collate_fn=traj_collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers
        )

        bounds = []
        for sample in tqdm(data_loader):
            bounds.append(sample["action"][..., :3].reshape([-1, 3]))

        bounds = torch.cat(bounds, dim=0)
        min_ = bounds.min(dim=0).values - self.args.workspace_normalizer_buffer
        max_ = bounds.max(dim=0).values + self.args.workspace_normalizer_buffer
        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    @staticmethod
    def get_criterion():
        return TrajectoryCriterion()

    def get_text_encoder(self):
        """Initialize the model."""
        return ClipTextEncoder(self.args.text_max_length)

    @torch.no_grad()
    def prepare_batch(self, sample, text_encoder, augment=False):
        # Actions
        if self.args.keypose_only:
            sample["action"] = sample["action"][:, [-1]]
            sample["action_mask"] = sample["action_mask"][:, [-1]]
        else:
            sample["action"] = sample["action"][:, 1:]
            sample["action_mask"] = sample["action_mask"][:, 1:]
        # Observations
        if self.depth2cloud is not None:
            pcds = self.depth2cloud(sample['pcds'].cuda(non_blocking=True))
        else:
            pcds = sample['pcds'].cuda(non_blocking=True)
        if augment:
            b, nc, _, h, w = sample['rgbs'].shape
            obs = torch.cat((
                sample['rgbs'].cuda(non_blocking=True).half() / 255,
                pcds
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)
            obs = self.aug(obs)
            rgbs = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcds = obs[:, 3:].reshape(b, nc, 3, h, w).float()
        else:
            rgbs = sample['rgbs'].cuda(non_blocking=True).float() / 255
            pcds = pcds.float()
        # Instructions
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if not self.args.precompute_instruction_encodings and self.args.backbone != 'florence2':
                instr = text_encoder(sample['instr'], "cuda")
            else:
                instr = sample["instr"]
        return (
            sample["action"].cuda(non_blocking=True),
            sample["action_mask"].cuda(non_blocking=True),
            rgbs,
            pcds,
            instr,
            sample["proprioception"].cuda(non_blocking=True)
        )

    def train_one_step(self, model, text_encoder, criterion,
                       optimizer, scaler, lr_scheduler, step_id, sample):
        """Run a single training step."""
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        # Forward pass
        action, action_mask, rgbs, pcds, instr, prop = self.prepare_batch(
            sample, text_encoder, augment=True
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(action, action_mask, rgbs, pcds, instr, prop)

            # Backward pass
            loss = criterion.compute_loss(out)
        scaler.scale(loss).backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            scaler.step(optimizer)
            scaler.update()

        # Step the lr scheduler
        lr_scheduler.step()

    @torch.inference_mode()
    def evaluate_nsteps(self, model, text_encoder, criterion, loader,
                        step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        if split == 'val':
            val_iters = -1
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            action, action_mask, rgbs, pcds, instr, prop = self.prepare_batch(
                sample, text_encoder, augment=True
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_action = model(
                    action, action_mask, rgbs, pcds, instr, prop,
                    run_inference=True
                )

                losses, losses_B = criterion.compute_metrics(
                    pred_action, action, action_mask
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

            # Generate visualizations
            if i == 0 and dist.get_rank() == 0 and step_id > -1:
                viz_key = f'{split}-viz/viz'
                viz = generate_visualizations(
                    pred_action, action, action_mask
                )
                self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)


def traj_collate_fn(batch):
    # Values for these come as tensors
    keys = [
        "action", "proprioception",
        "rgbs", "pcds"
    ]
    if isinstance(batch[0].get("instr", None), torch.Tensor):
        keys.append("instr")
    ret_dict = {k_: torch.stack([item[k_] for item in batch]) for k_ in keys}
    ret_dict["action_mask"] = torch.zeros(
        ret_dict["action"].shape[:-1], dtype=bool
    )

    # Values for these come as lists
    list_keys = ["task"]
    if isinstance(batch[0].get("instr", None), list):
        list_keys.append("instr")
    for key in list_keys:
        ret_dict[key] = [item[key][0] for item in batch]
    return ret_dict


class TrajectoryCriterion:

    def __init__(self):
        pass

    def compute_loss(self, pred, gt=None, mask=None, is_loss=True):
        if not is_loss:
            assert gt is not None and mask is not None
            return self.compute_metrics(pred, gt, mask)[0]['action_mse']
        return pred

    @staticmethod
    def compute_metrics(pred, gt, mask):
        # pred/gt are (B, L, 7), mask (B, L)
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        # symmetric quaternion eval
        quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
        quat_l1_ = (pred[..., 3:7] + gt[..., 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        # gripper openess
        openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).bool()
        tr = 'traj_'

        # Trajectory metrics
        ret_1, ret_2 = {
            tr + 'action_mse': F.mse_loss(pred, gt),
            tr + 'pos_l2': pos_l2.mean(),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
            tr + 'rot_l1': quat_l1.mean(),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
            tr + 'gripper': openess.flatten().float().mean()
        }, {
            tr + 'pos_l2': pos_l2.mean(-1),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
            tr + 'rot_l1': quat_l1.mean(-1),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1)
        }

        # Keypose metrics
        if pred.ndim == gt.ndim == 3:
            pos_l2 = ((pred[:, -1, :3] - gt[:, -1, :3]) ** 2).sum(-1).sqrt()
            quat_l1 = (pred[:, -1, 3:7] - gt[:, -1, 3:7]).abs().sum(-1)
            quat_l1_ = (pred[:, -1, 3:7] + gt[:, -1, 3:7]).abs().sum(-1)
        else:
            pos_l2 = ((pred[:, -1, :, :3] - gt[:, -1, :, :3]) ** 2).sum(-1).sqrt()
            quat_l1 = (pred[:, -1, :, 3:7] - gt[:, -1, :, 3:7]).abs().sum(-1)
            quat_l1_ = (pred[:, -1, :, 3:7] + gt[:, -1, :, 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        ret_1.update({
            'pos_l2_final': pos_l2.mean(),
            'pos_l2_final<0.01': (pos_l2 < 0.01).float().mean(),
            'rot_l1': quat_l1.mean(),
            'rot_l1<0025': (quat_l1 < 0.025).float().mean()
        })
        ret_2.update({
            'pos_l2_final': pos_l2,
            'pos_l2_final<0.01': (pos_l2 < 0.01).float(),
            'rot_l1': quat_l1,
            'rot_l1<0.025': (quat_l1 < 0.025).float(),
        })

        return ret_1, ret_2


def fig_to_numpy(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


def generate_visualizations(pred, gt, mask, box_size=0.3):
    batch_idx = 0
    pred = pred[batch_idx].detach().cpu().numpy()
    gt = gt[batch_idx].detach().cpu().numpy()
    mask = mask[batch_idx].detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    if pred.ndim == 2 and gt.ndim == 2:
        ax.scatter3D(
            pred[~mask][:, 0], pred[~mask][:, 1], pred[~mask][:, 2],
            color='red', label='pred'
        )
        ax.scatter3D(
            gt[~mask][:, 0], gt[~mask][:, 1], gt[~mask][:, 2],
            color='blue', label='gt'
        )
        center = gt[~mask].mean(0)
    elif pred.ndim == 3 and gt.ndim == 3:
        ax.scatter3D(
            pred[~mask][:, 0, 0], pred[~mask][:, 0, 1], pred[~mask][:, 0, 2],
            color='red', label='pred-left'
        )
        if(pred[~mask].shape[1]>1):
            ax.scatter3D(
                pred[~mask][:, 1, 0], pred[~mask][:, 1, 1], pred[~mask][:, 1, 2],
                color='magenta', label='pred-right'
            )
        ax.scatter3D(
            gt[~mask][:, 0, 0], gt[~mask][:, 0, 1], gt[~mask][:, 0, 2],
            color='blue', label='gt-left'
        )
        if(pred[~mask].shape[1]>1):
            ax.scatter3D(
                gt[~mask][:, 1, 0], gt[~mask][:, 1, 1], gt[~mask][:, 1, 2],
                color='cyan', label='gt-right'
            )
        center = np.reshape(gt[~mask], (-1, gt.shape[-1])).mean(0)
    else:
        raise ValueError("Invalid dimensions")

    ax.set_xlim(center[0] - box_size, center[0] + box_size)
    ax.set_ylim(center[1] - box_size, center[1] + box_size)
    ax.set_zlim(center[2] - box_size, center[2] + box_size)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.legend()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    img = fig_to_numpy(fig, dpi=120)
    plt.close()
    return img.transpose(2, 0, 1)


if __name__ == '__main__':
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
    if not args.not_seed:
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
