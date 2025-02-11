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
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange

from engine import BaseTrainTester
from datasets.dataset_isaac import IsaacDataset
from diffuser_actor.encoder.text.clip import ClipTextEncoder
from diffuser_actor.policy.denoise_actor_seg import DenoiseActor
from diffuser_actor.depth2cloud.isaac import IsaacDepth2Cloud
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
    parser.add_argument('--precompute_instruction_encodings', type=str2bool, required=True)
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

    return parser.parse_args()


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        super().__init__(args)
        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(
                degrees=0,
                scale=(0.75, 1.25),
                padding_mode="reflection",
                p=0.98
            ),
            # K.RandomPerspective(p=0.2)
        ).cuda()
        self._depth2cloud = IsaacDepth2Cloud()

    def get_datasets(self):
        """Initialize datasets."""
        dataset_cls = IsaacDataset

        # Initialize datasets with arguments
        train_dataset = dataset_cls(
            root=self.args.train_data_dir,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=8
        )
        test_dataset = dataset_cls(
            root=self.args.eval_data_dir,
            relative_action=self.args.relative_action,
            mem_limit=0.1
        )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = DenoiseActor(
            backbone=self.args.backbone,
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            num_attn_heads=self.args.num_attn_heads,
            use_instruction=self.args.use_instruction,
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model,
            nhist=self.args.num_history,
            relative=self.args.relative_action,
        )
        print("Model parameters:", count_parameters(_model))

        return _model

    def get_workspace_normalizer(self, data_loader=None):
        print("Computing workspace normalizer...")

        # Initialize datasets with arguments
        train_dataset = IsaacDataset(
            root=self.args.train_data_dir,
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
        min_bound = bounds.min(dim=0).values - self.args.workspace_normalizer_buffer
        max_bound = bounds.max(dim=0).values + self.args.workspace_normalizer_buffer
        normalizer = nn.Parameter(torch.stack([min_bound, max_bound]),
                                  requires_grad=False)

        return normalizer

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
        pcds = self._depth2cloud(
            sample['pcds'].cuda(non_blocking=True),
            sample['proj_matrix'].cuda(non_blocking=True),
            sample['extrinsics'].cuda(non_blocking=True)
        )
        segs = sample['segs'].cuda(non_blocking=True)
        if augment:
            b, nc, _, h, w = sample['rgbs'].shape
            obs = torch.cat((
                sample['rgbs'].cuda(non_blocking=True).half() / 255,
                pcds,
                segs[:, :, None].half()
            ), 2)  # (B, ncam, 7, H, W)
            obs = obs.reshape(-1, 7, h, w)
            obs = self.aug(obs)
            rgbs = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcds = obs[:, 3:6].reshape(b, nc, 3, h, w).float()
            segs = obs[:, 6].reshape(b, nc, h, w).bool()
            ##########
            _pcds = self._depth2cloud(
                sample['pcds'].cuda(non_blocking=True),
                sample['proj_matrix'].cuda(non_blocking=True),
                sample['extrinsics'].cuda(non_blocking=True)
            )[:, :, :1].repeat(1, 1, 3, 1, 1)
            min_ = _pcds.view(b, nc, -1).min(-1).values
            max_ = _pcds.view(b, nc, -1).max(-1).values
            orig_obs = torch.cat((
                sample['rgbs'].cuda(non_blocking=True).half() / 255,
                (_pcds - min_[:, :, None, None, None]) / (max_ - min_)[:, :, None, None, None],
                sample['segs'].cuda(non_blocking=True)[:, :, None].half().repeat(1, 1, 3, 1, 1)
            ), 4)
            _pcds = pcds.half()[:, :, :1].repeat(1, 1, 3, 1, 1)
            min_ = _pcds.view(b, nc, -1).min(-1).values
            max_ = _pcds.view(b, nc, -1).max(-1).values
            aug_obs = torch.cat((
                rgbs.half(),
                (_pcds - min_[:, :, None, None, None]) / (max_ - min_)[:, :, None, None, None],
                segs[:, :, None].half().repeat(1, 1, 3, 1, 1)
            ), 4)
            cat_obs = torch.cat((orig_obs, aug_obs), 3)
            for b in range(len(cat_obs)):
                for c in range(len(cat_obs[b])):
                    plt.imshow(cat_obs[b, c].permute(1, 2, 0).float().cpu().numpy())
                    plt.savefig(f'ex_{b}_{c}.jpg')
                    plt.close()
            jnkj
        else:
            rgbs = sample['rgbs'].cuda(non_blocking=True).float() / 255
            pcds = pcds.float()
        if rgbs.shape[-1] != 256:
            b, nc, c, h, w = sample['rgbs'].shape
            rgbs = F.interpolate(
                rgbs.reshape(b * nc, c, h, w),
                (256, 256),
                mode='nearest'
            ).reshape(b, nc, c, 256, 256)
        return (
            sample["action"].cuda(non_blocking=True),
            sample["action_mask"].cuda(non_blocking=True),
            rgbs,
            pcds,
            segs,
            sample["proprioception"].cuda(non_blocking=True)
        )

    def main(self, collate_fn=None):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(collate_fn)

        # Get model
        model = self.get_model()
        if not self.args.checkpoint:
            normalizer = self.get_workspace_normalizer(train_loader)
            model.workspace_normalizer.copy_(normalizer)
            dist.barrier()

        # Get criterion
        criterion = self.get_criterion()

        # Get optimizer
        optimizer = self.get_optimizer(model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        scaler = torch.GradScaler()

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        model = DistributedDataParallel(
            model, device_ids=[self.args.local_rank],
            broadcast_buffers=False, find_unused_parameters=True
        )

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            assert os.path.isfile(self.args.checkpoint)
            start_iter, best_loss = self.load_checkpoint(model, optimizer)
        print(model.module.workspace_normalizer)

        # Get text encoder
        text_encoder = self.get_text_encoder().cuda()

        # Eval only
        if bool(self.args.eval_only):
            if dist.get_rank() == 0:
                print("Test evaluation.......")
                model.eval()
                new_loss = self.evaluate_nsteps(
                    model, text_encoder, criterion, test_loader, step_id=-1,
                    val_iters=max(25, self.args.val_iters)
                )
            dist.barrier()
            return model

        # Step the lr scheduler to the current step
        for _ in range(start_iter):
            lr_scheduler.step()

        # Training loop
        iter_loader = iter(train_loader)
        model.train()
        for step_id in trange(start_iter, self.args.train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            self.train_one_step(
                model, text_encoder, criterion, optimizer, scaler, lr_scheduler,
                step_id, sample
            )

            if (step_id + 1) % self.args.val_freq == 0 and dist.get_rank() == 0:
                print("Train evaluation.......")
                model.eval()
                new_loss = self.evaluate_nsteps(
                    model, text_encoder, criterion, train_loader, step_id,
                    val_iters=max(5, self.args.val_iters),
                    split='train'
                )
                # save model
                best_loss = self.save_checkpoint(
                    model, optimizer, step_id,
                    new_loss, best_loss
                )
                model.train()
            dist.barrier()

        return model

    def train_one_step(self, model, text_encoder, criterion,
                       optimizer, scaler, lr_scheduler, step_id, sample):
        """Run a single training step."""
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        # Forward pass
        action, action_mask, rgbs, pcds, segs, prop = self.prepare_batch(
            sample, text_encoder, augment=True
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(action, action_mask, rgbs, pcds, segs, prop)

            # Backward pass
            loss = criterion.compute_loss(out)
        scaler.scale(loss).backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            scaler.step(optimizer)
            scaler.update()

        # Step the lr scheduler
        lr_scheduler.step()

    def store(self, rgbs):
        # b, nc, h, w
        from matplotlib import pyplot as plt
        for r, rgb in enumerate(rgbs):
            rgb = rgb.reshape(-1, rgb.shape[-1])
            plt.imshow(rgb.detach().cpu())
            plt.savefig(f'{r}.jpg')

    @torch.inference_mode()
    def evaluate_nsteps(self, model, text_encoder, criterion, loader,
                        step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if i == val_iters:
                break

            action, action_mask, rgbs, pcds, segs, prop = self.prepare_batch(
                sample, text_encoder, augment=True
            )
            # self.store(segs)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_action = model(
                    action, action_mask, rgbs, pcds, segs, prop,
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
        "action", "action_mask", "proprioception",
        "rgbs", "pcds", "segs", "extrinsics", "proj_matrix"
    ]
    ret_dict = {k_: torch.stack([item[k_] for item in batch]) for k_ in keys}

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
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    train_tester = TrainTester(args)
    train_tester.main(collate_fn=traj_collate_fn)
