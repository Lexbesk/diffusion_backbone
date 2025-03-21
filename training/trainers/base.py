import os
import random

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from utils.common_utils import count_parameters
from ..schedulers import TriStageLRScheduler
from ..utils import compute_metrics, generate_visualizations


class BaseTrainTester:
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args, dataset_cls, model_cls, depth2cloud):
        """Initialize."""
        if dist.get_rank() == 0:
            args_dict = vars(args)
            conf = OmegaConf.create(args_dict)
            output_file = str(args.log_dir / "config.yaml")
            OmegaConf.save(conf, output_file)

        self.args = args
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls
        self.depth2cloud = depth2cloud
        self.aug = None

        if dist.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=args.log_dir)

    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            relative_action=self.args.relative_action,
            mem_limit=self.args.memory_limit
        )
        val_dataset = self.dataset_cls(
            root=self.args.eval_data_dir,
            instructions=self.args.val_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1
        )
        return train_dataset, val_dataset

    def get_loaders(self):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        # Datasets
        train_dataset, val_dataset = self.get_datasets()
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=traj_collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g
        )
        # No sampler for val!
        if dist.get_rank() == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size_val,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=traj_collate_fn,
                pin_memory=True,
                sampler=None,
                drop_last=False
            )
        else:
            val_loader = None
        return train_loader, val_loader

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = self.model_class(
            backbone=self.args.backbone,
            finetune_backbone=self.args.finetune_backbone,
            finetune_text_encoder=self.args.finetune_text_encoder,
            num_vis_instr_attn_layers=self.args.num_vis_instr_attn_layers,
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            nhist=self.args.num_history,
            nhand=2 if self.args.bimanual else 1,
            relative=self.args.relative_action,
            quaternion_format=self.args.quaternion_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model
        )
        print("Model parameters:", count_parameters(_model))

        return _model

    def get_workspace_normalizer(self):
        print("Computing workspace normalizer...")

        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            actions_only=True
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            collate_fn=actions_collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers
        )

        # Loop and compute action min-max
        bounds = []
        for sample in tqdm(data_loader):
            bounds.append(sample["action"][..., :3].reshape([-1, 3]))

        bounds = torch.cat(bounds, dim=0)
        min_ = bounds.min(dim=0).values - self.args.workspace_normalizer_buffer
        max_ = bounds.max(dim=0).values + self.args.workspace_normalizer_buffer
        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        if self.args.finetune_backbone:
            optimizer_grouped_parameters.append(
                {"params": [], "weight_decay": self.args.wd, "lr": 0.1 * self.args.lr}
            )
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if self.args.finetune_backbone and 'backbone' in name:
                optimizer_grouped_parameters[2]["params"].append(param)
            elif any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer

    def get_lr_scheduler(self, optimizer):
        """Initialize learning rate scheduler."""
        if self.args.lr_scheduler == "constant":
            scheduler = ConstantLR(
                optimizer, factor=1.0, total_iters=self.args.train_iters
            )
        elif self.args.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.train_iters)
        elif self.args.lr_scheduler == "tristage":
            scheduler = TriStageLRScheduler(optimizer)
        else:
            raise NotImplementedError

        return scheduler

    def main(self):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, val_loader = self.get_loaders()

        # Get model
        model = self.get_model()
        if not self.args.checkpoint or not os.path.exists(self.args.checkpoint):
            normalizer = self.get_workspace_normalizer()
            model.workspace_normalizer.copy_(normalizer)
            dist.barrier()

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
            start_iter, best_loss = self.load_checkpoint(model, optimizer)
        print(model.module.workspace_normalizer)

        # Eval only
        if bool(self.args.eval_only):
            if dist.get_rank() == 0:
                print("Test evaluation.......")
                model.eval()
                new_loss = self.evaluate_nsteps(
                    model, val_loader, step_id=-1,
                    val_iters=-1
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

            self.train_one_step(model, optimizer, scaler, lr_scheduler, sample)

            if (step_id + 1) % self.args.val_freq == 0 and dist.get_rank() == 0:
                print("Train evaluation.......")
                model.eval()
                new_loss = self.evaluate_nsteps(
                    model, train_loader, step_id,
                    val_iters=10,
                    split='train'
                )
                print("Test evaluation.......")
                new_loss = self.evaluate_nsteps(
                    model, val_loader, step_id,
                    val_iters=-1
                )
                # save model
                best_loss = self.save_checkpoint(
                    model, optimizer, step_id,
                    new_loss, best_loss
                )
                model.train()
            dist.barrier()

        return model

    def _run_depth2cloud(self, sample):
        return None  # implemented in child classes

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        # Actions
        if self.args.keypose_only:
            sample["action"] = sample["action"][:, [-1]]
            sample["action_mask"] = sample["action_mask"][:, [-1]]

        # Observations
        pcds = self._run_depth2cloud(sample)
        if augment:
            b, nc, _, h, w = sample['rgb'].shape
            obs = torch.cat((
                sample['rgb'].cuda(non_blocking=True).half() / 255,
                pcds.half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)
            obs = self.aug(obs)
            rgbs = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcds = obs[:, 3:].reshape(b, nc, 3, h, w).float()
        else:
            rgbs = sample['rgb'].cuda(non_blocking=True).float() / 255

        return (
            sample["action"].cuda(non_blocking=True),
            sample["action_mask"].cuda(non_blocking=True),
            rgbs,
            None,
            pcds,
            sample["instr"],
            sample["proprioception"].cuda(non_blocking=True)
        )

    def train_one_step(self, model, optimizer, scaler, lr_scheduler, sample):
        """Run a single training step."""
        optimizer.zero_grad()

        # Forward pass
        action, action_mask, rgbs, rgb2d, pcds, instr, prop = self.prepare_batch(
            sample, augment=True
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(action, action_mask, rgbs, rgb2d, pcds, instr, prop)

        # Backward pass
        scaler.scale(loss).backward()

        # Update
        scaler.step(optimizer)
        scaler.update()

        # Step the lr scheduler
        lr_scheduler.step()

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if i == val_iters:
                break

            action, action_mask, rgbs, rgb2d, pcds, instr, prop = self.prepare_batch(
                sample, augment=False
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_action = model(
                    action, action_mask, rgbs, rgb2d, pcds, instr, prop,
                    run_inference=True
                )

            losses, losses_B = compute_metrics(pred_action, action)

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
                viz = generate_visualizations(pred_action, action)
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

        return -values[f'{split}-losses/mean/traj_pos_acc_001']

    def load_checkpoint(self, model, optimizer):
        """Load from checkpoint."""
        print("=> trying checkpoint '{}'".format(self.args.checkpoint))
        if not os.path.exists(self.args.checkpoint):
            print('Warning: checkpoint was not found, starting from scratch')
            print('The main process will compute workspace bounds')
            return 0, None

        model_dict = torch.load(
            self.args.checkpoint,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(model_dict["weight"])
        if 'optimizer' in model_dict:
            optimizer.load_state_dict(model_dict["optimizer"])
        start_iter = model_dict.get("iter", 0)
        best_loss = model_dict.get("best_loss", None)

        print("=> loaded successfully '{}' (step {})".format(
            self.args.checkpoint, model_dict.get("iter", 0)
        ))
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_loss

    def save_checkpoint(self, model, optimizer, step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        if new_loss is None or best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            torch.save({
                "weight": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / "best.pth")
        torch.save({
            "weight": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / "last.pth")
        if (step_id + 1) % 40000 == 0:
            torch.save({
                "weight": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / f"interm{step_id + 1}.pth")
        return best_loss


def traj_collate_fn(batch):
    _dict = {}

    # Values for these come as lists
    list_keys = ["task", "instr"]
    for key in list_keys:
        _dict[key] = [item[key][0] for item in batch]

    # Treat rest as tensors
    _dict.update({
        k_: (
            torch.stack([item[k_] for item in batch])
            if batch[0][k_] is not None else None
        )
        for k_ in batch[0].keys() if k_ not in list_keys
    })
    # Append action_mask for inference
    _dict["action_mask"] = torch.zeros(_dict["action"].shape[:-1], dtype=bool)

    return _dict


def actions_collate_fn(batch):
    return {"action": torch.stack([item["action"] for item in batch])}
