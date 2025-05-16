import os
import random

from kornia import augmentation as K
import numpy as np
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

from modeling.encoder.text import fetch_tokenizers
from utils.common_utils import count_parameters
from utils.pytorch3d_transforms import relative_to_absolute
from ..schedulers import TriStageLRScheduler
from ..utils import compute_metrics


class BaseTrainTester:
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args, dataset_cls, model_cls, depth2cloud, im_size):
        """Initialize."""
        self.args = args
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls
        self.depth2cloud = depth2cloud
        self.aug = K.AugmentationSequential(
            # K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(
                degrees=0,
                translate=0.0,
                scale=(0.75, 1.25),
                padding_mode="reflection",
                p=0.8
            ),
            # K.RandomRotation((-5, 5), p=0.3),
            K.RandomResizedCrop(
                size=(im_size, im_size),
                scale=(0.95, 1.05),
                p=0.1
            )
        ).cuda()

        if dist.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=args.log_dir)

    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            relative_action=self.args.relative_action,
            mem_limit=self.args.memory_limit,
            chunk_size=self.args.chunk_size
        )
        val_dataset = self.dataset_cls(
            root=self.args.eval_data_dir,
            instructions=self.args.val_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            chunk_size=self.args.chunk_size
        )
        return train_dataset, val_dataset

    def get_loaders(self):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            # np.random.seed(np.random.get_state()[1][0] + worker_id)
        # Datasets
        train_dataset, val_dataset = self.get_datasets()
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        train_sampler = DistributedSampler(train_dataset, drop_last=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size // self.args.chunk_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=base_collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g,
            prefetch_factor=4,
            persistent_workers=True
        )
        # No sampler for val!
        if dist.get_rank() == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size_val // self.args.chunk_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=base_collate_fn,
                pin_memory=True,
                sampler=None,
                drop_last=False,
                prefetch_factor=4,
                persistent_workers=True
            )
        else:
            val_loader = None
        return train_loader, val_loader, train_sampler

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = self.model_cls(
            backbone=self.args.backbone,
            output_level=self.args.output_level,
            upsample=self.args.upsample,
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
        count_parameters(_model)
        # Somehow necessary for torch.compile to work without DDP complaining:
        if hasattr(_model, 'encoder') and hasattr(_model.encoder, 'feature_pyramid'):
            _model.encoder.feature_pyramid = _model.encoder.feature_pyramid.to(
                memory_format=torch.channels_last
            )

        return _model

    @torch.no_grad()
    def get_workspace_normalizer(self, ndims=3):
        print("Computing workspace normalizer...")

        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            actions_only=True,
            chunk_size=self.args.chunk_size
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=max(self.args.batch_size, 64) // self.args.chunk_size,
            collate_fn=actions_collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers
        )

        # Loop and compute action min-max
        min_, max_ = torch.ones(ndims) * 10000, -torch.ones(ndims) * 10000
        for sample in tqdm(data_loader):
            action = sample["action"][..., :ndims].reshape([-1, ndims])
            min_ = torch.min(min_, action.min(0).values)
            max_ = torch.max(max_, action.max(0).values)

        min_ = min_ - self.args.workspace_normalizer_buffer
        max_ = max_ + self.args.workspace_normalizer_buffer

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
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", 'norm']
        for name, param in model.named_parameters():
            if self.args.finetune_backbone and 'backbone' in name:
                optimizer_grouped_parameters[2]["params"].append(param)
            elif any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95)
        )
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
        train_loader, val_loader, train_sampler = self.get_loaders()
        # # Warmup
        # for sample in tqdm(train_loader):
        #     pass

        # Get model
        model = self.get_model()
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        if not self.args.checkpoint or not os.path.exists(self.args.checkpoint):
            normalizer = self.get_workspace_normalizer()
            model.workspace_normalizer.copy_(normalizer)
            dist.barrier(device_ids=[torch.cuda.current_device()])

        # Get optimizer
        optimizer = self.get_optimizer(model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        scaler = torch.GradScaler()

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        # make sure to compile before DDP!
        if self.args.use_compile:
            model.compute_loss = torch.compile(model.compute_loss, fullgraph=True)
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
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return model

        # Step the lr scheduler to the current step
        for _ in range(start_iter):
            lr_scheduler.step()

        # Training loop
        model.train()
        samples_per_epoch = len(train_loader)
        epoch = start_iter // samples_per_epoch + 1
        train_sampler.set_epoch(epoch)
        iter_loader = iter(train_loader)
        for step_id in trange(start_iter, self.args.train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                # when the iterator is exhausted, we need to reset it
                # and increment the epoch
                epoch += 1
                train_sampler.set_epoch(epoch)
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
            dist.barrier(device_ids=[torch.cuda.current_device()])

        return model

    def _run_depth2cloud(self, sample):
        return None  # implemented in child classes

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        # Actions
        if self.args.keypose_only:
            sample["action"] = sample["action"][:, [-1]]

        # Observations
        pcds = self._run_depth2cloud(sample)
        if augment:
            b, nc, _, h, w = sample['rgb'].shape
            obs = torch.cat((
                sample['rgb'].cuda(non_blocking=True).half() / 255,
                pcds.half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.view(-1, 6, h, w)
            obs = self.aug(obs)
            rgbs = obs[:, :3].view(b, nc, 3, h, w).float()
            pcds = obs[:, 3:].view(b, nc, 3, h, w).float()
        else:
            rgbs = sample['rgb'].cuda(non_blocking=True).float() / 255
        rgb2d = sample["rgb2d"]
        if rgb2d is not None:
            rgb2d = sample['rgb2d'].cuda(non_blocking=True).float() / 255

        # Check for history requirements
        proprio = sample["proprioception"].cuda(non_blocking=True)
        nhist_ = proprio.size(1)  # proprio is B nhist nhand 7+X
        assert nhist_ >= self.args.num_history, "not enough proprio timesteps"
        proprio = proprio[:, :max(self.args.num_history, 1)]

        return (
            sample["action"].cuda(non_blocking=True),
            torch.zeros(sample["action"].shape[:-1], dtype=bool, device='cuda'),
            rgbs,
            rgb2d,
            pcds,
            sample["instr"],
            proprio
        )

    def _model_forward(self, model, sample, training=True):
        action, action_mask, rgbs, rgb2d, pcds, instr, prop = self.prepare_batch(
            sample, augment=training
        )
        # from time import time
        # torch.cuda.synchronize()
        # start = time()
        instr = self.tokenizer(instr).cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                action, action_mask, rgbs, rgb2d, pcds, instr, prop,
                run_inference=not training
            )
        # torch.cuda.synchronize()
        # print("Time taken for forward pass: ", time() - start)
        return out  # loss if training, else action

    def train_one_step(self, model, optimizer, scaler, lr_scheduler, sample):
        """Run a single training step."""
        optimizer.zero_grad()

        # Forward pass
        loss = self._model_forward(model, sample)

        # Backward pass
        scaler.scale(loss).backward()

        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

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
            if i == val_iters or i > 1000:
                break

            pred_action = self._model_forward(model, sample, training=False)
            gt_action = sample["action"].cuda(non_blocking=True)
            if self.args.relative_action:
                pred_action = relative_to_absolute(
                    pred_action[:, :, 0],
                    sample["proprioception"].cuda(non_blocking=True)[:, :, 0],
                    qform=self.args.quaternion_format
                )
                gt_action = relative_to_absolute(
                    gt_action[:, :, 0],
                    sample["proprioception"].cuda(non_blocking=True)[:, :, 0],
                    qform=self.args.quaternion_format
                )

            losses, losses_B = compute_metrics(pred_action, gt_action)

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
        # Load weights flexibly
        msn, unxpct = model.load_state_dict(model_dict["weight"], strict=False)
        if msn:
            print(f"Missing keys (not found in checkpoint): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("All keys matched successfully!")
        # Load optimizer
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
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()

        # Best checkpoint
        if new_loss is not None and (best_loss is None or new_loss < best_loss):
            best_loss = new_loss
            torch.save({
                "weight": model_state,
                "optimizer": optimizer_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / "best.pth")

        # Last checkpoint (always saved)
        torch.save({
            "weight": model_state,
            "optimizer": optimizer_state,
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / "last.pth")

        # Save intermediate checkpoints
        if (step_id + 1) % 40000 == 0:
            torch.save({
                "weight": model_state,
                "optimizer": optimizer_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / f"interm{step_id + 1}.pth")

        return best_loss


def base_collate_fn(batch):
    _dict = {}

    # Values for these come as lists
    list_keys = ["task", "instr"]
    for key in list_keys:
        _dict[key] = []
        for item in batch:
            _dict[key].extend(item[key])

    # Treat rest as tensors
    _dict.update({
        k_: (
            torch.cat([item[k_] for item in batch])
            if batch[0][k_] is not None else None
        )
        for k_ in batch[0].keys() if k_ not in list_keys
    })

    return _dict


def actions_collate_fn(batch):
    return {"action": torch.cat([item["action"] for item in batch])}
