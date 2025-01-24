"""Shared utilities for all main scripts."""

import os
import pickle
import random
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from utils.tristage_scheduler import TriStageLRScheduler


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(self, args):
        """Initialize."""
        if dist.get_rank() == 0:
            args_dict = vars(args)
            conf = OmegaConf.create(args_dict)
            output_file = str(args.log_dir / "config.yaml")
            OmegaConf.save(conf, output_file)

        self.args = args

        if dist.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=args.log_dir)

    def get_datasets(self):
        """Initialize datasets."""
        raise NotImplementedError

    def get_loaders(self, collate_fn=default_collate):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        # Datasets
        train_dataset, test_dataset = self.get_datasets()
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
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g
        )
        # No sampler for val!
        if dist.get_rank() == 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.args.batch_size_val,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                sampler=None,
                drop_last=False
            )
        else:
            test_loader = None
        return train_loader, test_loader

    def get_model(self):
        """Initialize the model."""
        raise NotImplementedError

    def get_workspace_normalizer(self, data_loader):
        """Compute workspace normalizer."""
        raise NotImplementedError

    def get_text_encoder(self):
        """Initialize the model."""
        raise NotImplementedError

    @staticmethod
    def get_criterion():
        """Get loss criterion for training."""
        # criterion is a class, must have compute_loss and compute_metrics
        raise NotImplementedError

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
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

    def main(self, collate_fn=default_collate):
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
                print("Test evaluation.......")
                new_loss = self.evaluate_nsteps(
                    model, text_encoder, criterion, test_loader, step_id,
                    val_iters=max(5, self.args.val_iters),
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
        raise NotImplementedError

    @torch.no_grad()
    def evaluate_nsteps(self, model, text_encoder, criterion,
                        loader, step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        raise NotImplementedError

    def compute_workspace_bounds(self, dataloader):
        raise NotImplementedError

    def load_checkpoint(self, model, optimizer):
        """Load from checkpoint."""
        print("=> loading checkpoint '{}'".format(self.args.checkpoint))

        model_dict = torch.load(self.args.checkpoint, map_location="cpu",
                                weights_only=True)
        model.load_state_dict(model_dict["weight"])
        if 'optimizer' in model_dict:
            optimizer.load_state_dict(model_dict["optimizer"])
            for p in range(len(optimizer.param_groups)):
                optimizer.param_groups[p]['lr'] = self.args.lr
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
        return best_loss

    def synchronize_between_processes(self, a_dict):
        all_dicts = all_gather(a_dict)

        if not is_dist_avail_and_initialized() or dist.get_rank() == 0:
            merged = {}
            for key in all_dicts[0].keys():
                device = all_dicts[0][key].device
                merged[key] = torch.cat([
                    p[key].to(device) for p in all_dicts
                    if key in p
                ])
            a_dict = merged
        return a_dict


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)

    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty(
            (max_size,), dtype=torch.uint8, device="cuda"
        ))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,),
            dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
