from copy import deepcopy
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from modeling.encoder.text import fetch_tokenizers
from ..common_utils import count_parameters
from ..depth2cloud import fetch_depth2cloud
from ..data_preprocessors import fetch_data_preprocessor
from ..ema import EMA
from ..schedulers import fetch_scheduler
from .utils import compute_metrics
from types import SimpleNamespace
from datasets.base_dex import DexDataset
from mujoco_visualization import val_batch



class DexonomyTrainTester:
    """Train/test a trajectory optimization algorithm."""
    

    def __init__(self, args, dataset_cls, model_cls):
        """Initialize."""
        self.args = args
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls

        self.preprocessor = fetch_data_preprocessor(self.args.dataset)()

        if dist.get_rank() == 0 and not self.args.eval_only:
            self.writer = SummaryWriter(log_dir=args.log_dir)

    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        if not self.args.save_for_mujoco:
            train_dataset = self.dataset_cls(
                root=self.args.train_data_dir,
                mem_limit=self.args.memory_limit,
                chunk_size=self.args.chunk_size
            )
            val_dataset = self.dataset_cls(
                root=self.args.eval_data_dir,
                copies=1,
                mem_limit=0.1,
                chunk_size=self.args.chunk_size
            )
        if self.args.save_for_mujoco:
            object_paths = ['assets/object/DGN_5k', 'assets/object/objaverse_5k']
            dataset_config = {
                'num_workers': 8,
                'num_points': 1024,
                'joint_num': 22,
                'grasp_type_lst': ["10_Power_Disk"],
                'grasp_path': 'assets/grasp/Dexonomy_GRASP_shadow/succ_collect',
                'object_path': None,
                'split_path': 'valid_split',
                'pc_path': 'vision_data/azure_kinect_dk',  # relative to object_path
                'batch_size': self.args.batch_size
            }
            dataset_config = SimpleNamespace(**dataset_config)
            train_dataset_lst = []
            val_dataset_lst = []
            for p in object_paths:
                object_path = p
                dataset_config.object_path = object_path
                dataset_config.batch_size = self.args.batch_size
                train_dataset_lst.append(DexDataset(dataset_config, "train"))
                dataset_config.batch_size = self.args.batch_size_val
                val_dataset_lst.append(DexDataset(dataset_config, "eval"))
            train_dataset = torch.utils.data.ConcatDataset(train_dataset_lst)
            val_dataset = torch.utils.data.ConcatDataset(val_dataset_lst)
        
        return train_dataset, val_dataset

    def get_loaders(self):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Datasets
        train_dataset, val_dataset = self.get_datasets()
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        train_sampler = DistributedSampler(train_dataset, drop_last=True)
        if self.args.save_for_mujoco:
            collate_fn = None
        else:
            collate_fn = base_collate_fn
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size // self.args.chunk_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
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
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
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
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            nhist=self.args.num_history,
            nhand=2 if self.args.bimanual else 1,
            num_shared_attn_layers=self.args.num_shared_attn_layers,
            relative=self.args.relative_action,
            rotation_format=self.args.rotation_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model,
            lv2_batch_size=self.args.lv2_batch_size,
            visualize_denoising_steps=self.args.visualize_denoising_steps,
            accurate_joint_pos=self.args.accurate_joint_pos,
            guidance_weight=self.args.guidance_weight,
        )

        # Print basic modules' parameters
        if dist.get_rank() == 0:
            count_parameters(_model)

        # Somehow necessary for torch.compile to work without DDP complaining:
        if hasattr(_model, 'encoder') and hasattr(_model.encoder, 'feature_pyramid'):
            _model.encoder.feature_pyramid = _model.encoder.feature_pyramid.to(
                memory_format=torch.channels_last
            )
        # Useful for some models to ensure parameters are contiguous
        for name, param in _model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        return _model

    @torch.no_grad()
    def get_workspace_normalizer(self, ndims=3):
        print("Computing workspace normalizer...")

        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            mem_limit=self.args.memory_limit,
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
            action = sample["grasp_qpos"][..., :ndims].reshape([-1, ndims])
            min_ = torch.min(min_, action.min(0).values)
            max_ = torch.max(max_, action.max(0).values)
        print(f"Action min: {min_}, max: {max_}") #  min: tensor([-0.2550, -0.2662, -0.2713]), max: tensor([0.2718, 0.2776, 0.2732])
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
                {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
            )
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]  # , 'norm']
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

    def main(self):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, val_loader, train_sampler = self.get_loaders()

        # Get model
        model = self.get_model()
        # self.tokenizer = fetch_tokenizers(self.args.backbone)
        if not os.path.exists(self.args.checkpoint):
            # normalizer = self.get_workspace_normalizer()
            # model.workspace_normalizer.copy_(normalizer)
            dist.barrier(device_ids=[torch.cuda.current_device()])
        # Get optimizer
        optimizer = self.get_optimizer(model)
        lr_scheduler = fetch_scheduler(
            self.args.lr_scheduler, optimizer, self.args.train_iters
        )
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


        # Initialize EMA copy
        ema_model = deepcopy(model)
        self.ema = EMA()


        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            start_iter, best_loss = self.load_checkpoint(model, ema_model, optimizer)
        print(model.module.workspace_normalizer)

        # Eval only
        if self.args.eval_only:
            if dist.get_rank() == 0:
                print("Test evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id=-1,
                    val_iters=-1
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return ema_model if self.args.use_ema else model
        
        if self.args.eval_overfit:
            if dist.get_rank() == 0:
                print("Testing overfitting on the training dataset ...")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    train_loader, step_id=-1,
                    val_iters=-1, split='train'
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return ema_model if self.args.use_ema else model

        # Step the lr scheduler to the current step
        for _ in range(start_iter):
            lr_scheduler.step()

        # Step the sampler to the currect "epoch"
        samples_per_epoch = len(train_loader)
        epoch = start_iter // samples_per_epoch + 1
        train_sampler.set_epoch(epoch)  # ensures new batches are sampled

        # Training loop
        model.train()
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
                
            

            loss_value = self.train_one_step(model, optimizer, scaler, lr_scheduler, sample)
            # print(model.module.report_timers(reset=True))
            self.ema.step(model, ema_model, self.args.use_ema, step_id)
            
            # Log loss
            if dist.get_rank() == 0:
                if step_id % 200 == 0:
                    print(f"Step {step_id}: loss = {loss_value:.4f}")
                    if self.writer is not None:
                        self.writer.add_scalar(
                            "train/loss", loss_value, step_id
                        )

            if (step_id + 1) % self.args.val_freq == 0 and dist.get_rank() == 0:
                print("Train evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    train_loader, step_id,
                    val_iters=2,
                    split='train'
                )
                print("Test evaluation.......")
                new_loss = self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id,
                    # val_iters=1250
                    val_iters=10
                )
                # save model
                best_loss = self.save_checkpoint(
                    model, ema_model, optimizer, step_id,
                    new_loss, best_loss
                )
                model.train()
            dist.barrier(device_ids=[torch.cuda.current_device()])

        return ema_model if self.args.use_ema else model

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        # print(sample["partial_points"].shape)
        # print(sample["grasp_qpos"].shape)
        # print(sample["pregrasp_qpos"].shape)
        # print(sample["squeeze_qpos"].shape)
        # print(sample["grasp_type_id"].shape)
        out = {}
        out['partial_points'] = sample['partial_points'].cuda(non_blocking=True).float()
        out['grasp_qpos'] = sample['grasp_qpos'].cuda(non_blocking=True)
        out['pregrasp_qpos'] = sample['pregrasp_qpos'].cuda(non_blocking=True)
        out['squeeze_qpos'] = sample['squeeze_qpos'].cuda(non_blocking=True)
        # out['grasp_type_id'] = sample['grasp_type_id'].cuda(non_blocking=True)
        gtype = sample['grasp_type_id']
        # If it comes as shape (B,1), squeeze; if already (B,), leave it.
        if gtype.dim() == 2 and gtype.size(-1) == 1:
            gtype = gtype.squeeze(-1)
        out['grasp_type_id'] = gtype.to(device='cuda', dtype=torch.long, non_blocking=True)
        # print(sample['anchor_visible'].shape, sample['anchor_visible'])
        out['anchor_visible'] = sample['anchor_visible'].cuda(non_blocking=True)
        out['obj_pose'] = sample['obj_pose'].cuda(non_blocking=True)
        out['obj_scale'] = sample['obj_scale'].cuda(non_blocking=True)
        out['obj_path'] = sample['obj_path']
        
        # out = self.preprocessor.translate_to_center_frame(out)
        out = self.preprocessor.wild_parallel_augment(out)
        return out

    def _model_forward(self, model, batch, training=True):

        # if self.args.pre_tokenize:
        #     instr = self.tokenizer(instr).cuda(non_blocking=True)
        
        # save the batch for debugging
        if dist.get_rank() == 0 and training:
            if not os.path.exists(self.args.log_dir / "batch_inspect.pt"):
                torch.save(batch, self.args.log_dir / "batch_inspect.pt")
                print("Saved batch for inspection at", self.args.log_dir / "batch_inspect.pt")
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            
            if self.args.condition_on_grasp_type_id:
                grasp_type_id = batch['grasp_type_id']
            else:
                grasp_type_id = None
                print('not using grasp type id, pure unconditional path')
            out = model(batch['grasp_qpos'],
                        batch['pregrasp_qpos'],
                        batch['squeeze_qpos'],
                        batch['partial_points'],
                        focus_idx=batch['anchor_visible'],
                        grasp_type_id=grasp_type_id,
                        run_inference=not training
                        )
        return out  # loss if training, else action

    def train_one_step(self, model, optimizer, scaler, lr_scheduler, sample):
        # t0 = torch.cuda.Event(enable_timing=True)
        # t0.record()
        """Run a single training step."""
        
        sample = self.prepare_batch(sample)
        
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
        
        # t1 = torch.cuda.Event(enable_timing=True)
        # t1.record()
        # torch.cuda.synchronize()
        # elapsed_time = t0.elapsed_time(t1)
        # print(f"Step time: {elapsed_time:.2f} ms")
        return loss.item()

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        values = {}
        device = next(model.parameters()).device
        model.eval()

        mujoco_succ_num, mujoco_fail_num = 0, 0
        mujoco_succ_num_anchor, mujoco_fail_num_anchor = 0, 0
        mujoco_succ_num_multitype, mujoco_fail_num_multitype = 0, 0
        for i, sample in tqdm(enumerate(loader)):
            if i == val_iters:
                break
            
            sample = self.prepare_batch(sample)
            
            if self.args.val_set_all_anchor and self.args.test_mujoco and step_id >= 1 and split == 'val':
                print('Anchor conditioned validation...')
                anchor_sample = deepcopy(sample)
                anchor_sample['anchor_visible'] = torch.ones_like(anchor_sample['anchor_visible'], dtype=torch.bool)
                pred_anchored_grasp, pred_anchored_pregrasp, pred_anchored_squeeze = self._model_forward(model, anchor_sample, training=False)
                
                anchor_sample_to_save = {}
                anchor_sample_to_save["grasp_qpos"] = pred_anchored_grasp.cpu().numpy()
                anchor_sample_to_save["pregrasp_qpos"] = pred_anchored_pregrasp.cpu().numpy()
                anchor_sample_to_save["squeeze_qpos"] = pred_anchored_squeeze.cpu().numpy()
                anchor_sample_to_save["partial_points"] = anchor_sample["partial_points"].cpu().numpy()
                anchor_sample_to_save["anchor_visible"] = anchor_sample["anchor_visible"].cpu().numpy()
                anchor_sample_to_save["grasp_type_id"] = anchor_sample["grasp_type_id"].cpu().numpy()
                anchor_sample_to_save["obj_path"] = anchor_sample["obj_path"]
                anchor_sample_to_save["obj_scale"] = anchor_sample["obj_scale"].cpu().numpy()
                anchor_sample_to_save["obj_pose"] = anchor_sample["obj_pose"].cpu().numpy()
                
                succ_num_batch_anchor, fail_num_batch_anchor = val_batch(anchor_sample_to_save, self.args.log_dir, self.args.vis_freq)
                mujoco_succ_num_anchor += succ_num_batch_anchor
                mujoco_fail_num_anchor += fail_num_batch_anchor
                    
            if True and self.args.test_mujoco and step_id >= 1 and split == 'val':
                print('Multitype validation...')
                anchor_sample = deepcopy(sample)
                orig = anchor_sample["grasp_type_id"]
                grasp_type_ids = torch.tensor(
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 20, 22, 23, 24, 25, 26,
                    27, 28, 29, 2, 30, 31, 32, 33, 3, 4, 5, 6, 7, 8, 9],
                    dtype=orig.dtype,
                    device=orig.device
                )
                rand_indices = torch.randint(
                    low=0,
                    high=grasp_type_ids.shape[0],
                    size=orig.shape,
                    device=orig.device
                )
                anchor_sample["grasp_type_id"] = grasp_type_ids[rand_indices]
                pred_anchored_grasp, pred_anchored_pregrasp, pred_anchored_squeeze = self._model_forward(model, anchor_sample, training=False)
                
                anchor_sample_to_save = {}
                anchor_sample_to_save["grasp_qpos"] = pred_anchored_grasp.cpu().numpy()
                anchor_sample_to_save["pregrasp_qpos"] = pred_anchored_pregrasp.cpu().numpy()
                anchor_sample_to_save["squeeze_qpos"] = pred_anchored_squeeze.cpu().numpy()
                anchor_sample_to_save["partial_points"] = anchor_sample["partial_points"].cpu().numpy()
                anchor_sample_to_save["anchor_visible"] = anchor_sample["anchor_visible"].cpu().numpy()
                anchor_sample_to_save["grasp_type_id"] = anchor_sample["grasp_type_id"].cpu().numpy()
                anchor_sample_to_save["obj_path"] = anchor_sample["obj_path"]
                anchor_sample_to_save["obj_scale"] = anchor_sample["obj_scale"].cpu().numpy()
                anchor_sample_to_save["obj_pose"] = anchor_sample["obj_pose"].cpu().numpy()
                
                succ_num_batch_multitype, fail_num_batch_multitype = val_batch(anchor_sample_to_save, self.args.log_dir, self.args.vis_freq)
                mujoco_succ_num_multitype += succ_num_batch_multitype
                mujoco_fail_num_multitype += fail_num_batch_multitype

            print('Regular validation...')
            pred_grasp, pred_pregrasp, pred_squeeze = self._model_forward(model, sample, training=False)
            
            if self.args.visualize_denoising_steps:
                # Save visualization data
                if dist.get_rank() == 0:
                    visualization_data = model.module.visualization_data
                    # print(visualization_data.keys())
                    # print(visualization_data["grasps"].shape)
                    # print(sample["grasp_qpos"].shape)
                    # print(visualization_data["partial_points"].shape)
                    visualization_data['grasps'] = np.concatenate(
                        [np.expand_dims(sample["grasp_qpos"].cpu().numpy(), axis=1), visualization_data["grasps"]],
                        axis=1
                    )
                    visualization_data['joint_positions'] = np.concatenate(
                        [np.expand_dims(model.module.fk_layer(sample["grasp_qpos"], accurate_pos=True).cpu().numpy(), axis=1), visualization_data["joint_positions"]],
                        axis=1
                    )
                    # find the dir of checkpoint
                    save_vis_path = os.path.join(os.path.dirname(self.args.checkpoint), f"visualization_denoise_process_batch{i}.npz")
                    print(f"Saving visualization data to {save_vis_path}")
                    np.savez(
                        save_vis_path,
                        grasps=visualization_data["grasps"],
                        partial_points=visualization_data["partial_points"],
                        joint_positions=visualization_data["joint_positions"]
                    )
            gt_grasp = sample["grasp_qpos"].cuda(non_blocking=True)
            gt_pregrasp = sample["pregrasp_qpos"].cuda(non_blocking=True)
            gt_squeeze = sample["squeeze_qpos"].cuda(non_blocking=True)
            
            
            sample_to_save = {}
            sample_to_save["grasp_qpos"] = pred_grasp.cpu().numpy()
            sample_to_save["pregrasp_qpos"] = pred_pregrasp.cpu().numpy()
            sample_to_save["squeeze_qpos"] = pred_squeeze.cpu().numpy()
            sample_to_save["partial_points"] = sample["partial_points"].cpu().numpy()
            sample_to_save["anchor_visible"] = sample["anchor_visible"].cpu().numpy()
            sample_to_save["grasp_type_id"] = sample["grasp_type_id"].cpu().numpy()
            sample_to_save["obj_path"] = sample["obj_path"]
            sample_to_save["obj_scale"] = sample["obj_scale"].cpu().numpy()
            sample_to_save["obj_pose"] = sample["obj_pose"].cpu().numpy()
            
            if self.args.test_mujoco:
                succ_num_batch, fail_num_batch = val_batch(sample_to_save, self.args.log_dir, self.args.vis_freq)
                mujoco_succ_num += succ_num_batch
                mujoco_fail_num += fail_num_batch
            
            
            if self.args.save_for_mujoco:
                # Save the sample to a file
                save_path_dir = os.path.join(os.path.dirname(self.args.checkpoint), 'mujoco_samples')
                os.makedirs(save_path_dir, exist_ok=True)
                save_path = os.path.join(save_path_dir, f"sample_{split}_{i}.npz")
                print(f"Saving sample to {save_path}")
                np.savez(save_path, **sample_to_save)
                # save the ground truth
                gt_save_path = os.path.join(save_path_dir, f"gt_sample_{split}_{i}.npz")
                print(f"Saving ground truth to {gt_save_path}")
                np.savez(gt_save_path,
                         grasp_qpos=gt_grasp.cpu().numpy(),
                         pregrasp_qpos=gt_pregrasp.cpu().numpy(),
                         squeeze_qpos=gt_squeeze.cpu().numpy(),
                         partial_points=sample["partial_points"].cpu().numpy(),
                         anchor_visible=sample["anchor_visible"].cpu().numpy(),
                         grasp_type_id=sample["grasp_type_id"].cpu().numpy(),
                         obj_path=sample["obj_path"],
                         obj_scale=sample["obj_scale"].cpu().numpy(),
                         obj_pose=sample["obj_pose"].cpu().numpy()
                )

            losses, losses_B = compute_metrics(pred_grasp, gt_grasp)
            
            if self.args.eval_overfit:
                print(f"Step {step_id}, batch {i}: "
                    f"losses: {losses}, losses_B: {losses_B}")

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # # Gather per-task statistics
            # grasp_id = np.array(sample["grasp_type_id"])
            # for n, l in losses_B.items():
            #     for task in np.unique(tasks):
            #         key = f"{split}-loss/{task}/{n}"
            #         l_task = l[tasks == task].mean()
            #         if key not in values:
            #             values[key] = torch.Tensor([]).to(device)
            #         values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

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
                
        # Log mujoco success rate
        if self.args.test_mujoco:
            if dist.get_rank() == 0:
                print(f"Mujoco success rate: {mujoco_succ_num / (mujoco_succ_num + mujoco_fail_num):.03f} "
                    f"({mujoco_succ_num}/{mujoco_fail_num})")
                self.writer.add_scalar(f"{split}-losses/mujoco_success_rate", 
                                    mujoco_succ_num / (mujoco_succ_num + mujoco_fail_num), step_id)
                print(f"Mujoco success rate (anchor): {mujoco_succ_num_anchor / (mujoco_succ_num_anchor + mujoco_fail_num_anchor + 1e-3):.03f} "
                    f"({mujoco_succ_num_anchor}/{mujoco_fail_num_anchor})")
                self.writer.add_scalar(f"{split}-losses/mujoco_success_rate_anchor",
                                    mujoco_succ_num_anchor / (mujoco_succ_num_anchor + mujoco_fail_num_anchor + 1e-3), step_id)
                print(f"Mujoco success rate (multitype): {mujoco_succ_num_multitype / (mujoco_succ_num_multitype + mujoco_fail_num_multitype + 1e-3):.03f} "
                    f"({mujoco_succ_num_multitype}/{mujoco_fail_num_multitype})")
                self.writer.add_scalar(f"{split}-losses/mujoco_success_rate_multitype",
                                    mujoco_succ_num_multitype / (mujoco_succ_num_multitype + mujoco_fail_num_multitype + 1e-3), step_id)

        return -values[f'{split}-losses/mean/grasp_pos_acc_001']

    def load_checkpoint(self, model, ema_model, optimizer):
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
        # EMA weights
        if model_dict.get("ema_weight") is not None:
            ema_model.load_state_dict(model_dict["ema_weight"], strict=True)
        # Useful for resuming training
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

    def save_checkpoint(self, model, ema_model, optimizer,
                        step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        model_state = model.state_dict()
        ema_state = ema_model.state_dict() if self.args.use_ema else None

        # Best checkpoint
        if best_loss is None or new_loss < best_loss:
            best_loss = new_loss
            torch.save({
                "weight": model_state,
                "ema_weight": ema_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / "best.pth")

        # Last checkpoint (always saved)
        torch.save({
            "weight": model_state,
            "ema_weight": ema_state,
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / "last.pth")

        # Save intermediate checkpoints
        if (step_id + 1) % 100000 == 0:
            torch.save({
                "weight": model_state,
                "ema_weight": ema_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / f"interm{step_id + 1}.pth")

        return best_loss


def base_collate_fn(batch):
    _dict = {}

    # Values for these come as lists
    list_keys = ['obj_path']
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
    return {"grasp_qpos": torch.cat([item["grasp_qpos"] for item in batch])}


def relative_to_absolute(action, proprio):
    # action (B, T, 8), proprio (B, 1, 7)
    pos = proprio[..., :3] + action[..., :3].cumsum(1)

    orn = proprio[..., 3:6] + action[..., 3:6].cumsum(1)
    orn = (orn + torch.pi) % (2 * torch.pi) - torch.pi

    return torch.cat([pos, orn, action[..., 6:]], -1)


def from_delta_action(deltas, anchor_action, qform='xyzw'):
    """
    Reconstruct absolute actions from deltas and initial anchor action.

    Args:
        deltas: (..., N, 8) — delta actions (relative to previous timestep)
        anchor_action: (..., 1, 8) — starting pose
        qform: 'xyzw' or 'wxyz'

    Returns:
        actions: (..., N, 8) — absolute action trajectory
    """
    assert deltas.shape[-1] == 8
    abs_actions = [anchor_action.squeeze(-2)]  # (..., 8)

    for t in range(deltas.shape[-2]):
        prev = abs_actions[-1]
        delta = deltas[..., t, :]

        # Position update
        pos = prev[..., :3] + delta[..., :3]

        # Quaternion update
        if qform == 'xyzw':
            prev_q = prev[..., [6, 3, 4, 5]]
            delta_q = delta[..., [6, 3, 4, 5]]
            new_q = pytorch3d_transforms.quaternion_multiply(delta_q, prev_q)[..., [1, 2, 3, 0]]
        elif qform == 'wxyz':
            prev_q = prev[..., 3:7]
            delta_q = delta[..., 3:7]
            new_q = pytorch3d_transforms.quaternion_multiply(delta_q, prev_q)
        else:
            raise ValueError("Invalid quaternion format")

        # Gripper remains as-is (no delta)
        grip = delta[..., -1:]

        new_action = torch.cat([pos, new_q, grip], dim=-1)
        abs_actions.append(new_action)

    # Stack and remove the anchor (first entry)
    actions = torch.stack(abs_actions[1:], dim=-2)  # (..., N, 8)

    return actions
