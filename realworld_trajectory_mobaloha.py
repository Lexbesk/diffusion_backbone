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

import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scipy.spatial.transform import Rotation
from utils.utils_with_mobaloha import to_absolute_action

class Arguments(BaseArguments):
    instructions: Optional[Path] = None


class Tester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # Get model
        self.model = self.get_model()
        self.model.eval()

        # Get criterion
        self.criterion = self.get_criterion()

        # Get optimizer
        self.optimizer = self.get_optimizer(self.model)

        # Move model to devices
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model = DistributedDataParallel(
            self.model, device_ids=[self.args.local_rank],
            broadcast_buffers=False, find_unused_parameters=True
        )
        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            assert os.path.isfile(self.args.checkpoint)
            start_iter, best_loss = self.load_checkpoint(self.model, self.optimizer)


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

    def get_transform(self, trans_7D):
        '''
        trans_7D: N*7

        return N*4*4
        '''

        quat = trans_7D[:, 3:7]
        trans = trans_7D[:, 0:3]
        # print("trans: ", trans.shape)
        trans = trans.reshape(-1,3)
        rot = Rotation.from_quat(quat)
        rot_mat = rot.as_matrix()
        transform_matrix = np.zeros( (trans_7D.shape[0], 4, 4))
        transform_matrix[:,3,3] = 1.0
        transform_matrix[:,0:3,0:3] = rot_mat
        transform_matrix[:,0:3,3] = trans
        return transform_matrix

    def get_7D_tranform(self, transform_matrix):
        '''
        transform: N*4*4

        return N*7
        '''
        trans_7D = np.zeros( (transform_matrix.shape[0], 7))

        rot_mat = transform_matrix[:,0:3,0:3]
        rot = Rotation.from_matrix(rot_mat)
        rot_quat = rot.as_quat()
        trans_7D[:,3:7] = rot_quat

        trans = transform_matrix[:,0:3,3]
        trans = trans.reshape(-1,3)
        trans_7D[:,0:3] = trans
        return trans_7D

    @torch.no_grad()
    def run(self, rgb_obs, pcd_obs, curr_gripper, instruction):
        '''
        output is (N,2,8)
        '''
        device = next(self.model.parameters()).device

        rgb_obs = rgb_obs.to(torch.float32)
        rgb_obs = rgb_obs.to(device)

        pcd_obs = pcd_obs.to(torch.float32)
        pcd_obs = pcd_obs.to(device)

        curr_gripper = curr_gripper.to(torch.float32)
        curr_gripper = curr_gripper.to(device)

        instruction = instruction.to(torch.float32)
        instruction = instruction.to(device)

        traj_mask = torch.zeros( (1, self.args.interpolation_length - 1) ).to(torch.float32)
        traj_mask = traj_mask.to(device)      

        # gt_trajectory = torch.zeros( (1, self.args.interpolation_length) )
        self.model.eval()
        action = self.model(
            gt_trajectory = None,
            trajectory_mask = traj_mask,
            rgb_obs = rgb_obs,
            pcd_obs = pcd_obs,
            instruction = instruction,
            curr_gripper = curr_gripper,
            run_inference=True
        )

        print("action: ", action.shape)
        if(self.args.keypose_only):
            action = action[0:1,0:1,:,:]

        if(self.args.relative_action):
            action = to_absolute_action( action.flatten(-2, -1).cpu(), curr_gripper.cpu().flatten(-2, -1)).unflatten(-1, (2, 8) )

        action_np = action.cpu().numpy()
        action_np = action_np.reshape(-1,2,8)

        # print("action_np: ", action_np.shape)
        return action_np



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
    tester = Tester(args)


