import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np

from diffuser_actor.noise_scheduler.rectified_flow import RFScheduler
from diffuser_actor.noise_scheduler.ddpm import DDPMScheduler
from diffuser_actor.encoder.encoder import Encoder
# from .denoise_actor import TransformerHead
from .denoise_actor import DenoiseActor as BaseDenoiseActor
from diffuser_actor.utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)


class DenoiseActor(nn.Module):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 rotation_parametrization='6D',
                 quaternion_format='xyzw',
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 nhist=3,
                 relative=False):
        super().__init__()
        self.non_gripper_actor = BaseDenoiseActor(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            use_instruction=use_instruction,
            fps_subsampling_factor=fps_subsampling_factor,
            rotation_parametrization=rotation_parametrization,
            quaternion_format=quaternion_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            nhist=nhist,
            relative=relative
        )
        self.gripper_actor = BaseDenoiseActor(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            use_instruction=use_instruction,
            fps_subsampling_factor=fps_subsampling_factor,
            rotation_parametrization=rotation_parametrization,
            quaternion_format=quaternion_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            nhist=nhist,
            relative=relative
        )
        self.workspace_normalizer = nn.Parameter(
            torch.Tensor([[0., 0., 0.], [1., 1., 1.]]),
            requires_grad=False
        )
        del self.non_gripper_actor.workspace_normalizer
        del self.gripper_actor.workspace_normalizer
        self.non_gripper_actor.workspace_normalizer = self.workspace_normalizer
        self.gripper_actor.workspace_normalizer = self.workspace_normalizer

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        is_gripper_camera,
        run_inference=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+4+X)
            is_gripper_camera: a boolean tensor of shape (num_cameras,)

        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            gripper_action = self.gripper_actor(
                gt_trajectory,
                trajectory_mask,
                rgb_obs[:, is_gripper_camera],
                pcd_obs[:, is_gripper_camera],
                instruction,
                curr_gripper,
                run_inference=True
            )
            non_gripper_action = self.non_gripper_actor(
                gt_trajectory,
                trajectory_mask,
                rgb_obs[:, ~is_gripper_camera],
                pcd_obs[:, ~is_gripper_camera],
                instruction,
                curr_gripper,
                run_inference=True
            )
            return gripper_action, non_gripper_action

        gripper_loss = self.gripper_actor(
            gt_trajectory,
            trajectory_mask,
            rgb_obs[:, is_gripper_camera],
            pcd_obs[:, is_gripper_camera],
            instruction,
            curr_gripper,
            run_inference=False      
        )
        non_gripper_loss = self.non_gripper_actor(
            gt_trajectory,
            trajectory_mask,
            rgb_obs[:, ~is_gripper_camera],
            pcd_obs[:, ~is_gripper_camera],
            instruction,
            curr_gripper,
            run_inference=False      
        )

        return gripper_loss + non_gripper_loss