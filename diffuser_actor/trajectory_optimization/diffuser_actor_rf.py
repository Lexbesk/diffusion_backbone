import torch
import torch.nn.functional as F

from diffuser_actor.utils.schedulers.scheduling_rf import RFScheduler
from .diffuser_actor import DiffuserActor


class RFDiffuserActor(DiffuserActor):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 point_sampling='fps',
                 gripper_loc_bounds=None,
                 rotation_parametrization='6D',
                 quaternion_format='xyzw',
                 diffusion_timesteps=100,
                 nhist=3,
                 relative=False,
                 lang_enhanced=False):
        super().__init__(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            use_instruction=use_instruction,
            fps_subsampling_factor=fps_subsampling_factor,
            point_sampling=point_sampling,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=rotation_parametrization,
            quaternion_format=quaternion_format,
            diffusion_timesteps=diffusion_timesteps,
            nhist=nhist,
            relative=relative,
            lang_enhanced=lang_enhanced
        )

        self.position_noise_scheduler = RFScheduler(
            num_train_timesteps=diffusion_timesteps,
            timestep_spacing="linspace",
            noise_sampler="logit_normal",
            noise_sampler_config={'mean': 0, 'std': 1},
        )
        self.rotation_noise_scheduler = RFScheduler(
            num_train_timesteps=diffusion_timesteps,
            timestep_spacing="linspace",
            noise_sampler="logit_normal",
            noise_sampler_config={'mean': 0, 'std': 1},
        )

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
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

        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
        if self._relative:
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = self.position_noise_scheduler.sample_noise_step(
            num_noise=len(noise), device=noise.device
        )

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3],
            timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9],
            timesteps
        )
        noisy_trajectory = torch.cat((pos, rot), -1)
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
        assert not cond_mask.any()

        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps, fixed_inputs
        )

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:9]
            target = noise - gt_trajectory
            loss = (
                30 * F.l1_loss(trans, target[..., :3], reduction='mean')
                + 10 * F.l1_loss(rot, target[..., 3:9], reduction='mean')
            )
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 9:]
                loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
            total_loss = total_loss + loss
        return total_loss
