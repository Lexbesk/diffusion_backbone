import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .denoise_refactored_actor import DenoiseActor, TransformerHead
from diffuser_actor.utils.utils import normalise_quat


class BimanualDenoiseActor(DenoiseActor):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 num_attn_heads=9,
                 fps_subsampling_factor=5,
                 rotation_parametrization='6D',
                 quaternion_format='xyzw',
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 nhist=3,
                 relative=False,
                 ayush=False):
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            use_instruction=use_instruction,
            num_attn_heads=num_attn_heads,
            fps_subsampling_factor=fps_subsampling_factor,
            rotation_parametrization=rotation_parametrization,
            quaternion_format=quaternion_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            nhist=nhist * 2,
            relative=relative,
            ayush=ayush
        )
        self.traj_encoder = nn.Linear(9, embedding_dim)
        self.prediction_head = BimanualTransformerHead(
            embedding_dim=embedding_dim,
            use_instruction=use_instruction,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist * 2,
            num_attn_heads=num_attn_heads
        )

    def encode_inputs(self, visible_rgb, visible_pcd, instruction,
                      curr_gripper):
        fixed_inputs = super().encode_inputs(
            visible_rgb, visible_pcd, instruction, curr_gripper.flatten(1, 2)
        )

        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pos
        ) = fixed_inputs

        return (curr_gripper, context_feats, context, instr_feats,
                adaln_gripper_feats, fps_feats, fps_pos)

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            query_trajectory,
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pos
        ) = fixed_inputs

        # Encode point features using visual and positional estimates
        trajectory_feats = self.traj_encoder(trajectory)

        # Unnormalized trajectory
        trajectory = trajectory.clone()
        trajectory[..., :3] = self.unnormalize_pos(trajectory[..., :3])
        if self._relative:
            trajectory[..., :3] = trajectory[..., :3] + query_trajectory[:, [-1], :, :3]

        return self.prediction_head(
            trajectory_feats,
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pos=fps_pos
        )


    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper
    ):
        B, nhist, nhand, _ = curr_gripper.shape
        assert nhand == 2
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        curr_gripper = self.convert_rot(
            curr_gripper.flatten(1, 2)
        ).unflatten(1, (nhist, nhand))

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        # Condition on start-end pose
        traj_len = trajectory_mask.size(1)
        D = curr_gripper.shape[-1]
        cond_data = torch.zeros((B, traj_len, nhand, D), device=rgb_obs.device)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs
        )

        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[..., 3:7] = normalise_quat(trajectory[..., 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(
            trajectory.flatten(1, 2)
        ).unflatten(1, (traj_len, nhand))
        # unnormalize position
        trajectory[..., :3] = self.unnormalize_pos(trajectory[..., :3])
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()

        return trajectory

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
            gt_trajectory: (B, trajectory_length, 2, 3+4+X)
            trajectory_mask: (B, trajectory_length)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 2, 3+4+X)

        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
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
        B, traj_len, nhand, D = gt_trajectory.shape
        _, nhist, _, _ = curr_gripper.shape
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[..., :3] = self.normalize_pos(gt_trajectory[..., :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(
            gt_trajectory.flatten(1, 2)
        ).unflatten(1, (traj_len, nhand))
        curr_gripper = self.convert_rot(
            curr_gripper.flatten(1, 2)
        ).unflatten(1, (nhist, nhand))

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

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

        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps, fixed_inputs
        )

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:9]
            denoise_target = self.position_noise_scheduler.prepare_target(
                noise, gt_trajectory, noisy_trajectory, timesteps
            )
            loss = (
                30 * F.l1_loss(trans, denoise_target[..., :3], reduction='mean')
                + 10 * F.l1_loss(rot, denoise_target[..., 3:9], reduction='mean')
            )
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 9:]
                loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
            total_loss = total_loss + loss
        return total_loss


class BimanualTransformerHead(TransformerHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instruction=False,
                 rotation_parametrization='quat',
                 nhist=3):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            use_instruction=use_instruction,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist,
        )
        self.hand_embed = nn.Embedding(2, embedding_dim)

        del self.traj_encoder

    def forward(self, trajectory_feats, trajectory, timestep,
                context_feats, context, instr_feats, adaln_gripper_feats,
                fps_feats, fps_pos):
        """
        Arguments:
            trajectory_feats: (B, trajectory_length, 2, C)
            trajectory: (B, trajectory_length, 2, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist * 2, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
        """
        B, traj_len, nhand, C = trajectory.shape
        assert nhand == 2
        # Trajectory features
        traj_feats = trajectory_feats
        traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(len(traj_feats), 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')
        context_feats = einops.rearrange(context_feats, 'b l c -> l b c')
        adaln_gripper_feats = einops.rearrange(
            adaln_gripper_feats, 'b l c -> l b c'
        )
        pos_pred, rot_pred, openess_pred = self.prediction_head(
            trajectory[..., :3], traj_feats,
            context[..., :3], context_feats,
            timestep, adaln_gripper_feats,
            fps_feats, fps_pos,
            instr_feats
        )
        output = [
            torch.cat((pos_pred, rot_pred, openess_pred), -1)
                 .unflatten(1, (traj_len, nhand))
        ]

        return output
