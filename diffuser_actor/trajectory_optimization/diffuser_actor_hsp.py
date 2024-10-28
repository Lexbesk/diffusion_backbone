import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


from diffuser_actor.utils.encoder import Encoder
from diffuser_actor.utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)
from diffuser_actor.utils.schedulers.scheduling_rf import RFScheduler
from .diffuser_actor import DiffusionHead


class DiffuserActor(nn.Module):

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
                 diffusion_timesteps=[5, 5],
                 nhist=3,
                 relative=False,
                 lang_enhanced=False,
                 sampling_levels=[0, 0],
                 cropped_num_scene_tokens=[-1, 256]):
        super().__init__()
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self._relative = relative
        self._point_sampling = point_sampling
        self._sampling_levels = sampling_levels
        self._cropped_num_scene_tokens = cropped_num_scene_tokens
        self.use_instruction = use_instruction
        self.encoder = Encoder(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=max(sampling_levels) + 1,
            nhist=nhist,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor
        )
        self.prediction_heads = nn.ModuleList()
        self.noise_schedulers = []
        for level in range(len(sampling_levels)):
            self.prediction_heads.append(
                DiffusionHead(
                    embedding_dim=embedding_dim,
                    use_instruction=use_instruction,
                    rotation_parametrization=rotation_parametrization,
                    nhist=nhist,
                    lang_enhanced=lang_enhanced
                )
            )
            self.noise_schedulers.append(
                RFScheduler(
                    num_train_timesteps=diffusion_timesteps[level],
                    timestep_spacing="linspace",
                    noise_sampler="logit_normal",
                    noise_sampler_config={'mean': 0, 'std': 1},
                )
            )
        self.n_steps = diffusion_timesteps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def encode_inputs(self, visible_rgb, visible_pcd, instruction,
                      curr_gripper):
        """Generate multi-level visual features, contextualizing them with
        language instructions.

        Arguments:
            visible_rgb: (B, num_cameras, 3, H, W) in [0, 1]
            visible_pcd: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+4+X)
        """
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(
            visible_rgb, visible_pcd
        )

        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = self.encoder.encode_instruction(instruction)

        # Keep only low-res scale
        context_feats_pyramid = []
        context_pcds_pyramid = []
        adaln_gripper_feats_pyramid = []
        for level in self._sampling_levels:
            _, ncam, _, h, w = rgb_feats_pyramid[level].shape
            context_feats = einops.rearrange(
                rgb_feats_pyramid[level],
                "b ncam c h w -> b (ncam h w) c"
            )

            context_pcds = pcd_pyramid[level]

            # Cross-attention vision to language
            if self.use_instruction:
                # Attention from vision to language
                context_feats = self.encoder.vision_language_attention(
                    context_feats, instr_feats
                )

            # Encode gripper history (B, nhist, F)
            adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
                curr_gripper, context_feats, context_pcds
            )
            
            context_feats = einops.rearrange(
                context_feats, "b (ncam h w) c -> b ncam h w c",
                ncam=ncam, h=h, w=w
            )
            context_pcds = einops.rearrange(
                context_pcds, "b (ncam h w) c -> b ncam h w c",
                ncam=ncam, h=h, w=w
            )

            context_feats_pyramid.append(context_feats)
            context_pcds_pyramid.append(context_pcds)
            adaln_gripper_feats_pyramid.append(adaln_gripper_feats)

        return (
            context_feats_pyramid, context_pcds_pyramid,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats_pyramid,  # gripper history features
        )

    def _prepare_fixed_inputs(self, fixed_inputs, anchor, sampling_level):
        """Select the fixed inputs for the current sampling level.
        Also, crop the scene tokens based on the input trajectory position.

        Arguments:
            fixed_inputs: A tuple with 4 elements:
                context_feats_pyramid: A list of tensors of shape
                    (B, ncam, H, W, F), indicating the scene features
                context: A list of tensors of shape (B, ncam, H, W, F),
                    indicating the scene point cloud
                instr_feats: A tensor of shape (B, max_instruction_length, F)
                adaln_gripper_feats: A list of tensors of shape (B, nhist, F),
                    indicating the gripper history features
            anchor: A tensor of shape (B, 3+4+X), indicating the previous-level
                estimate of the end-effector pose
            sampling_level: An integer indicating the current sampling level
        
        Returns:
            selected_fixed_inputs: A tuple with 5 elements:
                context_feats: (B, ncam, H, W, F)
                context: (B, ncam, H, W, F)
                instr_feats: (B, max_instruction_length, F)
                adaln_gripper_feats: (B, nhist, F)
                sampling_level: int
        """
 
        context_feats = fixed_inputs[0][sampling_level]
        context = fixed_inputs[1][sampling_level]
        instr_feats = fixed_inputs[2] if self.use_instruction else None
        adaln_gripper_feats = fixed_inputs[3][sampling_level]
        context_feats, context = self._crop_scene(
            context_feats, context, anchor, sampling_level
        )

        processed_fixed_inputs = (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            sampling_level
        )

        return processed_fixed_inputs

    def _crop_scene(self, context_feats, context, anchor, sampling_level):
        """Select the fixed inputs for the current sampling level.
        Also, crop the scene tokens based on the input trajectory position.

        Arguments:
            context_feats: (B, ncam, H, W, F)
            context: (B, ncam, H, W, F),
            anchor: (B, 3+4+X)
            sampling_level: int
        """
        context_feats = einops.rearrange(
            context_feats, "b ncam h w c -> b (ncam h w) c"
        )
        context = einops.rearrange(
            context, "b ncam h w c -> b (ncam h w) c"
        )
        num_cropped_tokens = self._cropped_num_scene_tokens[sampling_level]

        if num_cropped_tokens > 0:
            # Local fine RGB features
            b = anchor.shape[0]
            anchor = anchor[..., :3].reshape(b, 1, 3).half()
            dist = torch.norm(anchor - context.half(), dim=-1)
            indices = dist.topk(
                k=num_cropped_tokens, dim=-1, largest=False).indices

            c = context_feats.shape[-1]
            context_feats = torch.gather(
                context_feats, 1, indices.unsqueeze(-1).expand(-1, -1, c))
            c = context.shape[-1]
            context = torch.gather(
                context, 1, indices.unsqueeze(-1).expand(-1, -1, c))
        
        return context_feats, context

    def _policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            sampling_level
        ) = fixed_inputs

        # Legacy operation, needs to remove
        fps_feats = context_feats.transpose(0, 1)
        fps_pos = self.encoder.relative_pe_layer(context)

        return self.prediction_heads[sampling_level](
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pos=fps_pos
        )

    def _conditional_sample(self, init_trajectory, fixed_inputs):
        sampling_level = fixed_inputs[-1]

        # Setting the inference steps for the noise scehduler
        noise_scheduler = self.noise_schedulers[sampling_level]
        n_steps = self.n_steps[sampling_level]
        noise_scheduler.set_timesteps(n_steps)

        # Noisy condition data
        trajectory = init_trajectory

        # Iterative denoising
        timesteps = noise_scheduler.timesteps
        for t in timesteps:
            out = self._policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs,
            )
            out = out[-1]  # keep only last layer's output
            trajectory = noise_scheduler.step(
                out[..., :9], t, trajectory[..., :9]
            ).prev_sample

        trajectory = torch.cat((trajectory, out[..., 9:]), -1)

        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )
        B, nhist, D = curr_gripper.shape


        # Start sampling hierarchically
        # Initial trajectory sampled from pure noise
        trajectory = torch.randn(
            size=(B, trajectory_mask.size(1), D),
            dtype=torch.float,
            device=rgb_obs.device
        )
        for level in range(len(self._sampling_levels)):
            selected_fixed_inputs = self._prepare_fixed_inputs(
                fixed_inputs, trajectory, level
            )

            # Sample
            trajectory = self._conditional_sample(
                trajectory[..., :9], selected_fixed_inputs
            )

        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()

        return trajectory

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def convert_rot(self, signal):
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == '6D':
            # The following code expects wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
            rot = quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if len(rot.shape) == 4:
                B, L, D1, D2 = rot.shape
                rot = rot.reshape(B * L, D1, D2)
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
                rot_6d = rot_6d.reshape(B, L, 6)
            else:
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
            signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        if self._rotation_parametrization == '6D':
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            if len(signal.shape) == 3:
                B, L, _ = signal.shape
                rot = signal[..., 3:9].reshape(B * L, 6)
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
                quat = quat.reshape(B, L, 4)
            else:
                rot = signal[..., 3:9]
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            # The above code handled wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        return signal

    def convert2rel(self, pcd, curr_gripper):
        """Convert coordinate system relaative to current gripper."""
        center = curr_gripper[:, -1, :3]  # (batch_size, 3)
        bs = center.shape[0]
        pcd = pcd - center.view(bs, 1, 3, 1, 1)
        curr_gripper = curr_gripper.clone()
        curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
        return pcd, curr_gripper

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

        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        # Prepare inputs
        total_loss = 0

        # Rectified flow map two distributions.  For the first round, we set
        # the second distribution to be a random noise.
        init_trajectory = torch.randn(
            gt_trajectory.shape, device=gt_trajectory.device
        )
        for level in range(len(self._sampling_levels)):
            # Select fixed inputs for the current level
            selected_fixed_inputs = self._prepare_fixed_inputs(
                fixed_inputs, gt_trajectory, level
            )

            # Sample a random timestep
            noise_scheduler = self.noise_schedulers[level]
            timesteps = noise_scheduler.sample_noise_step(
                num_noise=len(init_trajectory), device=init_trajectory.device
            )

            # Add noise to the clean trajectories
            noisy_trajectory = noise_scheduler.add_noise(
                gt_trajectory[..., :9], init_trajectory[..., :9],
                timesteps
            )

            # Predict the noise residual
            pred = self._policy_forward_pass(
                noisy_trajectory, timesteps, selected_fixed_inputs
            )

            # Compute loss
            for layer_pred in pred:
                target = init_trajectory - gt_trajectory
                loss = (
                    10 * F.l1_loss(layer_pred[..., :9], target[..., :9], reduction='mean')
                )
                if torch.numel(gt_openess) > 0:
                    openess = layer_pred[..., 9:]
                    loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
                total_loss += loss

            # Set the second distribution of the next sampling level to the
            # predicted original sample
            init_trajectory = noise_scheduler.batch_original_step(
                pred[-1][..., :9], timesteps, noisy_trajectory[..., :9]
            ).detach()

        return total_loss
