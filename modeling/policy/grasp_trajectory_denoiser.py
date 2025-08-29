import torch
from torch import nn
from torch.nn import functional as F

from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb
from ..utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)
from ..utils.position_encodings import RotaryPositionEncoding3D
from ..utils.depth_encoder import DepthLightCNN, GoalGraspToken
from ..utils.dexterousact_token_encoders import ActionTokenEncoder, ObjectPoseTokenEncoder, HistoryStateTokenEncoder, TokenPredictor, zeros_xyz, build_slices
from ..utils.fk_layer import FKLayer
import time
from collections import defaultdict

from utils.forward_kinematics.pk_utils import build_chain_from_mjcf_path, get_urdf_limits
from pytorch3d.ops import sample_farthest_points


class DexterousActor(nn.Module):
    def __init__(self,
                 # Encoder and decoder arguments
                 embedding_dim=128,
                 num_attn_heads=8,
                 nhist=3,
                 nhand=1,
                 nfuture=32,
                 pcd_feat_dim=256,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_wxyz',
                 # Denoising arguments
                 denoise_timesteps=1000,
                 denoise_model="rectified_flow",
                 # Training arguments
                 lv2_batch_size=1,
                 # hand arguments
                 urdf_path='/data/user_data/austinz/Robots/manipulation/analogical_manipulation/assets/robots/franka_shadow.urdf',
                 visualize_denoising_steps=False,
                 accurate_joint_pos=False,
                 guidance_weight=None,
                 ):
        super().__init__()
        # Arguments to be accessed by the main class
        self._rotation_format = rotation_format
        self._relative = relative
        self._lv2_batch_size = lv2_batch_size
        dtype = torch.float32
        self.dtype = dtype
        self.timers = defaultdict(float)
        self._profiling = False
        self.visualize_denoising_steps = visualize_denoising_steps
        self.visualization_data = {} # "partial_points": [B, N, 3], "grasps": [B, M, 29]}
        self.accurate_joint_pos = accurate_joint_pos
        self.nfuture = nfuture
        
        self.guidance_weight = guidance_weight
        
        self.urdf_path = urdf_path
        joint_names, jmin, jmax = get_urdf_limits(urdf_path)
        self.jmin = torch.from_numpy(jmin).to('cuda').to(dtype)  # (22,)
        self.jmax = torch.from_numpy(jmax).to('cuda').to(dtype)  # (22,)
        self.joint_names = joint_names
        print(joint_names, self.jmin, self.jmax, 'jmin, jmax from urdf')

        self.act_enc = ActionTokenEncoder(dof=len(self.jmin), d=embedding_dim, include_err=False, include_delta=False)
        self.obj_enc = ObjectPoseTokenEncoder(d=embedding_dim, include_delta=False, center_first=False)
        self.state_enc = HistoryStateTokenEncoder(dof=len(self.jmin), d=embedding_dim)
        self.depth_enc = DepthLightCNN(d=embedding_dim, add_validity_channel=True, robust_norm=True, dropout=0.1)
        self.goal_proj = nn.Linear(3, embedding_dim)
        self.goal_grasp_tokener = GoalGraspToken(d_token=embedding_dim, n_heads=4)
        
        # for FK layer
        self.out_links = ["lftip", "rftip", "mftip", "fftip", "thtip", "wrist"]
        probe_points_local = {
        }

        self.fk = FKLayer(
            urdf_path=urdf_path,
            joint_names=self.joint_names,
            out_links=self.out_links,
            probe_points_local=probe_points_local,
            compile_torch=True,            # if torch>=2.0
        ).to('cuda')

        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist,
            nfuture=nfuture,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format == 'euler' else 6,
            angle_dim=22,
        )

        # Noise/denoise schedulers and hyperparameters
        self.position_scheduler, self.rotation_scheduler, self.angle_scheduler = fetch_schedulers(
            denoise_model, denoise_timesteps
        )
        self.n_steps = denoise_timesteps

        # Normalization for the 3D space, will be loaded in the main process
        if rotation_format == 'euler':  # normalize pos+rot
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0, 0, 0, 0, 0], [1., 1, 1, 1, 1, 1]]),
                requires_grad=False
            )
        else:
            distance = 1.0
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[-distance, -distance, -distance], [distance, distance, distance]]),
                requires_grad=False
            )
                        
    def fk_layer(self, joint_angles):
        (B, nhist, J) = joint_angles.shape
        pos_hist, _, _ = self.fk(joint_angles, need_rot=False)  # dict of link->(B, nhist, 3)

        # Example: stack 6 points (wrist + 5 fingertips) to match your ee_fingers layout
        ee_hist = torch.stack([pos_hist["wrist"],
                            pos_hist["lftip"],
                            pos_hist["fftip"],
                            pos_hist["mftip"],
                            pos_hist["rftip"],
                            pos_hist["thtip"]
                            ], dim=2)  # (B, nhist, 6, 3)
        # # If you want orientations (for contact frames, etc.)
        # pos_future, rot_future, _ = self.fk(q_future, need_rot=True)
        # R_thumb = rot_future["rh_thtip"]    # (B, nfuture, 3, 3)
        # p_thumb = pos_future["rh_thtip"]
        return ee_hist[:, :, 0]

    def policy_forward_pass(self, noisy_actions, timestep, fixed_inputs, train, condition=True):
        _t0 = self._start_timer()
        
        act_hist, obj_pose_hist, q_hist, v_hist, ee_fingers, depth_hist, obj_init_pcl_cam, goal_pos, grasp_cond = fixed_inputs
        
        # crop depth to 0 to 10m
        depth_hist = torch.clamp(depth_hist, 0, 10)        
        
        noisy_act_future = noisy_actions[:, :, :len(self.jmin)]          # (B, nfuture, 31)
        noisy_q_future = noisy_actions[:, :, len(self.jmin):2*len(self.jmin)]  # (B, nfuture, 31)
        noisy_obj_pose_future = noisy_actions[:, :, 2*len(self.jmin):]          # (B, nfuture, 7)
        
        action_hist_tokens = self.act_enc(act_hist, q_hist=None)
        action_future_tokens = self.act_enc(noisy_act_future, q_hist=None)
        q_future_tokens = self.act_enc(noisy_q_future, q_hist=None)
        # obj_hist_tokens = self.obj_enc(obj_pose_hist) # obj history pose is encoded in history states
        obj_future_tokens = self.obj_enc(noisy_obj_pose_future)
        state_hist_tokens = self.state_enc(
            q_hist=q_hist,                    # (B, nhist, 31)  already normalized
            v_hist=v_hist,                    # (B, nhist, 31)  (ideally z-scored)
            ee_fingers=ee_fingers,            # (B, nhist, 6, 3) in robot frame
        )
        depth_hist_tokens = self.depth_enc(depth_hist)
        
        # condition signals
        goal_tok = self.goal_proj(goal_pos)
        goal_tok = goal_tok.unsqueeze(1)  # (B, 1, d)
        grasp_tok = self.goal_grasp_tokener(obj_init_pcl_cam, grasp_cond)  # (B,1,D)
        
        # action_xyz = self.fk_layer(noisy_act_future)  # (B, nfuture, 6, 3)
        # print(action_xyz.shape, 'action xyz shape')

        B = action_hist_tokens.shape[0]
        device = action_hist_tokens.device
        D = action_hist_tokens.shape[-1]
        
        to_check = {
            "action_future": action_future_tokens,
            "q_future": q_future_tokens,
            "obj_future": obj_future_tokens,
            "action_hist": action_hist_tokens,
            "state_hist": state_hist_tokens,
            "depth_hist": depth_hist_tokens,
            "goal_tok": goal_tok,
            "grasp_tok": grasp_tok,
        }
        for name, t in to_check.items():
            assert t is not None and t.shape[-1] == D, f"{name} has dim {t.shape[-1]} != {D}"
        
        q_list = [
            ("act_future", action_future_tokens),   # (B, nfuture_act, D)
            ("q_future", q_future_tokens),         # (B, nfuture_q, D)
            ("obj_future", obj_future_tokens),      # (B, nfuture_obj, D)
        ]
        q_tokens = torch.cat([t for _, t in q_list], dim=1).contiguous()       # (B, Nq, D)
        q_xyz = zeros_xyz(q_tokens)                                              # (B, Nq, 3)
        q_slices = build_slices(q_list)

        # Context keys for CROSS-ATTN = observations + conditions (no future tokens)
        k_ctx_list = [
            ("act_hist", action_hist_tokens),       # (B, nhist_act, D)
            ("state_hist", state_hist_tokens),      # (B, nhist_state, D)
            ("depth_hist", depth_hist_tokens),      # (B, nhist_depth, D)
            ("goal", goal_tok),                     # (B, 1, D)
            ("grasp", grasp_tok),                   # (B, 1, D)
        ]
        k_tokens = torch.cat([t for _, t in k_ctx_list], dim=1).contiguous()   # (B, Nk, D)
        k_xyz = zeros_xyz(k_tokens)                                             # (B, Nk, 3)
        k_slices = build_slices(k_ctx_list)

        # Keys for SELF-ATTN = context + the denoising tokens themselves
        k_self_list = k_ctx_list + q_list
        k_self_tokens = torch.cat([t for _, t in k_self_list], dim=1).contiguous()  # (B, Nk+Nq, D)
        k_self_xyz = zeros_xyz(k_self_tokens)                                        # (B, Nk+Nq, 3)
        k_self_slices = build_slices(k_self_list)
        
        token_groups = {
            "cross_attn": {
                "q": q_tokens,         # (B, Nq, D)   future-only
                "k": k_tokens,         # (B, Nk, D)   obs + conditions
                "q_xyz": q_xyz,        # (B, Nq, 3)   zeros
                "k_xyz": k_xyz,        # (B, Nk, 3)   zeros
                "q_slices": q_slices,  # dict of name -> (start, end)
                "k_slices": k_slices,
            },
            "self_attn": {
                "q": q_tokens,            # (B, Nq, D)   same queries
                "k": k_self_tokens,       # (B, Nk+Nq, D) context + queries
                "q_xyz": q_xyz.clone(),   # (B, Nq, 3)
                "k_xyz": k_self_xyz,      # (B, Nk+Nq, 3)
                "q_slices": q_slices,
                "k_slices": k_self_slices,
            },
        }
    
        out = self.prediction_head(
            token_groups,
            timestep,
            train=train,
            condition=condition
        )
        
        self._stop_timer(_t0, "transformer_head")
        return out

    def conditional_sample(self, denoise_content, device, fixed_inputs):
        # Set schedulers
        self.position_scheduler.set_timesteps(self.n_steps, device=device)

        # Iterative denoising
        timesteps = self.position_scheduler.timesteps
        # print(timesteps, 'timesteps')
        for t_ind, t in enumerate(timesteps):
            t_batch = t * torch.ones(len(denoise_content), device=device, dtype=torch.long)
            if self.guidance_weight is not None:  # e.g., 1.5
                pred_uncond = self.policy_forward_pass(denoise_content, t_batch, fixed_inputs, train=False, condition=False)[-1]
                pred_cond   = self.policy_forward_pass(denoise_content, t_batch, fixed_inputs, train=False, condition=True)[-1]
                pred = pred_uncond + self.guidance_weight * (pred_cond - pred_uncond)
            else:
                pred = self.policy_forward_pass(denoise_content, t_batch, fixed_inputs, train=False, condition=True)[-1]
            denoise_content = self.position_scheduler.step(pred, t_ind, denoise_content).prev_sample

        return denoise_content

    def compute_grasp(self, batch):
        q_hist = batch['q_hist']                    # (B, nhist, 31)
        v_hist = batch['v_hist']                    # (B, nhist, 31)
        ee_fingers = batch['ee_fingers']            # (B, nhist, 6, 3)
        obj_pose_hist = batch['obj_pose_hist']      # (B, nhist, 7)
        act_hist = batch['act_hist']                # (B, nhist, 31)
        depth_hist = batch['depth_hist']            # (B, nhist, H, W, 1)
        goal_pos = batch['goal_pos']                # (B, 3)
        grasp_cond = batch['grasp_cond']            # (B, 29)
        object_scale = batch['object_scale']  # (B, 1)
        object_asset = batch['object_asset']  # text path
        obj_init_pcl_cam = batch['obj_init_pcl_cam']  # (B, 1024, 3)
        
        q_hist = self.normalize_actions(q_hist)
        act_hist = self.normalize_actions(act_hist)
        obj_pose_hist = self.normalize_pos(obj_pose_hist)
        obj_pose_hist = self.convert_rot(obj_pose_hist)
        goal_pos = self.normalize_pos(goal_pos)
        grasp_cond = self.normalize_pos(grasp_cond)
        grasp_cond = self.convert_rot(grasp_cond)
        grasp_cond = self.normalize_finger_angles(grasp_cond)
        obj_init_pcl_cam = self.normalize_pos(obj_init_pcl_cam)
        
        fixed_inputs = act_hist, obj_pose_hist, q_hist, v_hist, ee_fingers, depth_hist, obj_init_pcl_cam, goal_pos, grasp_cond

        # Sample from learned model starting from noise
        out_dim = 9
        denoise_content = torch.randn(
            size=(q_hist.shape[0], self.nfuture, 2*len(self.jmin) + out_dim),
            device=q_hist.device
        )
        denoise_content = self.conditional_sample(
            denoise_content,
            device=denoise_content.device,
            fixed_inputs=fixed_inputs
        )
        
        norm_action_future = denoise_content[:, :, :len(self.jmin)]          # (B, nfuture, 31)
        norm_q_future = denoise_content[:, :, len(self.jmin):2*len(self.jmin)]  # (B, nfuture, 31)
        norm_obj_pose_future = denoise_content[:, :, 2*len(self.jmin):]          # (B, nfuture, 9)
        
        action_future = self.unnormalize_actions(norm_action_future)
        q_future = self.unnormalize_actions(norm_q_future)
        obj_pose_future = self.unconvert_rot(norm_obj_pose_future)
        obj_pose_future = self.unnormalize_pos(obj_pose_future)
        
        return action_future, q_future, obj_pose_future

    def compute_loss(self, batch):
        # Encode observations, states, instructions
        """ batch
            sample keys: dict_keys(['q_hist', 'v_hist', 'ee_fingers', 'obj_pose_hist', 'act_hist', 'goal_pos', 'grasp_cond', 'intrinsics', 'extrinsics', 'obj_pose_future', 'act_future', 'depth_hist'])
            torch.Size([64, 8, 370, 640, 1]) depth hist shape
            torch.Size([64, 8, 31]) q hist shape
            torch.Size([64, 8, 31]) v hist shape
            torch.Size([64, 8, 6, 3]) ee fingers shape
            torch.Size([64, 8, 7]) obj pose hist shape
            torch.Size([64, 8, 31]) act hist shape
            torch.Size([64, 3]) goal pos shape
            torch.Size([64, 29]) grasp cond shape
            torch.Size([64, 3, 3]) intrinsics shape
            torch.Size([64, 4, 4]) extrinsics shape
            torch.Size([64, 32, 7]) obj pose future shape
            torch.Size([64, 32, 31]) act future shape
        """
        
        q_hist = batch['q_hist']                    # (B, nhist, 31)
        v_hist = batch['v_hist']                    # (B, nhist, 31)
        ee_fingers = batch['ee_fingers']            # (B, nhist, 6, 3)
        obj_pose_hist = batch['obj_pose_hist']      # (B, nhist, 7)
        act_hist = batch['act_hist']                # (B, nhist, 31)
        depth_hist = batch['depth_hist']            # (B, nhist, H, W, 1)
        goal_pos = batch['goal_pos']                # (B, 3)
        grasp_cond = batch['grasp_cond']            # (B, 29)
        intrinsics = batch['intrinsics']            # (B, 3, 3)
        extrinsics = batch['extrinsics']            # (B, 4, 4)
        obj_pose_future = batch['obj_pose_future']  # (B, nfuture, 7)
        act_future = batch['act_future']            # (B, nfuture, 31)
        q_future = batch['q_future']            # (B, nfuture, 31), optional
        object_scale = batch['object_scale']  # (B, 1)
        object_asset = batch['object_asset']  # text path
        obj_init_pcl_cam = batch['obj_init_pcl_cam']  # (B, 1024, 3)
        
        q_hist = self.normalize_actions(q_hist)
        act_hist = self.normalize_actions(act_hist)
        act_future = self.normalize_actions(act_future)
        q_future = self.normalize_actions(q_future)
        obj_pose_hist = self.normalize_pos(obj_pose_hist)
        obj_pose_hist = self.convert_rot(obj_pose_hist)
        obj_pose_future = self.normalize_pos(obj_pose_future)
        obj_pose_future = self.convert_rot(obj_pose_future)
        goal_pos = self.normalize_pos(goal_pos)
        grasp_cond = self.normalize_pos(grasp_cond)
        grasp_cond = self.convert_rot(grasp_cond)
        grasp_cond = self.normalize_finger_angles(grasp_cond)
        obj_init_pcl_cam = self.normalize_pos(obj_init_pcl_cam)
        denoise_content = torch.cat([act_future, q_future, obj_pose_future], dim=-1)  # (B, nfuture, 31+31+7)
        
        fixed_inputs = act_hist, obj_pose_hist, q_hist, v_hist, ee_fingers, depth_hist, obj_init_pcl_cam, goal_pos, grasp_cond
        
        # Loop lv2_batch_size times and sample different noises with same input
        # Trick to effectively increase the batch size without re-encoding
        total_loss = 0
        for _ in range(self._lv2_batch_size):
            # Sample noise
            noise = torch.randn(denoise_content.shape, device=denoise_content.device)
            # Sample a random timestep
            # breakpoint()
            timesteps = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )

            noisy_actions = self.position_scheduler.add_noise(
                denoise_content, noise,
                timesteps
            )

            # Predict the noise residual
            pred = self.policy_forward_pass(
                noisy_actions,
                timesteps, fixed_inputs, train=True
            )

            # Compute loss
            for layer_pred in pred:
                denoise_target = self.position_scheduler.prepare_target(
                    noise, denoise_content 
                ) # default gt_grasp itself (or noise)
                loss = F.mse_loss(layer_pred, denoise_target)
                
                total_loss = total_loss + loss
        return total_loss / self._lv2_batch_size

    def normalize_actions(self, q):
        normalized = (q - self.jmin) / (self.jmax - self.jmin) * 2.0 - 1.0
        normalized = torch.clamp(normalized, min=-1.0, max=1.0)
        return normalized
    
    def unnormalize_actions(self, q_norm):
        unnormalized = (q_norm + 1.0) / 2.0 * (self.jmax - self.jmin) + self.jmin
        return unnormalized

    def normalize_pos(self, signal):
        n = min(self.workspace_normalizer.size(-1), signal.size(-1))
        _min = self.workspace_normalizer[0][:n].float()
        _max = self.workspace_normalizer[1][:n].float()
        return torch.cat((
            (signal[..., :n] - _min) / (_max - _min) * 2.0 - 1.0,
            signal[..., n:]
        ), -1)

    def unnormalize_pos(self, signal):
        n = min(self.workspace_normalizer.size(-1), signal.size(-1))
        _min = self.workspace_normalizer[0][:n].float()
        _max = self.workspace_normalizer[1][:n].float()
        return torch.cat((
            (signal[..., :n] + 1.0) / 2.0 * (_max - _min) + _min,
            signal[..., n:]
        ), -1)
        
    def normalize_angles(self, signal):
        # Normalize angles to [-1, 1]
        assert signal.size(-1) > 9
        if signal.size(-1) > 9:
            angles = signal[..., 9:]
            angles = (angles - self.jmin) / (self.jmax - self.jmin) * 2.0 - 1.0
            angles = torch.clamp(angles, min=-1.0, max=1.0) 
            return torch.cat((signal[..., :9], angles), -1)
        return signal

    def unnormalize_angles(self, signal):
        # Unnormalize angles from [-1, 1] to [jmin, jmax]
        if signal.size(-1) == 29:
            angles = signal[..., 7:]
            angles = (angles + 1.0) / 2.0 * (self.jmax - self.jmin) + self.jmin
            return torch.cat((signal[..., :7], angles), -1)
        return signal
    
    def normalize_finger_angles(self, signal):
        # Normalize angles to [-1, 1]
        assert signal.size(-1) > 9
        if signal.size(-1) > 9:
            angles = signal[..., 9:]
            angles = (angles - self.jmin[-22:]) / (self.jmax[-22:] - self.jmin[-22:]) * 2.0 - 1.0
            angles = torch.clamp(angles, min=-1.0, max=1.0) 
            return torch.cat((signal[..., :9], angles), -1)
        return signal
    
    def unnormalize_finger_angles(self, signal):
        # Unnormalize angles from [-1, 1] to [jmin, jmax]
        if signal.size(-1) == 29:
            angles = signal[..., 9:]
            angles = (angles + 1.0) / 2.0 * (self.jmax[-22:] - self.jmin[-22:]) + self.jmin[-22:]
            return torch.cat((signal[..., :9], angles), -1)
        return signal

    def convert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        rot = normalise_quat(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        # The following code expects wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        # Convert to rotation matrix
        rot = quaternion_to_matrix(rot)
        # Convert to 6D
        if len(rot.shape) == 4:
            B, L, D1, D2 = rot.shape
            rot = rot.reshape(B * L, D1, D2)
            rot = get_ortho6d_from_rotation_matrix(rot)
            rot = rot.reshape(B, L, 6)
        else:
            rot = get_ortho6d_from_rotation_matrix(rot)
        # Concatenate pos, rot, other state info
        signal = torch.cat([signal[..., :3], rot], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
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
        # The above code handled wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (1, 2, 3, 0)]
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal
    
    def _pose_l1(self, pred, target):
        pos_l  = 30 * F.l1_loss(pred[...,:3],  target[...,:3])
        rot_l  = 10 * F.l1_loss(pred[...,3:9], target[...,3:9])
        ang_l  = 10 * F.l1_loss(pred[...,9:],  target[...,9:])
        return pos_l + rot_l + ang_l
    
    def _start_timer(self):
        """Return (event/time) object marking the start of a region."""
        if not self._profiling:
            return None
        if torch.cuda.is_available():
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()
    
    def _stop_timer(self, start, key):
        """Accumulate elapsed time into self.timers[key]."""
        if (not self._profiling) or (start is None):
            return
        if torch.cuda.is_available():
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            self.timers[key] += start.elapsed_time(end)      # ms
        else:
            self.timers[key] += (time.perf_counter() - start) * 1e3  # ms

    def forward(
        self,
        batch,
        run_inference=False,
        ):
        """
        Arguments:
            Batch Keys: (The current observation for the policy batch)
            - q_hist: (B, nhist, 31)  history of joint angles  
            - v_hist: (B, nhist, 31)  history of joint velocities                                                                       
            - ee_fingers: (B, nhist, 6, 3)  history of wrist+5fingers positions                  (robot base frame)                     
            - obj_pose_hist: (B, nhist, 7)  history of object pose                               (camera frame, xyz + quaternion wxyz) 
            - act_hist: (B, nhist, 31)  history of full joint angles, executed or teacher's                                                
            - depth_hist: (B, nhist, H, W, 1)  history of depth image observations                                                  
            - goal_pos: (B, 3)  target object position (camera frame)
            - grasp_cond: (B, 7 + 22)  target grasp condition (camera frame)
            - obj_init_pcl_cam  (B, 1024, 3)
            - intrinsics: (B, 3, 3)  camera intrinsics
            - extrinsics: (B, 4, 4)  camera extrinsics (world→camera)
            
            Target Batch Keys: (The targets for future nfuture steps, only for training)
            - obj_pose_future: (B, nfuture, 7)  future object pose
            - act_future: (B, nfuture, 31)  future full joint angles, teacher's policy

        Returns:
            - loss: scalar, if run_inference is False
            - action: (B, nfuture, 31), at inference
        """
        if run_inference:
            return self.compute_grasp(
                batch
            )

        return self.compute_loss(
            batch
        )


class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=128,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 num_shared_attn_layers_head=20,
                 nhist=4,
                 nfuture=4,
                 rotary_pe=True,
                 rot_dim=6,
                 angle_dim=22,
                 type_embed_mode="conditions_only"  # {"conditions_only", "all", "none"}
                 ):
        super().__init__()
        
        embedding_dim_fw = embedding_dim * 4  # Feed-forward dimension
        self.nhist = nhist
        self.nfuture = nfuture

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # For classifier-free guidance:
        self.class_dropout_prob = 0.1  # p to drop conditioning during training
        self.null_goal = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.null_grasp = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        
        self.type_embed_mode = type_embed_mode
        self.token_type_ids = {
            "act_hist": 0, "state_hist": 1, "depth_hist": 2,
            "goal": 3, "grasp": 4,
            "act_future": 5, "obj_future": 7, "q_future": 6,
        }
        self.num_token_types = max(self.token_type_ids.values()) + 1
        self.type_emb = nn.Embedding(self.num_token_types, embedding_dim)

        # Estimate attends to context (no subsampling)
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim_fw,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=False
        )

        # Shared attention layers
        self.self_attn = nn.ModuleList([
            AttentionModule(
                num_layers=1,
                d_model=embedding_dim,
                dim_fw=embedding_dim_fw,
                dropout=0.1,
                n_heads=num_attn_heads,
                pre_norm=False,
                rotary_pe=rotary_pe,
                use_adaln=True,
                is_self=False
            )
            for _ in range(num_shared_attn_layers)
        ])

        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        
        self.to_actions   = TokenPredictor(embedding_dim, out_dim=31, num_blocks=2, dropout=0.1)
        self.to_q         = TokenPredictor(embedding_dim, out_dim=31, num_blocks=2, dropout=0.1)
        self.to_obj_pose  = TokenPredictor(embedding_dim, out_dim=9,  num_blocks=2, dropout=0.1)


    def forward(self, token_groups, timesteps, train=True, condition=True):

        time_embs = self.encode_denoising_timestep(timesteps)
        rel_grasp_pos, rel_pcd_pos, rel_pos = self.get_positional_embeddings(token_groups["cross_attn"]["q_xyz"], token_groups["cross_attn"]["k_xyz"])
        
        q = token_groups["cross_attn"]["q"].clone()
        k_cross = token_groups["cross_attn"]["k"].clone()
        k_self = token_groups["self_attn"]["k"].clone()
        
        self._add_type_embeddings_(q, token_groups["cross_attn"]["q_slices"],
                                   list(token_groups["cross_attn"]["q_slices"].keys()))
        # Cross-attn keys: obs + conditions
        self._add_type_embeddings_(k_cross, token_groups["cross_attn"]["k_slices"],
                                   list(token_groups["cross_attn"]["k_slices"].keys()))
        # Self-attn keys: obs + conditions + future
        self._add_type_embeddings_(k_self, token_groups["self_attn"]["k_slices"],
                                   list(token_groups["self_attn"]["k_slices"].keys()))
        
        if not train:
            if not condition:
                if "goal" in token_groups["cross_attn"]["k_slices"]:
                    s, e = token_groups["cross_attn"]["k_slices"]["goal"]
                    k_cross[:, s:e, :] = self.null_goal
                if "grasp" in token_groups["cross_attn"]["k_slices"]:
                    s, e = token_groups["cross_attn"]["k_slices"]["grasp"]
                    k_cross[:, s:e, :] = self.null_grasp
                if "goal" in token_groups["self_attn"]["k_slices"]:
                    s, e = token_groups["self_attn"]["k_slices"]["goal"]
                    k_self[:, s:e, :] = self.null_goal
                if "grasp" in token_groups["self_attn"]["k_slices"]:
                    s, e = token_groups["self_attn"]["k_slices"]["grasp"]
                    k_self[:, s:e, :] = self.null_grasp
        else:
            # During training, randomly drop conditions for classifier-free guidance
            if self.class_dropout_prob > 0:
                if "goal" in token_groups["cross_attn"]["k_slices"]:
                    if torch.rand(1).item() < self.class_dropout_prob:
                        s, e = token_groups["cross_attn"]["k_slices"]["goal"]
                        k_cross[:, s:e, :] = self.null_goal
                if "grasp" in token_groups["cross_attn"]["k_slices"]:
                    if torch.rand(1).item() < self.class_dropout_prob:
                        s, e = token_groups["cross_attn"]["k_slices"]["grasp"]
                        k_cross[:, s:e, :] = self.null_grasp
                if "goal" in token_groups["self_attn"]["k_slices"]:
                    if torch.rand(1).item() < self.class_dropout_prob:
                        s, e = token_groups["self_attn"]["k_slices"]["goal"]
                        k_self[:, s:e, :] = self.null_goal
                if "grasp" in token_groups["self_attn"]["k_slices"]:
                    if torch.rand(1).item() < self.class_dropout_prob:
                        s, e = token_groups["self_attn"]["k_slices"]["grasp"]
                        k_self[:, s:e, :] = self.null_grasp
            

        x = self.cross_attn(
            seq1=q,
            seq2=k_cross,
            seq1_pos=rel_grasp_pos,
            seq2_pos=rel_pcd_pos,
            ada_sgnl=time_embs
        )[-1]
        
        for layer in self.self_attn:
            x = layer(
                seq1=x,
                seq2=k_self,
                seq1_pos=rel_grasp_pos,
                seq2_pos=torch.cat((rel_pcd_pos, rel_grasp_pos), 1),
                ada_sgnl=time_embs
            )[-1]
        
        # print(x.shape, 'transformer output x shape')
        act_future_pred     = self.to_actions(x[:, :self.nfuture, :])   
        q_future_pred = self.to_q(x[:, self.nfuture:2*self.nfuture, :])
        obj_pose_future_pred= self.to_obj_pose(x[:, 2*self.nfuture:, :])

        return [
            torch.cat([act_future_pred, q_future_pred, obj_pose_future_pred], dim=-1)
        ]
        
    def _add_type_embeddings_(self, toks, slices, which_keys):
        """
        In-place add type embeddings to specified token segments.
        toks: (B, N, D)
        slices: dict name -> (start, end)
        which_keys: iterable of token names to mark
        """
        if self.type_embed_mode == "none":
            return toks
        if self.type_embed_mode == "conditions_only":
            # keep only goal/grasp even if more provided
            which_keys = [k for k in which_keys if k in ("goal", "grasp")]
        # else "all": use all in which_keys

        if not which_keys:
            return toks

        B, _, D = toks.shape
        for name in which_keys:
            if name not in slices:
                continue
            s, e = slices[name]
            type_id = self.token_type_ids[name]
            # (1, 1, D) -> (1, e-s, D) -> (B, e-s, D)
            toks[:, s:e, :] += self.type_emb.weight[type_id].view(1, 1, D)
        return toks

    def encode_denoising_timestep(self, timestep):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        return time_feats

    def get_positional_embeddings(
        self,
        q_xyzs, k_xyzs # in cross attention
    ):
        rel_grasp_pos = self.relative_pe_layer(q_xyzs)
        rel_pcd_pos = self.relative_pe_layer(k_xyzs)
        # rel_fps_pos = self.relative_pe_layer(fps_scene_pos)
        rel_pos = torch.cat([rel_grasp_pos, rel_pcd_pos], 1)
        return rel_grasp_pos, rel_pcd_pos, rel_pos



from collections import OrderedDict
if __name__ == "__main__":
    # Test the GraspDenoiser class
    def trainable_params(m: torch.nn.Module) -> int:
        """Number of parameters that will be updated by the optimiser."""
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    model = GraspDenoiser()
    modules = OrderedDict(
        TransformerHead     = model.prediction_head,  # the large decoder stack
        ActionEncoder      = model.act_enc,
        ObjectEncoder      = model.obj_enc,
        StateEncoder       = model.state_enc,
        DepthEncoder       = model.depth_enc,
        GoalProjection     = model.goal_proj,
        GraspTokenizer     = model.goal_grasp_tokener,
    )
    totals = {name: trainable_params(mod) for name, mod in modules.items()}
    totals["TOTAL (model)"] = trainable_params(model)
    
    width   = max(len(k) for k in totals) + 2
    header  = f'{"Module":{width}} #Params    (M)'
    print('=' * (width + 20))
    print(header)
    print('=' * (width + 20))
    for k, v in totals.items():
        print(f'{k:{width}} {v:>12,}  ({v/1e6:5.2f} M)')
    print('=' * (width + 20))
    # print(model)
    # Add more tests as needed