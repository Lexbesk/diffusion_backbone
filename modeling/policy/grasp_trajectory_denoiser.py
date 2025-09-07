import torch
from torch import nn
from torch.nn import functional as F

from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule, build_xattn_mask_modality_temporal, TorchCrossAttnBlock, plot_attn_mask
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
from ..utils.dexterousact_token_encoders import ActionTokenEncoder, ObjectPoseTokenEncoder, HistoryStateTokenEncoder, TokenPredictor, zeros_xyz, build_slices, TokenPredictorPlus, TimeProj, sinusoidal_time_embedding
from ..utils.fk_layer import FKLayer
import time
from collections import defaultdict
import numpy as np

from diffusion_backbone.utils.forward_kinematics.pk_utils import build_chain_from_mjcf_path, get_urdf_limits
from pytorch3d.ops import sample_farthest_points
import time

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
        self.nhist = nhist
        
        self.guidance_weight = guidance_weight
        
        self.urdf_path = urdf_path
        # joint_names, jmin, jmax = get_urdf_limits(urdf_path)
        joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'WRJ2', 'WRJ1', 'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1']
        jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -0.5235988, -0.7853982, -0.43633232, 0.0, 0.0, 0.0, 0.0, -0.43633232, 0.0, 0.0, 0.0, -0.43633232, 0.0, 0.0, 0.0, -0.43633232, 0.0, 0.0, 0.0, -1.047, 0.0, -0.2618, -0.5237, 0.0])
        jmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.6981317, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 1.047, 1.309, 0.2618, 0.5237, 1.571])
        self.jmin = torch.from_numpy(jmin).to('cuda').to(dtype)  # (22,)
        self.jmax = torch.from_numpy(jmax).to('cuda').to(dtype)  # (22,)
        self.joint_names = joint_names
        print(joint_names, self.jmin, self.jmax, 'jmin, jmax from urdf')

        self.act_enc = ActionTokenEncoder(dof=len(self.jmin), d=embedding_dim, include_err=False, include_delta=False)
        self.q_enc = ActionTokenEncoder(dof=len(self.jmin), d=embedding_dim, include_err=False, include_delta=False)
        self.obj_enc = ObjectPoseTokenEncoder(d=embedding_dim, include_delta=False, center_first=False)
        self.state_enc = HistoryStateTokenEncoder(dof=len(self.jmin), d=embedding_dim)
        self.depth_enc = DepthLightCNN(d=embedding_dim, add_validity_channel=True, robust_norm=True, dropout=0.1)
        self.goal_proj = nn.Linear(3, embedding_dim)
        self.goal_grasp_tokener = GoalGraspToken(d_token=embedding_dim, n_heads=4)

        self.time_proj = TimeProj(d_model=embedding_dim)
        self.type_vocab = {
            "act_hist": 0, "state_hist": 1, "depth_hist": 2,
            "goal": 3, "grasp": 4, "act_future": 5, "q_future": 6, "obj_future": 7
        }
        self.type_emb = nn.Embedding(len(self.type_vocab), embedding_dim)
        self.emb_ln = nn.LayerNorm(embedding_dim)

        
        # for FK layer
        self.out_links = ["lftip", "rftip", "mftip", "fftip", "thtip", "wrist"]
        probe_points_local = {
        }

        self.debug_target_param = torch.nn.Parameter(torch.zeros(1, 4, 31+31+9))  # dummy param for .to(device) call

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

    def policy_forward_pass(self, noisy_actions, timestep, fixed_inputs, train, condition=True):
        _t0 = self._start_timer()
        
        act_hist, obj_pose_hist, q_hist, v_hist, ee_fingers, depth_hist, obj_init_pcl_cam, goal_pos, grasp_cond = fixed_inputs
        depth_hist = torch.clamp(depth_hist, 0, 10)
        noisy_act_future = noisy_actions[:, :, :len(self.jmin)]          # (B, nfuture, 31)
        noisy_q_future = noisy_actions[:, :, len(self.jmin):2*len(self.jmin)]  # (B, nfuture, 31)
        noisy_obj_pose_future = noisy_actions[:, :, 2*len(self.jmin):]          # (B, nfuture, 7)
        
        # Encoders
        action_hist_tokens = self.act_enc(act_hist, q_hist=None)
        action_future_tokens = self.act_enc(noisy_act_future, q_hist=None)
        q_future_tokens = self.q_enc(noisy_q_future, q_hist=None)
        obj_future_tokens = self.obj_enc(noisy_obj_pose_future)
        state_hist_tokens = self.state_enc(q_hist=q_hist, v_hist=v_hist, ee_fingers=ee_fingers)
        depth_hist_tokens = self.depth_enc(depth_hist)
        # condition signals
        goal_tok = self.goal_proj(goal_pos).unsqueeze(1)
        grasp_tok = self.goal_grasp_tokener(obj_init_pcl_cam, grasp_cond)  # (B,1,D)

        device = noisy_actions.device
        B, D  = action_hist_tokens.shape[0], action_hist_tokens.shape[-1]

        # Future-history timestep embeddings
        t_hist_ids   = torch.arange(-self.nhist, 0,  device=device, dtype=torch.float32)
        t_future_ids = torch.arange(1, self.nfuture+1, device=device, dtype=torch.float32)
        temb_hist   = self.time_proj(t_hist_ids)[None, :, :].expand(B, self.nhist, -1)
        temb_future = self.time_proj(t_future_ids)[None, :, :].expand(B, self.nfuture, -1)
        action_hist_tokens   = action_hist_tokens   + temb_hist
        state_hist_tokens    = state_hist_tokens    + temb_hist
        depth_hist_tokens    = depth_hist_tokens    + temb_hist
        action_future_tokens = action_future_tokens + temb_future
        q_future_tokens      = q_future_tokens      + temb_future
        obj_future_tokens    = obj_future_tokens    + temb_future

        # Add type embeddings
        # print(self.type_emb.weight.shape, 'shape of type emb')
        # print(self.type_emb.weight, 'weight of type emb')
        action_hist_tokens   = action_hist_tokens   + self.type_emb.weight[self.type_vocab["act_hist"]][None, None, :]
        state_hist_tokens    = state_hist_tokens    + self.type_emb.weight[self.type_vocab["state_hist"]][None, None, :]
        depth_hist_tokens    = depth_hist_tokens    + self.type_emb.weight[self.type_vocab["depth_hist"]][None, None, :]
        goal_tok             = goal_tok             + self.type_emb.weight[self.type_vocab["goal"]][None, None, :]
        grasp_tok            = grasp_tok            + self.type_emb.weight[self.type_vocab["grasp"]][None, None, :]
        action_future_tokens = action_future_tokens + self.type_emb.weight[self.type_vocab["act_future"]][None, None, :]
        q_future_tokens      = q_future_tokens      + self.type_emb.weight[self.type_vocab["q_future"]][None, None, :]
        obj_future_tokens    = obj_future_tokens    + self.type_emb.weight[self.type_vocab["obj_future"]][None, None, :]

        q_list = [
            ("act_future", action_future_tokens),
            ("q_future", q_future_tokens),
            ("obj_future", obj_future_tokens),
        ]
        q_tokens = torch.cat([t for _, t in q_list], dim=1)

        k_ctx_list = [
            ("act_hist", action_hist_tokens),       
            ("state_hist", state_hist_tokens),     
            ("depth_hist", depth_hist_tokens), 
            ("goal", goal_tok),                  
            ("grasp", grasp_tok),                
        ]
        k_self_list = k_ctx_list + q_list
        k_self_tokens = torch.cat([t for _, t in k_self_list], dim=1)

        # Absolute sinusoidal positional embeddings
        S_Q  = q_tokens.size(1)
        S_KS = k_self_tokens.size(1)

        q_pos  = sinusoidal_time_embedding(torch.arange(S_Q,  device=device), D).unsqueeze(0).expand(B, -1, -1)
        ks_pos = sinusoidal_time_embedding(torch.arange(S_KS, device=device), D).unsqueeze(0).expand(B, -1, -1)

        q_tokens      = self.emb_ln(q_tokens      + q_pos)    # optional LN to stabilize
        k_self_tokens = self.emb_ln(k_self_tokens + ks_pos)

        nh_act   = action_hist_tokens.size(1)
        nh_state = state_hist_tokens.size(1)
        nh_depth = depth_hist_tokens.size(1)
        n_goal   = goal_tok.size(1)
        n_grasp  = grasp_tok.size(1)
        nf_act   = action_future_tokens.size(1)
        nf_state = q_future_tokens.size(1)
        nf_obj   = obj_future_tokens.size(1)

        attn_mask, q_off, kv_off = build_xattn_mask_modality_temporal(
            nh_act, nh_state, nh_depth, n_goal, n_grasp,
            nf_act, nf_state, nf_obj,
            device=device,
            allow_state_to_see_objects=False,
            constrain_obj=True
        )

        # plot_attn_mask(attn_mask, q_off, kv_off,
        #        q_labels=("Af_q","Sf_q","Of_q"),
        #        kv_labels=("Ah","Sh","Dh","G","GR","Af","Sf","Of"),
        #        savepath='attn_mask.png')

        out = self.prediction_head(
            q_tokens, # query
            k_self_tokens, # k and v for self-attn
            attn_mask,
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
                # pred_uncond = self.policy_forward_pass(denoise_content, t_batch, fixed_inputs, train=False, condition=False)[-1]
                # pred_cond   = self.policy_forward_pass(denoise_content, t_batch, fixed_inputs, train=False, condition=True)[-1]
                # pred = pred_uncond + self.guidance_weight * (pred_cond - pred_uncond)
                pred = self.policy_forward_pass(denoise_content, t_batch, fixed_inputs, train=False, condition=True)[-1]
            else:
                pred = self.policy_forward_pass(denoise_content, t_batch, fixed_inputs, train=False, condition=True)[-1]
            denoise_content = self.position_scheduler.step(pred, t_ind, denoise_content).prev_sample

        return denoise_content

    def compute_grasp(self, batch):
        q_hist = batch['q_hist']                    # (B, nhist, 31)
        v_hist = batch['v_hist']                    # (B, nhist, 31)
        ee_fingers = batch['ee_fingers']            # (B, nhist, 6, 3)
        # obj_pose_hist = batch['obj_pose_hist']      # (B, nhist, 7)
        act_hist = batch['act_hist']                # (B, nhist, 31)
        depth_hist = batch['depth_hist']            # (B, nhist, H, W, 1)
        goal_pos = batch['goal_pos']                # (B, 3)
        grasp_cond = batch['grasp_cond']            # (B, 29)
        # object_scale = batch['object_scale']  # (B, 1)
        # object_asset = batch['object_asset']  # text path
        obj_init_pcl_cam = batch['obj_init_pcl_cam']  # (B, 1024, 3)
        
        q_hist = self.normalize_actions(q_hist)
        act_hist = self.normalize_actions(act_hist)
        # obj_pose_hist = self.normalize_pos(obj_pose_hist)
        # obj_pose_hist = self.convert_rot(obj_pose_hist)
        goal_pos = self.normalize_pos(goal_pos)
        grasp_cond = self.normalize_pos(grasp_cond)
        grasp_cond = self.convert_rot(grasp_cond)
        grasp_cond = self.normalize_finger_angles(grasp_cond)
        obj_init_pcl_cam = self.normalize_pos(obj_init_pcl_cam)
        
        fixed_inputs = act_hist, None, q_hist, v_hist, ee_fingers, depth_hist, obj_init_pcl_cam, goal_pos, grasp_cond

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

        # print(q_hist[:, 0], 'q hist')
        # print(act_hist[:, 0], 'act hist')
        
        q_hist = self.normalize_actions(q_hist)
        act_hist = self.normalize_actions(act_hist)
        assert torch.all(q_hist.abs() <= 1.02) and torch.all(act_hist.abs() <= 1.02), "History actions not normalized properly"
        # print(act_future, 'act future before norm')
        act_future = self.normalize_actions(act_future)
        q_future = self.normalize_actions(q_future)
        # print(act_future[:, 0], 'act future')
        # print(q_future, 'q future')
        # print(act_future.abs().max(), 'max')
        # print(act_future.abs().argmax(), 'argmax') 
        # print(q_future.abs().max(), 'max')
        # print(q_future.abs().argmax(), 'argmax')

        assert torch.all(act_future.abs() <= 1.02) and torch.all(q_future.abs() <= 1.02), f"Future actions not normalized properly"
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
        # print(denoise_content.shape, 'denoise content shape')
        
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

            # print(timesteps, 'timesteps for loss')
            # print(noise, 'noise')

            noisy_actions = self.position_scheduler.add_noise(
                denoise_content, noise,
                timesteps
            )

            # Predict the noise residual
            pred = self.policy_forward_pass(
                noisy_actions,
                timesteps, fixed_inputs, train=True
            )
            # print(noisy_actions[0, 0], 'denoise content')

            # Compute loss
            for layer_pred in pred:
                denoise_target = self.position_scheduler.prepare_target(
                    noise, denoise_content 
                ) # default gt_grasp itself (or noise)
                loss = 0

                # layer_pred = self.debug_target_param
                # print(denoise_target[0, :2], 'denoise target')
                # print(layer_pred[0, :2], 'layer pred')
                # print(layer_pred[:, :1, :len(self.jmin)], 'layer pred ')
                # print(denoise_content[:, :1, :len(self.jmin)], 'denoise content ')
                # loss_action = F.mse_loss(layer_pred[:, :, :len(self.jmin)], denoise_target[:, :, :len(self.jmin)])
                loss_action_arm = F.mse_loss(layer_pred[:, :, :9], denoise_target[:, :, :9])
                loss_action_finger = F.mse_loss(layer_pred[:, :, 9:len(self.jmin)], denoise_target[:, :, 9:len(self.jmin)])
                loss_q = F.mse_loss(layer_pred[:, :, len(self.jmin):2*len(self.jmin)], denoise_target[:, :, len(self.jmin):2*len(self.jmin)])
                loss_obj = F.mse_loss(layer_pred[:, :, 2*len(self.jmin):], denoise_target[:, :, 2*len(self.jmin):])
                # loss = loss + 100 * loss_action + 100 * loss_q + loss_obj
                loss = loss + loss_action_finger + loss_q
                loss = loss + loss_action_arm
                loss = loss + loss_obj
                
                total_loss = total_loss + loss
        return total_loss / self._lv2_batch_size

    def normalize_actions(self, q):
        # normalized = (q - self.jmin) / (self.jmax - self.jmin) * 2.0 - 1.0
        normalized = (q - self.jmin) / (self.jmax - self.jmin) * 1.0 - 0.5
        # normalized = torch.clamp(normalized, min=-1.0, max=1.0)
        return normalized
    
    def unnormalize_actions(self, q_norm):
        # unnormalized = (q_norm + 1.0) / 2.0 * (self.jmax - self.jmin) + self.jmin
        unnormalized = (q_norm + 0.5) / 1.0 * (self.jmax - self.jmin) + self.jmin
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # drain previous kernels
        t0 = time.perf_counter()

        if run_inference:
            out = self.compute_grasp(
                batch
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()  # wait for this call to finish
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            print(f"[step] model() took {dt:.2f} ms")

            return out

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
                 rotary_pe=False,
                 rot_dim=6,
                 angle_dim=22,
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
        
        # # Estimate attends to context (no subsampling)
        # self.cross_attn = AttentionModule(
        #     num_layers=2,
        #     d_model=embedding_dim,
        #     dim_fw=embedding_dim_fw,
        #     dropout=0.1,
        #     n_heads=num_attn_heads,
        #     pre_norm=False,
        #     rotary_pe=rotary_pe,
        #     use_adaln=True,
        #     is_self=False
        # )

        # # Shared attention layers
        # self.self_attn = nn.ModuleList([
        #     AttentionModule(
        #         num_layers=1,
        #         d_model=embedding_dim,
        #         dim_fw=embedding_dim_fw,
        #         dropout=0.1,
        #         n_heads=num_attn_heads,
        #         pre_norm=False,
        #         rotary_pe=rotary_pe,
        #         use_adaln=True,
        #         is_self=False
        #     )
        #     for _ in range(num_shared_attn_layers)
        # ])

        # self.cross_blocks = nn.ModuleList([
        #     TorchCrossAttnBlock(d_model=embedding_dim, n_heads=num_attn_heads, dropout=0.1)
        #     for _ in range(num_shared_attn_layers)
        # ])

        d_cond  = self.encode_denoising_timestep(torch.ones(1)).shape[-1]

        self.cross_blocks = nn.ModuleList([
            TorchCrossAttnBlock(d_model=embedding_dim, n_heads=num_attn_heads, dropout=0.1, d_cond=d_cond)
            for _ in range(num_shared_attn_layers)
        ])


        # self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        
        # self.to_actions   = TokenPredictor(embedding_dim, out_dim=31, num_blocks=2, dropout=0.1)
        # self.to_q         = TokenPredictor(embedding_dim, out_dim=31, num_blocks=2, dropout=0.1)
        # self.to_obj_pose  = TokenPredictor(embedding_dim, out_dim=9,  num_blocks=2, dropout=0.1)

        self.to_actions = TokenPredictorPlus(d_model=embedding_dim, out_dim=31, num_blocks=6, expansion=6.0, drop_path=0.1)
        self.to_q = TokenPredictorPlus(d_model=embedding_dim, out_dim=31, num_blocks=6, expansion=6.0, drop_path=0.1)
        self.to_obj_pose = TokenPredictorPlus(d_model=embedding_dim, out_dim=9, num_blocks=6, expansion=6.0, drop_path=0.1)



    def forward(self, q_tokens, k_self_tokens, attn_mask, timesteps, train=True, condition=True):
        time_embs = self.encode_denoising_timestep(timesteps)        

        x = q_tokens

        for cross_layer in self.cross_blocks:
            x = cross_layer(x, k_self_tokens, attn_mask=attn_mask, key_padding_mask=None, cond=time_embs)

        
        # if not train:
        #     if not condition:
        #         if "goal" in token_groups["cross_attn"]["k_slices"]:
        #             s, e = token_groups["cross_attn"]["k_slices"]["goal"]
        #             k_cross[:, s:e, :] = self.null_goal
        #         if "grasp" in token_groups["cross_attn"]["k_slices"]:
        #             s, e = token_groups["cross_attn"]["k_slices"]["grasp"]
        #             k_cross[:, s:e, :] = self.null_grasp
        #         if "goal" in token_groups["self_attn"]["k_slices"]:
        #             s, e = token_groups["self_attn"]["k_slices"]["goal"]
        #             k_self[:, s:e, :] = self.null_goal
        #         if "grasp" in token_groups["self_attn"]["k_slices"]:
        #             s, e = token_groups["self_attn"]["k_slices"]["grasp"]
        #             k_self[:, s:e, :] = self.null_grasp
        # else:
        #     # During training, randomly drop conditions for classifier-free guidance
        #     if self.class_dropout_prob > 0:
        #         if "goal" in token_groups["cross_attn"]["k_slices"]:
        #             if torch.rand(1).item() < self.class_dropout_prob:
        #                 s, e = token_groups["cross_attn"]["k_slices"]["goal"]
        #                 k_cross[:, s:e, :] = self.null_goal
        #         if "grasp" in token_groups["cross_attn"]["k_slices"]:
        #             if torch.rand(1).item() < self.class_dropout_prob:
        #                 s, e = token_groups["cross_attn"]["k_slices"]["grasp"]
        #                 k_cross[:, s:e, :] = self.null_grasp
        #         if "goal" in token_groups["self_attn"]["k_slices"]:
        #             if torch.rand(1).item() < self.class_dropout_prob:
        #                 s, e = token_groups["self_attn"]["k_slices"]["goal"]
        #                 k_self[:, s:e, :] = self.null_goal
        #         if "grasp" in token_groups["self_attn"]["k_slices"]:
        #             if torch.rand(1).item() < self.class_dropout_prob:
        #                 s, e = token_groups["self_attn"]["k_slices"]["grasp"]
        #                 k_self[:, s:e, :] = self.null_grasp
        
        # x = self.cross_attn(
        #     seq1=x,
        #     seq2=k_cross,
        #     seq1_pos=rel_grasp_pos,
        #     seq2_pos=rel_pcd_pos,
        #     ada_sgnl=time_embs
        # )[-1]

        # x = self.cross_attn(
        #     seq1=x,
        #     seq2=k_cross,
        #     # seq1_pos=rel_grasp_pos,
        #     # seq2_pos=rel_pcd_pos,
        #     ada_sgnl=time_embs
        # )[-1]
        
        # for layer in self.self_attn:
        #     x = layer(
        #         seq1=x,
        #         seq2=k_self,
        #         # seq1_pos=rel_grasp_pos,
        #         # seq2_pos=torch.cat((rel_pcd_pos, rel_grasp_pos), 1),
        #         ada_sgnl=time_embs
        #     )[-1]

        
        # print(x.shape, 'transformer output x shape')
        act_future_pred     = self.to_actions(x[:, :self.nfuture, :])   
        q_future_pred = self.to_q(x[:, self.nfuture:2*self.nfuture, :])
        obj_pose_future_pred= self.to_obj_pose(x[:, 2*self.nfuture:, :])

        

        return [
            torch.cat([act_future_pred, q_future_pred, obj_pose_future_pred], dim=-1)
        ]

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