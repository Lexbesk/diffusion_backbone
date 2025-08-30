from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np

from modeling.policy.grasp_trajectory_denoiser import DexterousActor
from argparse import Namespace

import torch

def _unproject_masked_single(depth0: torch.Tensor, K: torch.Tensor, mask: torch.Tensor, N: int) -> torch.Tensor:
    """
    depth0: (H,W) or (H,W,1)
    K:      (3,3)
    mask:   (H,W) or (H,W,1)
    returns (N,3) in CAMERA frame
    """
    D = depth0.squeeze(-1).float()        # (H,W)
    M = (mask.squeeze(-1) > 0)            # (H,W) bool

    ys, xs = torch.where(M)
    if ys.numel() == 0:
        return torch.zeros((N, 3), dtype=torch.float32, device=depth0.device)

    z = D[ys, xs]
    valid = torch.isfinite(z) & (z > 0)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if z.numel() == 0:
        return torch.zeros((N, 3), dtype=torch.float32, device=depth0.device)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (xs.float() - cx) / fx * z
    Y = (ys.float() - cy) / fy * z
    P = torch.stack([X, Y, z], dim=-1)    # (M,3) CAM

    # fixed-size sampling/padding for batching
    M = P.shape[0]
    if M >= N:
        idx = torch.randperm(M, device=P.device)[:N]
        P = P[idx]
    else:
        reps = (N + M - 1) // M
        P = P.repeat(reps, 1)[:N]
    return P

def _unproject_masked(depth0: torch.Tensor, K: torch.Tensor, mask: torch.Tensor, N: int) -> torch.Tensor:
    """
    Light batched wrapper around _unproject_masked_single.

    Accepts:
      depth0: (H,W[,1]) or (B,H,W[,1]) or (B,T,H,W,1)
      K:      (3,3) or (B,3,3)  [for (B,T,...) we repeat K across T]
      mask:   (H,W[,1]) or (B,H,W[,1]) or (B,T,H,W[,1])

    Returns:
      (N,3) for single frame
      (B,N,3) for (B, ...)
      (B,T,N,3) for (B,T, ...)
    """
    # Single image path (original behavior)
    if depth0.dim() in (2, 3) and (K.dim() == 2):
        return _unproject_masked_single(depth0, K, mask, N)

    # Handle (B,H,W[,1])
    if depth0.dim() == 4 and depth0.shape[-1] in (1,):
        B, H, W, _ = depth0.shape
        assert mask.dim() in (3,4), "mask must be (B,H,W) or (B,H,W,1)"
        if mask.dim() == 4: mask = mask.squeeze(-1)
        # K can be (3,3) or (B,3,3)
        if K.dim() == 2:
            K = K.unsqueeze(0).expand(B, -1, -1)
        pcs = [
            _unproject_masked_single(depth0[i], K[i], mask[i], N)
            for i in range(B)
        ]
        return torch.stack(pcs, dim=0)    # (B,N,3)

    # Handle (B,T,H,W,1) (history)
    if depth0.dim() == 5 and depth0.shape[-1] == 1:
        B, T, H, W, _ = depth0.shape
        # mask could be (B,H,W) -> broadcast across T, or (B,T,H,W[,1])
        if mask.dim() == 3:
            mask_bt = mask[:, None].expand(B, T, H, W)  # (B,T,H,W)
        elif mask.dim() == 4 and mask.shape[-1] == 1:
            mask_bt = mask.squeeze(-1)                  # (B,T,H,W)
        elif mask.dim() == 4:
            mask_bt = mask                              # (B,T,H,W)
        elif mask.dim() == 5 and mask.shape[-1] == 1:
            mask_bt = mask.squeeze(-1)                  # (B,T,H,W)
        else:
            raise ValueError("Unsupported mask shape for (B,T,...) input.")

        # K can be (3,3) or (B,3,3). Repeat across T accordingly.
        if K.dim() == 2:
            K_bt = K.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1)   # (B,T,3,3)
        elif K.dim() == 3 and K.shape[0] == B:
            K_bt = K.unsqueeze(1).expand(B, T, -1, -1)                # (B,T,3,3)
        else:
            raise ValueError("K must be (3,3) or (B,3,3) for (B,T,...) input.")

        depth_flat = depth0.reshape(B*T, H, W, 1)
        mask_flat  = mask_bt.reshape(B*T, H, W)
        K_flat     = K_bt.reshape(B*T, 3, 3)

        pcs = [
            _unproject_masked_single(depth_flat[i], K_flat[i], mask_flat[i], N)
            for i in range(B*T)
        ]
        return torch.stack(pcs, dim=0).reshape(B, T, N, 3)

    raise ValueError(f"Unsupported input shapes: depth0={tuple(depth0.shape)}, K={tuple(K.shape)}, mask={tuple(mask.shape)}")

DIFFUSION_CONFIG = {'train_data_dir': '/data/user_data/austinz/Robots/manipulation/zarr_datasets/dexterousact/train.zarr', 
        'eval_data_dir': ('/data/user_data/austinz/Robots/manipulation/zarr_datasets/dexterousact/train.zarr'), 
        'dataset': 'DexterousAct', 
        'num_workers': 4, 'batch_size': 16, 
        'batch_size_val': 8, 'chunk_size': 1, 'memory_limit': 8, 'base_log_dir': ('train_logs'), 
        'exp_log_dir': ('DexterousAct_debug'), 'run_log_dir': ('run_Aug24-B16-lv2bs4-Bval8-DT10-nhist8-nfuture32-K4'), 
        'checkpoint': '/data/user_data/austinz/Robots/manipulation/diffusion_backbone/train_logs/DexterousAct_debug/run_Aug24-B16-lv2bs4-Bval8-DT10-nhist4-nfuture4-K4/last.pth', 
        'val_freq': 1000, 'eval_only': False, 'eval_overfit': False, 'lr': 1e-06, 'lr_scheduler': 'constant', 'wd': 0.005, 
        'train_iters': 600000, 'use_compile': False, 'use_ema': False, 'lv2_batch_size': 4, 'model_type': 'dexterousactor', 
        'bimanual': False, 'keypose_only': True, 'pre_tokenize': True, 'custom_img_size': None, 'backbone': 'clip', 
        'output_level': 'res3', 'upsample': False, 'finetune_backbone': False, 'finetune_text_encoder': False, 
        'fps_subsampling_factor': 5, 'embedding_dim': 256, 'num_attn_heads': 8, 'num_vis_instr_attn_layers': 3, 
        'num_history': 1, 'num_shared_attn_layers': 30, 'workspace_normalizer_buffer': 0.04, 'relative_action': False, 
        'rotation_format': 'quat_wxyz', 'denoise_timesteps': 10, 'denoise_model': 'rectified_flow', 'visualize_denoising_steps': False, 
        'accurate_joint_pos': True, 'save_for_mujoco': False, 'test_mujoco': True, 'vis_freq': 10000000000, 'val_set_all_anchor': True, 
        'condition_on_grasp_type_id': True, 'guidance_weight': 1.5, 'sample_times': 1, 'nhist': 4, 'nfuture': 4, 'K': 4, 
        'log_dir': ('analogical_manipulation/train_logs/DexterousAct_debug/run_Aug24-B16-lv2bs4-Bval8-DT10-nhist8-nfuture32-K4'), 'local_rank': 0}

DIFFUSION_CONFIG = Namespace(**DIFFUSION_CONFIG)

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class DiffusionPlayer(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.model = DexterousActor(
            embedding_dim=DIFFUSION_CONFIG.embedding_dim,
            num_attn_heads=DIFFUSION_CONFIG.num_attn_heads,
            nhist=DIFFUSION_CONFIG.num_history,
            nhand=2 if DIFFUSION_CONFIG.bimanual else 1,
            num_shared_attn_layers=DIFFUSION_CONFIG.num_shared_attn_layers,
            relative=DIFFUSION_CONFIG.relative_action,
            rotation_format=DIFFUSION_CONFIG.rotation_format,
            denoise_timesteps=DIFFUSION_CONFIG.denoise_timesteps,
            denoise_model=DIFFUSION_CONFIG.denoise_model,
            lv2_batch_size=DIFFUSION_CONFIG.lv2_batch_size,
            visualize_denoising_steps=DIFFUSION_CONFIG.visualize_denoising_steps,
            accurate_joint_pos=DIFFUSION_CONFIG.accurate_joint_pos,
            guidance_weight=DIFFUSION_CONFIG.guidance_weight,
        )
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        # self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
            # - q_hist: (B, nhist, 31)  history of joint angles  
            # - v_hist: (B, nhist, 31)  history of joint velocities                                                                       
            # - ee_fingers: (B, nhist, 6, 3)  history of wrist+5fingers positions                  (robot base frame)                     
            # - act_hist: (B, nhist, 31)  history of full joint angles, executed or teacher's                                                
            # - depth_hist: (B, nhist, H, W, 1)  history of depth image observations                                                  
            # - goal_pos: (B, 3)  target object position (camera frame)
            # - grasp_cond: (B, 7 + 22)  target grasp condition (camera frame)
            # - obj_init_pcl_cam  (B, 1024, 3)
            # - intrinsics: (B, 3, 3)  camera intrinsics
            # - extrinsics: (B, 4, 4)  camera extrinsics (world→camera)

        self.input_dict = {}
        self.input_dict_init = False


    def get_action(self, obs, is_deterministic = False):
        print(obs.shape, 'obs sum before preproc')
        print(obs.sum(), 'obs after preproc')
        # return obs[:, 0:31]
        if not self.input_dict_init:
            IMAGE_HEIGHT = 120
            IMAGE_WIDTH = 160
            q_hist = obs[:, :31].unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1)
            v_hist = obs[:, 31:62].unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1)
            ee_fingers = obs[:, 62:80].unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1).reshape(-1, DIFFUSION_CONFIG.nhist, 6, 3)
            act_hist = obs[:, :31].unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1)
            depth_hist = obs[:, 121 : 121 + IMAGE_HEIGHT * IMAGE_WIDTH].unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1).reshape(-1, DIFFUSION_CONFIG.nhist, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
            goal_pos = obs[:, 80:83] # camera frame
            grasp_cond = obs[:, 83:112] # camera frame 3 + 4 (wxyz) + 22 fingers
            intrinsics = obs[:, 112:121].reshape(-1, 3, 3)
            print(intrinsics, 'intrinsics')
            obj_seg = obs[:, 121 + IMAGE_HEIGHT * IMAGE_WIDTH : 121 + IMAGE_HEIGHT * IMAGE_WIDTH * 2].reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH)   
            obj_init_pcl_cam = _unproject_masked(depth_hist[:, -1], intrinsics, obj_seg, N=1024)

            self.input_dict = {
                'q_hist': q_hist,
                'v_hist': v_hist,
                'ee_fingers': ee_fingers,
                'act_hist': act_hist,
                'depth_hist': depth_hist,
                'goal_pos': goal_pos,
                'grasp_cond': grasp_cond,
                'obj_init_pcl_cam': obj_init_pcl_cam,
                'intrinsics': intrinsics,
            }
            self.input_dict_init = True
            print(self.input_dict['obj_init_pcl_cam'].shape, 'pcl shape')
            print(self.input_dict['depth_hist'].shape, 'depth hist shape')
            print(self.input_dict['q_hist'].shape, 'q hist shape')
            print(self.input_dict['grasp_cond'].shape, 'grasp cond shape')
            print(self.input_dict['intrinsics'].shape, 'intrinsics shape')
        
        else:
            # update history
            self.input_dict['q_hist'] = torch.cat([self.input_dict['q_hist'][:, 1:], obs[:, :31].unsqueeze(1)], dim=1)
            self.input_dict['v_hist'] = torch.cat([self.input_dict['v_hist'][:, 1:], obs[:, 31:62].unsqueeze(1)], dim=1)
            self.input_dict['ee_fingers'] = torch.cat([self.input_dict['ee_fingers'][:, 1:], obs[:, 62:80].unsqueeze(1).reshape(-1, 1, 6, 3)], dim=1)
            self.input_dict['act_hist'] = torch.cat([self.input_dict['act_hist'][:, 1:], obs[:, :31].unsqueeze(1)], dim=1)
            self.input_dict['depth_hist'] = torch.cat([self.input_dict['depth_hist'][:, 1:], obs[:, 121 : 121 + IMAGE_HEIGHT * IMAGE_WIDTH].unsqueeze(1).reshape(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)], dim=1)
            self.input_dict['goal_pos'] = obs[:, 80:83]
            self.input_dict['grasp_cond'] = obs[:, 83:112]\

        action_future, q_future, obj_pose_future = self.model(self.input_dict, run_inference=False)
        print(action_future.shape, 'action future shape')
        print(q_future.shape, 'q future shape')
        print(obj_pose_future.shape, 'obj future shape')
        current_action = action_future[:, 0, :]
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        # checkpoint = torch_ext.load_checkpoint(fn)
        model_dict = torch.load(
            DIFFUSION_CONFIG.checkpoint,
            map_location="cpu",
            weights_only=True
        )
        # self.model.load_state_dict(checkpoint['model'])
        msn, unxpct = self.model.load_state_dict(model_dict["weight"], strict=False)
        if msn:
            print(f"Missing keys (not found in checkpoint): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("All keys matched successfully!")
        # if self.normalize_input and 'running_mean_std' in checkpoint:
        #     self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = model_dict.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)
        self.is_rnn = False

    def reset(self):
        # self.init_rnn()
        pass