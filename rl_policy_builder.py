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

import time
import torch

import matplotlib
matplotlib.use("Agg")           # safe on headless machines
import matplotlib.pyplot as plt
import os

from datasets.utils import to_tensor, read_zarr_with_cache, T_to_pose7_wxyz, pose7_wxyz_to_T, T_inv, transform_points, pose7_xyzw_to_wxyz

# Option A: load a DDP/DP checkpoint into an unwrapped model (strict match)
import torch
from pathlib import Path
from typing import Dict, Tuple

def _extract_state_dict(obj: Dict) -> Dict:
    """
    Heuristically extract the actual state_dict from a checkpoint object.
    Prefers 'state_dict' > 'model' > raw dict of tensors.
    """
    # Common containers produced by trainers
    for k in ("state_dict", "model"):
        if k in obj and isinstance(obj[k], dict):
            return obj[k]
    # If top-level already looks like a state dict (string->Tensor mapping), use it
    if isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise ValueError("No state_dict-like mapping found in checkpoint.")

def _strip_prefixes(sd: Dict, prefixes=("module.", "model.", "ema_model.")) -> Dict:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def load_ckpt_strict(model: torch.nn.Module,
                     ckpt_path: str,
                     map_location: str = "cpu",
                     extra_prefixes: Tuple[str, ...] = ()) -> None:
    """
    Load weights into `model` ensuring 0 missing and 0 unexpected keys.
    Strips common DDP/EMA prefixes so you can use the model unwrapped.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    sd_raw = _extract_state_dict(ckpt["weight"])

    # First pass: strip common prefixes (+ any caller-specified)
    sd = _strip_prefixes(sd_raw, ("module.", "model.", "ema_model.", *extra_prefixes))

    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(sd.keys())
    missing    = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    if missing or unexpected:
        # Helpful diagnostics
        def take(xs, n=8): return xs[:n] + (["..."] if len(xs) > n else [])
        msg = [
            f"[load_ckpt_strict] Key mismatch after stripping prefixes.",
            f"  Missing ({len(missing)}): {take(missing)}",
            f"  Unexpected ({len(unexpected)}): {take(unexpected)}",
        ]
        raise RuntimeError("\n".join(msg))

    # All good — do a strict load
    msg = model.load_state_dict(sd, strict=True)
    # PyTorch returns an object with .missing_keys / .unexpected_keys; assert again for safety
    assert len(getattr(msg, "missing_keys", [])) == 0
    assert len(getattr(msg, "unexpected_keys", [])) == 0
    print(f"[load_ckpt_strict] Loaded OK from {Path(ckpt_path).name} ({len(sd)} tensors).")

# ---- Usage ---------------------------------------------------------
# model = build_your_model()
# load_ckpt_strict(model, "path/to/ckpt.pth", map_location="cpu")
# model.to("cuda:0").eval()


def save_depth_mask_images(D: torch.Tensor, M: torch.Tensor, out_prefix="dbg"):
    """
    D: (H,W) depth in meters (float), zeros/NaNs mean invalid
    M: (H,W) bool mask
    Saves: <prefix>_depth.png, <prefix>_mask.png, <prefix>_overlay.png
    """
    D = D.detach().cpu().float().numpy()
    M = M.detach().cpu().numpy().astype(bool)

    # Choose display range from valid depths
    valid = np.isfinite(D) & (D > 0)
    vmin, vmax = 0.0, 2.0

    # --- Depth image ---
    plt.figure(figsize=(5,4))
    im = plt.imshow(D, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="depth")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_depth.png", dpi=150)
    plt.close()

    # --- Mask image (binary) ---
    plt.figure(figsize=(5,4))
    plt.imshow(M, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_mask.png", dpi=150)
    plt.close()

    # --- Overlay: depth colormap with mask in red ---
    # Make an RGB image from the normalized depth
    denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
    depth01 = np.clip((D - vmin) / denom, 0, 1)
    depth_rgb = plt.cm.viridis(depth01)[..., :3]  # (H,W,3) RGB

    overlay = depth_rgb.copy()
    overlay[M] = [1.0, 0.0, 0.0]   # draw mask in red
    plt.figure(figsize=(5,4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_overlay.png", dpi=150)
    plt.close()

def _unproject_masked_single(depth0: torch.Tensor, K: torch.Tensor, mask: torch.Tensor, N: int, idx=0) -> torch.Tensor:
    """
    depth0: (H,W) or (H,W,1)
    K:      (3,3)
    mask:   (H,W) or (H,W,1)
    returns (N,3) in CAMERA frame
    """
    D = depth0.squeeze(-1).float()        # (H,W)
    M = (mask.squeeze(-1) > 0)            # (H,W) bool
    print(D, 'depth in unproject')
    os.makedirs("inspection", exist_ok=True)
    save_depth_mask_images(D, M, out_prefix=f"inspection/{idx:04d}")

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

    print(mask.shape, 'mask shape in unproject')
    print(mask[0].sum(), 'mask sum in unproject')
    print(mask[1].sum(), 'mask 1 sum in unproject')

    # Handle (B,H,W[,1])
    B, H, W, _ = depth0.shape
    pcs = [
        _unproject_masked_single(depth0[i], K[i], mask[i], N, i)
        for i in range(B)
    ]
    print(pcs, 'pcs')
    return torch.stack(pcs, dim=0)    # (B,N,3)


DIFFUSION_CONFIG = {'train_data_dir': '/data/user_data/austinz/Robots/manipulation/zarr_datasets/dexterousact/train.zarr', 
        'eval_data_dir': ('/data/user_data/austinz/Robots/manipulation/zarr_datasets/dexterousact/train.zarr'), 
        'dataset': 'DexterousAct', 
        'num_workers': 4, 'batch_size': 16, 
        'batch_size_val': 8, 'chunk_size': 1, 'memory_limit': 8, 'base_log_dir': ('train_logs'), 
        'exp_log_dir': ('DexterousAct_debug'), 'run_log_dir': ('run_Aug24-B16-lv2bs4-Bval8-DT10-nhist8-nfuture32-K4'), 
        'checkpoint': '/data/user_data/austinz/Robots/manipulation/diffusion_backbone/train_logs/DexterousAct_debug/run_Sep6_1000-B32-lv2bs1-Bval64-DT10-nhist4-nfuture4-K2-numlayers10-embedding_dim256/last.pth', 
        'val_freq': 1000, 'eval_only': False, 'eval_overfit': False, 'lr': 1e-06, 'lr_scheduler': 'constant', 'wd': 0.005, 
        'train_iters': 600000, 'use_compile': False, 'use_ema': False, 'lv2_batch_size': 4, 'model_type': 'dexterousactor', 
        'bimanual': False, 'keypose_only': True, 'pre_tokenize': True, 'custom_img_size': None, 'backbone': 'clip', 
        'output_level': 'res3', 'upsample': False, 'finetune_backbone': False, 'finetune_text_encoder': False, 
        'fps_subsampling_factor': 5, 'embedding_dim': 256, 'num_attn_heads': 8, 'num_vis_instr_attn_layers': 3, 
        'num_history': 1, 'num_shared_attn_layers': 10, 'workspace_normalizer_buffer': 0.04, 'relative_action': False, 
        'rotation_format': 'quat_wxyz', 'denoise_timesteps': 10, 'denoise_model': 'rectified_flow', 'visualize_denoising_steps': False, 
        'accurate_joint_pos': True, 'save_for_mujoco': False, 'test_mujoco': True, 'vis_freq': 10000000000, 'val_set_all_anchor': True, 
        'condition_on_grasp_type_id': True, 'guidance_weight': 1.5, 'sample_times': 1, 'nhist': 4, 'nfuture': 4, 'K': 4, 
        'log_dir': ('analogical_manipulation/train_logs/DexterousAct_debug/run_Aug24-B16-lv2bs4-Bval8-DT10-nhist8-nfuture32-K4'), 'local_rank': 0}

DIFFUSION_CONFIG = Namespace(**DIFFUSION_CONFIG)
STEP = 0

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

    @torch.no_grad()
    def get_action(self, obs, is_deterministic = False):
        # print(obs.shape, 'obs sum before preproc')
        # print(obs.sum(), 'obs sum before preproc')
        # print(obs, 'obs before preproc')
        if obs.sum().abs() <= 1e-6 and not self.input_dict_init:
            return obs[:, 0:31]
        IMAGE_HEIGHT = 120
        IMAGE_WIDTH = 160
        # act_hist = obs[:, :31].unsqueeze(1)
        # q_hist = obs[:, :31].unsqueeze(1)
        # v_hist = obs[:, 31:62].unsqueeze(1)
        ee_fingers = obs[:, 62:80].reshape(-1, 6, 3) # world frame, need to be in robot base frame
        # depth_hist = -obs[:, 144 : 144 + IMAGE_HEIGHT * IMAGE_WIDTH].unsqueeze(1).reshape(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        goal_pos = obs[:, 80:83] # world frame
        grasp_cond = obs[:, 83:112] # world frame 3 + 4 (wxyz) + 22 fingers
        intrinsics = obs[:, 112:121].reshape(-1, 3, 3)
        extrinsics = obs[:, 121:137].reshape(-1, 4, 4)
        robot_pose = obs[:, 137:144]

        # coordinate system transforms
        T_wc = extrinsics.float()                      # WORLD->CAMERA
        pose_wxyz = pose7_xyzw_to_wxyz(robot_pose.float())  # (1,7) wxyz
        T_wb = pose7_wxyz_to_T(pose_wxyz).squeeze(0)  # BASE->WORLD
        T_bw = T_inv(T_wb)                                    # WORLD->BASE

        # print(T_wc, 'T_w to c')
        # print(T_wb, 'T_w to b')
        # print(T_bw, 'T_b to w (should be robot pose)')
        wrist_w  = grasp_cond[:, :7]                # [1,7] world
        wrist_c  = T_to_pose7_wxyz(T_wc @ pose7_wxyz_to_T(pose7_xyzw_to_wxyz(wrist_w))).squeeze(0)  # [7]
        hand_q   = grasp_cond[:, 7:]         
        goal_w = goal_pos          # (3,) WORLD
        # print(goal_w, 'goal w')
        goal_c = transform_points(T_wc, goal_w.unsqueeze(-2)).squeeze()  # (3,) CAMERA
        # print(goal_c, 'goal c')
        # exit(0)
        # print(ee_fingers.shape, 'ee fingers shape before tf')
        ee_b = transform_points(T_bw, ee_fingers)
        # print(ee_b.shape, 'ee b shape')

        # print(ee_b, 'ee b first') 
        # print(wrist_c, 'wrist c')
        # print(hand_q, 'hand q')

        if not self.input_dict_init:
            q_hist = obs[:, :31].unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1)
            v_hist = obs[:, 31:62].unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1)
            ee_fingers = ee_b.unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1, 1)
            # print(ee_fingers.shape, 'ee fingers shape')
            act_hist = obs[:, :31].unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1)
            depth_hist = -obs[:, 144 : 144 + IMAGE_HEIGHT * IMAGE_WIDTH].unsqueeze(1).repeat(1, DIFFUSION_CONFIG.nhist, 1).reshape(-1, DIFFUSION_CONFIG.nhist, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
            depth_hist = torch.clamp(depth_hist, 0.0, 10.0)
            goal_pos = goal_c # camera frame
            # print(wrist_c.shape, 'wrist c')
            # print(hand_q.shape, 'hand q')
            grasp_cond = torch.cat([wrist_c, hand_q], dim=1) # camera frame
            obj_seg = obs[:, 144 + IMAGE_HEIGHT * IMAGE_WIDTH : 144 + IMAGE_HEIGHT * IMAGE_WIDTH * 2].reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH)   
            obj_init_pcl_cam = _unproject_masked(depth_hist[:, -1], intrinsics, obj_seg, N=1024)

            self.input_dict = {
                'q_hist': q_hist,
                'v_hist': v_hist,
                'ee_fingers': ee_fingers, # make sure this is in the robot's base frame
                'act_hist': act_hist,
                'depth_hist': depth_hist,
                'goal_pos': goal_pos, # camera frame
                'grasp_cond': grasp_cond, # camera frame
                'obj_init_pcl_cam': obj_init_pcl_cam, # (B, 1024, 3) in camera frame
                'intrinsics': intrinsics,
            }
            self.input_dict_init = True
            # save pcl to file to ply file to visualize
            import open3d as o3d
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(obj_init_pcl_cam[1].cpu().numpy())
            o3d.io.write_point_cloud("pcl.ply", pcl)
            # print(self.input_dict['obj_init_pcl_cam'][0], 'pcl shape')
            # print(self.input_dict['depth_hist'][0, 0], 'depth hist shape')
            # print(self.input_dict['q_hist'][0, 0], 'q hist')
            # print(self.input_dict['act_hist'][0, 0], 'act hist')
            # print(self.input_dict['v_hist'][0, 0], 'v hist')
            # print(self.input_dict['goal_pos'], 'goal pos')
            # print(self.input_dict['grasp_cond'], 'grasp cond')
            # print(self.input_dict['intrinsics'], 'intrinsics')
        
        else:
            # update history
            self.input_dict['q_hist'] = torch.cat([self.input_dict['q_hist'][:, 1:], obs[:, :31].unsqueeze(1)], dim=1)
            self.input_dict['v_hist'] = torch.cat([self.input_dict['v_hist'][:, 1:], obs[:, 31:62].unsqueeze(1)], dim=1)
            self.input_dict['ee_fingers'] = torch.cat([self.input_dict['ee_fingers'][:, 1:], ee_b.unsqueeze(1)], dim=1)
            # Action: we use the previous predicted action as the current action history
            depth_hist = -obs[:, 144 : 144 + IMAGE_HEIGHT * IMAGE_WIDTH].reshape(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
            depth_hist = torch.clamp(depth_hist, 0.0, 10.0)
            self.input_dict['depth_hist'] = torch.cat([self.input_dict['depth_hist'][:, 1:], obs[:, 144 : 144 + IMAGE_HEIGHT * IMAGE_WIDTH].unsqueeze(1).reshape(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)], dim=1)
            self.input_dict['depth_hist'] = torch.cat([self.input_dict['depth_hist'][:, 1:], depth_hist], dim=1)

            # self.input_dict['goal_pos'] = goal_c
            # self.input_dict['grasp_cond'] = torch.cat([wrist_c, hand_q], dim=1) # camera frame

        # print(self.input_dict['q_hist'], 'q hist shape after update')
        # print(self.input_dict['act_hist'], 'act hist shape after update')
        # print(self.input_dict['goal_pos'], 'goal pos after update')
        # print(self.input_dict['grasp_cond'], 'grasp cond after update')
        # print(self.input_dict['intrinsics'], 'intrinsics after update')
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # drain previous kernels

        t0 = time.perf_counter()

        action_future, q_future, obj_pose_future = self.model(self.input_dict, run_inference=True)
        global STEP
        STEP += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for this call to finish

        dt = (time.perf_counter() - t0) * 1000.0  # ms
        print(f"[step {STEP}] model() took {dt:.2f} ms")
        if STEP == 1:
            print(action_future, 'action future')
            print(q_future, 'q future')
        # print(q_future.shape, 'q future shape')
        # print(obj_pose_future.shape, 'obj future shape')
        current_action = action_future[:, 0, :]
        # print(current_action, 'current action before postproc')
        self.input_dict['act_hist'] = torch.cat([self.input_dict['act_hist'][:, 1:], current_action.unsqueeze(1)], dim=1)
        return current_action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        # model_dict = torch.load(
        #     DIFFUSION_CONFIG.checkpoint,
        #     map_location="cpu",
        #     weights_only=True
        # )
        # # self.model.load_state_dict(checkpoint['model'])
        # msn, unxpct = self.model.load_state_dict(model_dict["weight"], strict=False)

        load_ckpt_strict(self.model, DIFFUSION_CONFIG.checkpoint, map_location="cpu")
        self.model.to("cuda:0").eval()
        # if msn:
        #     print(f"Missing keys (not found in checkpoint): {len(msn)}")
        #     print(msn)
        # if unxpct:
        #     print(f"Unexpected keys (ignored): {len(unxpct)}")
        #     print(unxpct)
        # if not msn and not unxpct:
        #     print("All keys matched successfully!")
        # if self.normalize_input and 'running_mean_std' in checkpoint:
        #     self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        # env_state = model_dict.get('env_state', None)
        env_state = None
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)
        self.is_rnn = False

    def reset(self):
        # self.init_rnn()
        pass