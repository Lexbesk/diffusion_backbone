
"""

Dataset for DexterousAct grasps. (everything is in world frame, robot is at (-0.5, 0, 0) looking towards +x)
    Keys:
    - q_traj: (B, T, 31)  Trajectory of joint angles                                
    - v_traj: (B, T, 31)  Trajectory of joint velocities                                           
    - ee_fingers: (B, T, 6, 3)  Trajectory of 5fingers+wrist positions, in the order of [little, ring, middle, fore, thumb, wrist]                      
    - obj_pose_traj: (B, T, 7)  Trajectory of object pose                          
    - act_traj: (B, T, 31)  Trajectory of full joint angles, executed or teacher's                                              
    - depth_traj: (B, T, H, W)  Trajectory of depth image observations  
    - init_segmentation: (B, H, W)  Initial segmentation mask of the object in the depth image                                                      
    - goal_pos: (B, 3)  target object position (camera frame)
    - grasp_cond: (B, 7 + 22)  target grasp condition (camera frame)
    - intrinsics: (B, 3, 3)  camera intrinsics
    - extrinsics: (B, 4, 4)  camera extrinsics (world→camera)
    - robot_pose: (B, 7)  robot base pose (world frame)

"""

import numpy as np
import zarr
import json
from torch.utils.data import Dataset
from .utils import to_tensor, read_zarr_with_cache, T_to_pose7_wxyz, pose7_wxyz_to_T, T_inv, transform_points, pose7_xyzw_to_wxyz
from .base import BaseDataset
from typing import Dict, Any
import os
import torch

class DexterousActZarrDataset(Dataset):
    """
    Each __getitem__ returns ONE EPISODE dict (CPU tensors / numpy),
    leaving batching/windowing to collate_fn.
    """
    def __init__(self, root: str, mem_limit=8, chunk_size=1, copies=1):
        super().__init__()
        self.root = zarr.open(root, mode='r')

        # time-series (concatenated)
        self.q_traj        = self.root['q_traj']         # [total_T, 31]
        self.v_traj        = self.root['v_traj']         # [total_T, 31]
        self.ee_fingers    = self.root['ee_fingers']     # [total_T, 6, 3] (WORLD)
        self.obj_pose_traj = self.root['obj_pose_traj']  # [total_T, 7] (WORLD, xyzw)
        self.act_traj      = self.root['act_traj']       # [total_T, 31]
        self.depth_traj    = self.root['depth_traj']     # [total_T, H, W] (float32)

        # per-episode
        self.init_segmentation = self.root['init_segmentation']  # [B, H, W]
        self.goal_pos          = self.root['goal_pos']           # [B, 3] (CAMERA)
        self.goal_hand_pose    = self.root['goal_hand_pose']     # [B, 29] (WORLD)
        self.grasp_cond        = self.root['grasp_cond']         # [B, 29] (WORLD, =goal_hand_pose)
        self.intrinsics        = self.root['intrinsics']         # [B, 3, 3]
        self.extrinsics        = self.root['extrinsics']         # [B, 4, 4] (WORLD→CAMERA)
        self.robot_pose        = self.root['robot_pose']         # [B, 7] (WORLD, base pose)
        self.traj_ptr          = self.root['traj_ptr']           # [B+1]

        self.B = int(self.root.attrs['B'])
        self.H = int(self.root.attrs['H'])
        self.W = int(self.root.attrs['W'])

    def __len__(self):
        return self.B

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = int(self.traj_ptr[idx])
        e = int(self.traj_ptr[idx + 1])

        # Slice numpy; convert to torch only if it helps downstream
        ep = {
            "q_traj":        torch.from_numpy(self.q_traj[s:e]),                 # [T,31]
            "v_traj":        torch.from_numpy(self.v_traj[s:e]),                 # [T,31]
            "ee_fingers":    torch.from_numpy(self.ee_fingers[s:e]),             # [T,6,3] (WORLD)
            "obj_pose_traj": torch.from_numpy(self.obj_pose_traj[s:e]),          # [T,7]   (WORLD, xyzw)
            "act_traj":      torch.from_numpy(self.act_traj[s:e]),               # [T,31]
            "depth_traj":    -torch.from_numpy(self.depth_traj[s:e]).unsqueeze(-1), # [T,H,W,1]
            "init_segmentation": torch.from_numpy(self.init_segmentation[idx]),  # [H,W]
            "goal_pos":          torch.from_numpy(self.goal_pos[idx]),           # [3] (CAMERA)
            "goal_hand_pose":    torch.from_numpy(self.goal_hand_pose[idx]),     # [29] (WORLD)
            "grasp_cond":        torch.from_numpy(self.grasp_cond[idx]),         # [29] (WORLD, same)
            "intrinsics":        torch.from_numpy(self.intrinsics[idx]),         # [3,3]
            "extrinsics":        torch.from_numpy(self.extrinsics[idx]),         # [4,4] WORLD→CAMERA
            "robot_pose":        torch.from_numpy(self.robot_pose[idx]),         # [7] (WORLD, base pose)
        }
        return ep


def _hist_indices(t: int, nhist: int, T: int):
    lo = max(0, t - nhist + 1)
    hi = t + 1
    need = nhist - (hi - lo)
    return lo, hi, need

def _future_indices(t: int, nfuture: int):
    return t+1, t+1+nfuture


def _unproject_masked(depth0: torch.Tensor, K: torch.Tensor, mask: torch.Tensor, N: int) -> torch.Tensor:
    """
    depth0: (H,W) or (H,W,1) float32 depth in meters; first frame of the episode
    K:      (3,3) intrinsics
    mask:   (H,W) or (H,W,1) uint8; nonzero = foreground
    N:      output point count
    returns (N,3) in CAMERA frame
    """
    D = depth0.squeeze(-1).float()        # (H,W)
    M = (mask.squeeze(-1) > 0)            # (H,W) bool

    ys, xs = torch.where(M)
    if ys.numel() == 0:
        # no pixels in mask -> zeros
        return torch.zeros((N, 3), dtype=torch.float32)

    z = D[ys, xs]
    valid = torch.isfinite(z) & (z > 0)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if z.numel() == 0:
        return torch.zeros((N, 3), dtype=torch.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (xs.float() - cx) / fx * z
    Y = (ys.float() - cy) / fy * z
    P = torch.stack([X, Y, z], dim=-1)    # (M,3) CAM

    # fixed-size sampling/padding for batching
    M = P.shape[0]
    if M >= N:
        idx = torch.randperm(M)[:N]
        P = P[idx]
    else:
        reps = (N + M - 1) // M
        P = P.repeat(reps, 1)[:N]
    return P


def make_collate_train(nhist: int, nfuture: int, K: int, include_depth: bool = True, init_pcl_n=1024, test_mode=False):
    """
    Returns a collate_fn for DataLoader. Each episode contributes K random windows.
    """
    def collate_fn(episodes):
        # Count total windows
        times = []
        total = 0
        # print(nfuture, 'collate nfuture')
        for ep in episodes:
            T = ep["q_traj"].shape[0]
            hi = T - nfuture - 1
            # if test_mode:
            #     hi = 0
            if hi < 0:
                times.append([])
                continue
            idx = torch.randint(0, hi+1, (K,)).tolist()
            # idx = torch.randint(10, 11, (K,)).tolist() # fix to 5
            # print(idx, 'idx')
            times.append(idx)
            total += len(idx)
        if total == 0:
            raise RuntimeError("No valid windows. Check nfuture vs. episode lengths.")

        Dq  = episodes[0]["q_traj"].shape[-1]      # 31
        Dee = episodes[0]["ee_fingers"].shape[-2]  # 6
        H   = episodes[0]["depth_traj"].shape[-3] if include_depth else None
        W   = episodes[0]["depth_traj"].shape[-2] if include_depth else None

        # action_diff = episodes[0]["act_traj"][1:, :] - episodes[0]["act_traj"][:-1, :]
        
        # action_state_diff = episodes[0]["act_traj"][:-1, :] - episodes[0]["q_traj"][1:, :]
        # action_state_diff_scale = action_state_diff.abs().mean(dim=1)
        # action_scale = episodes[0]["act_traj"].abs().mean(dim=1)
        # state_scale = episodes[0]["q_traj"].abs().mean(dim=1)
        # # print(action_diff, 'action_diff_scale')
        # print(action_scale, 'action_scale')
        # print(state_scale, 'state_scale') pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
        # print(action_state_diff_scale, 'action_diff_scale')

        # Preallocate (CPU)
        q_hist       = torch.empty((total, nhist, Dq), dtype=torch.float32)
        v_hist       = torch.empty((total, nhist, Dq), dtype=torch.float32)
        ee_hist_base = torch.empty((total, nhist, Dee, 3), dtype=torch.float32) # in ROBOTBASE frame
        obj_hist_cam = torch.empty((total, nhist, 7), dtype=torch.float32)
        act_hist     = torch.empty((total, nhist, Dq), dtype=torch.float32)
        if include_depth:
            depth_hist = torch.empty((total, nhist, H, W, 1), dtype=torch.float32)

        goal_pos   = torch.empty((total, 3), dtype=torch.float32)
        grasp_cond = torch.empty((total, 7+22), dtype=torch.float32)  # wrist(cam,wxyz)+hand_q
        intrinsics = torch.empty((total, 3, 3), dtype=torch.float32)
        extrinsics = torch.empty((total, 4, 4), dtype=torch.float32)

        obj_future = torch.empty((total, nfuture, 7), dtype=torch.float32)  # CAMERA (wxyz)
        act_future = torch.empty((total, nfuture, Dq), dtype=torch.float32)
        q_future = torch.empty((total, nfuture, Dq), dtype=torch.float32)
        init_seg     = torch.empty((total, H, W, 1), dtype=torch.uint8)  # replicate per window
        object_scale = torch.empty((total,), dtype=torch.float32)        # replicate per window
        object_asset = []
        obj_init_pcl_cam  = torch.empty((total, init_pcl_n, 3), dtype=torch.float32)

        w = 0
        for ep, tlist in zip(episodes, times):
            if not tlist: continue
            T = ep["q_traj"].shape[0]

            # Episode transforms
            T_wc = ep["extrinsics"].float()                      # WORLD->CAMERA
            pose_wxyz = pose7_xyzw_to_wxyz(ep["robot_pose"].float().unsqueeze(0))  # (1,7) wxyz
            T_wb = pose7_wxyz_to_T(pose_wxyz).squeeze(0)  # BASE->WORLD
            T_bw = T_inv(T_wb)                                    # WORLD->BASE

            # print(T_wc, 'T_w to c')
            # print(T_wb, 'T_w to b')
            # print(T_bw, 'T_b to w (should be robot pose)')

            # grasp_cond conversion: world wrist -> camera wrist (wxyz), keep hand_q
            gc_world = ep["grasp_cond"].float()
            wrist_w  = gc_world[:7].unsqueeze(0)                  # [1,7] world
            wrist_c  = T_to_pose7_wxyz(T_wc @ pose7_wxyz_to_T(pose7_xyzw_to_wxyz(wrist_w))).squeeze(0)  # [7]
            hand_q   = gc_world[7:]                               # [22]

            # print(wrist_c, 'wrist in camera frame')
            # print(hand_q, 'hand q')
            seg = ep["init_segmentation"]        # (H,W) uint8
            if seg.ndim == 2:
                seg = seg.unsqueeze(-1)              # (H,W,1)
            seg = seg.to(torch.uint8)
            
            depth0 = ep["depth_traj"][0].float()  # first frame of the EPISODE (not slice)
            K_cam  = ep["intrinsics"].float()
            pcl_cam_ep = _unproject_masked(depth0, K_cam, seg, init_pcl_n)
            # print(pcl_cam_ep, 'pcl cam ep')
            scale_val = float(ep["object_scale"]) if "object_scale" in ep else 1.0
            asset_val = str(ep["object_asset"]) if "object_asset" in ep else ""
            
            goal_w = ep["goal_pos"].float()          # (3,) WORLD
            goal_c = transform_points(T_wc, goal_w.unsqueeze(0)).squeeze(0)  # (3,) CAMERA

            for t in tlist:
                # history
                lo, hi, need = _hist_indices(t, nhist, T)
                q   = ep["q_traj"][lo:hi]
                v   = ep["v_traj"][lo:hi]
                ee  = ep["ee_fingers"][lo:hi]         # WORLD
                obj = ep["obj_pose_traj"][lo:hi]      # WORLD (xyzw)
                act = ep["act_traj"][lo:hi]
                if include_depth:
                    dep = ep["depth_traj"][lo:hi]
                    
                first_q   = q[0:1]
                first_v   = v[0:1]
                first_ee  = ee[0:1]
                first_obj = obj[0:1]
                first_act = act[0:1]
                if include_depth:
                    first_dep = dep[0:1]

                # if need > 0:
                #     print(first_q, 'first q')
                #     print(first_v, 'first v')
                #     print(first_ee, 'first ee')
                #     print(first_obj, 'first obj')
                #     print(first_act, 'first act')
                #     if include_depth:
                #         print(first_dep, 'first dep')
                #     # print(ep["act_traj"][0:4], 'act first two frames')

                if need > 0:
                    q   = torch.cat([first_q.expand(need, *first_q.shape[1:]),   q],   0)
                    v   = torch.cat([first_v.expand(need, *first_v.shape[1:]),   v],   0)
                    ee  = torch.cat([first_ee.expand(need, *first_ee.shape[1:]), ee],  0)
                    obj = torch.cat([first_obj.expand(need, *first_obj.shape[1:]),obj],0)
                    act = torch.cat([first_act.expand(need, *first_act.shape[1:]),act],0)
                    if include_depth:
                        dep = torch.cat([first_dep.expand(need, *first_dep.shape[1:]), dep], 0)

                # WORLD->CAMERA for object poses (hist)
                T_obj_w = pose7_wxyz_to_T(pose7_xyzw_to_wxyz(obj))                    # [nhist,4,4]
                T_obj_c = T_wc.unsqueeze(0) @ T_obj_w
                obj_c   = T_to_pose7_wxyz(T_obj_c)                 # [nhist,7]

                # WORLD->BASE for ee points
                P = ee.reshape(nhist, -1, 3)                       # [nhist,6,3]
                Pb = transform_points(T_bw, P)                     # [nhist,6,3]
                ee_b = Pb.view(nhist, Dee, 3)

                # if need > 0:
                #     print(ee_b, 'ee_b')

                # futures
                flo, fhi = _future_indices(t, nfuture)
                obj_f_w  = ep["obj_pose_traj"][flo:fhi]            # WORLD (xyzw)
                act_f    = ep["act_traj"][flo:fhi]
                q_f      = ep["q_traj"][flo:fhi]
                T_objf_w = pose7_wxyz_to_T(pose7_xyzw_to_wxyz(obj_f_w))
                T_objf_c = T_wc.unsqueeze(0) @ T_objf_w
                obj_f_c  = T_to_pose7_wxyz(T_objf_c)               # [nfuture,7]

                # write
                q_hist[w]       = q
                v_hist[w]       = v
                ee_hist_base[w] = ee_b
                obj_hist_cam[w] = obj_c
                act_hist[w]     = act
                if include_depth:
                    depth_hist[w] = dep
                goal_pos[w]     = goal_c
                intrinsics[w]   = ep["intrinsics"].float()
                extrinsics[w]   = ep["extrinsics"].float()
                grasp_cond[w]   = torch.cat([wrist_c, hand_q], 0)
                obj_future[w]   = obj_f_c
                act_future[w]   = act_f
                q_future[w]     = q_f
                init_seg[w]     = seg
                object_scale[w] = scale_val
                object_asset.append(asset_val)
                obj_init_pcl_cam[w] = pcl_cam_ep
                w += 1

        batch = {
            # observations
            "q_hist":        q_hist,               # (B,nhist,31)
            "v_hist":        v_hist,               # (B,nhist,31)
            "ee_fingers":    ee_hist_base,         # (B,nhist,6,3) BASE
            "obj_pose_hist": obj_hist_cam,         # (B,nhist,7)   CAMERA, [x y z w x y z]
            "act_hist":      act_hist,             # (B,nhist,31)
            "goal_pos":      goal_pos,             # (B,3)         CAMERA
            "grasp_cond":    grasp_cond,           # (B,7+22)      CAMERA wrist + hand_q
            "intrinsics":    intrinsics,           # (B,3,3)
            "extrinsics":    extrinsics,           # (B,4,4) WORLD->CAMERA
            # targets
            "obj_pose_future": obj_future,         # (B,nfuture,7) CAMERA, [x y z w x y z]
            "act_future":      act_future,         # (B,nfuture,31)
            "q_future":       q_future,            # (B,nfuture,31)
            # NEW meta
            "init_seg":       init_seg,            # (B,H,W,1) uint8
            "object_scale":   object_scale,        # (B,)
            "object_asset":   object_asset,        # list[str] length B
        }


        if include_depth:
            batch["depth_hist"] = depth_hist       # (B,nhist,H,W,1)
        batch["obj_init_pcl_cam"] = obj_init_pcl_cam
        return batch
    return collate_fn



def make_collate_eval(nhist: int, include_depth: bool = True, init_pcl_n=1024):
    """
    Evaluation collate: one window per episode at t = 0.
    No future targets are produced.
    """
    def collate_fn(episodes):
        B = len(episodes)
        if B == 0:
            raise RuntimeError("Empty eval batch.")

        Dq  = episodes[0]["q_traj"].shape[-1]      # 31
        Dee = episodes[0]["ee_fingers"].shape[-2]  # 6
        if include_depth:
            H = episodes[0]["depth_traj"].shape[-3]
            W = episodes[0]["depth_traj"].shape[-2]

        # Preallocate (CPU)
        q_hist       = torch.empty((B, nhist, Dq), dtype=torch.float32)
        v_hist       = torch.empty((B, nhist, Dq), dtype=torch.float32)
        ee_hist_base = torch.empty((B, nhist, Dee, 3), dtype=torch.float32)
        obj_hist_cam = torch.empty((B, nhist, 7), dtype=torch.float32)  # CAMERA, wxyz
        act_hist     = torch.empty((B, nhist, Dq), dtype=torch.float32)
        if include_depth:
            depth_hist = torch.empty((B, nhist, H, W, 1), dtype=torch.float32)

        goal_pos   = torch.empty((B, 3), dtype=torch.float32)
        grasp_cond = torch.empty((B, 7+22), dtype=torch.float32)        # wrist(cam,wxyz) + 22
        intrinsics = torch.empty((B, 3, 3), dtype=torch.float32)
        extrinsics = torch.empty((B, 4, 4), dtype=torch.float32)
        init_seg     = torch.empty((B, H, W, 1), dtype=torch.uint8)  # replicate per window
        object_scale = torch.empty((B,), dtype=torch.float32)        # replicate per window
        object_asset = []
        obj_init_pcl_cam  = torch.empty((total, init_pcl_n, 3), dtype=torch.float32)

        for i, ep in enumerate(episodes):
            Tlen = ep["q_traj"].shape[0]
            # Episode transforms
            T_wc = ep["extrinsics"].float()                                    # WORLD->CAMERA
            T_wb = pose7_wxyz_to_T(pose7_xyzw_to_wxyz(ep["robot_pose"].float().unsqueeze(0))).squeeze(0)  # BASE->WORLD
            T_bw = T_inv(T_wb)                                                 # WORLD->BASE

            # Choose and slice history
            lo = 0
            hi = 1
            need = nhist - 1

            q   = ep["q_traj"][lo:hi]
            v   = ep["v_traj"][lo:hi]
            ee  = ep["ee_fingers"][lo:hi]             # WORLD
            obj = ep["obj_pose_traj"][lo:hi]          # WORLD (xyzw)
            act = ep["act_traj"][lo:hi]
            if include_depth:
                dep = ep["depth_traj"][lo:hi]

            # Left-pad by copying the first frame of the sliced history
            if need > 0:
                q   = torch.cat([q[0:1].expand(need, *q.shape[1:]),     q],   dim=0)
                v   = torch.cat([v[0:1].expand(need, *v.shape[1:]),     v],   dim=0)
                ee  = torch.cat([ee[0:1].expand(need, *ee.shape[1:]),   ee],  dim=0)
                obj = torch.cat([obj[0:1].expand(need, *obj.shape[1:]), obj], dim=0)
                act = torch.cat([act[0:1].expand(need, *act.shape[1:]), act], dim=0)
                if include_depth:
                    dep = torch.cat([dep[0:1].expand(need, *dep.shape[1:]), dep], dim=0)

            # WORLD->CAMERA for object poses (hist)  (output wxyz)
            T_obj_w = pose7_wxyz_to_T(pose7_xyzw_to_wxyz(obj))               # [nhist,4,4]
            T_obj_c = T_wc.unsqueeze(0) @ T_obj_w
            obj_c   = T_to_pose7_wxyz(T_obj_c)           # [nhist,7]

            # WORLD->BASE for ee points
            ee_b = transform_points(T_bw, ee.view(nhist, -1, 3)).view(nhist, Dee, 3)

            # grasp_cond: world wrist -> camera wrist (wxyz); keep hand_q
            gc_world = ep["grasp_cond"].float()          # [29] (wrist(7 wxyz WORLD), hand_q(22))
            wrist_w  = gc_world[:7].unsqueeze(0)         # [1,7]
            wrist_c  = T_to_pose7_wxyz(T_wc @ pose7_wxyz_to_T(pose7_xyzw_to_wxyz(wrist_w))).squeeze(0)  # [7]
            hand_q   = gc_world[7:]                      # [22]
            seg = ep["init_segmentation"]        # (H,W) uint8
            if seg.ndim == 2:
                seg = seg.unsqueeze(-1)              # (H,W,1)
            seg = seg.to(torch.uint8)
                
            depth0 = ep["depth_traj"][0].float()  # first frame of the EPISODE (not slice)
            K_cam  = ep["intrinsics"].float()
            pcl_cam_ep = _unproject_masked(depth0, K_cam, seg, init_pcl_n)

            scale_val = float(ep["object_scale"]) if "object_scale" in ep else 1.0
            asset_val = str(ep["object_asset"]) if "object_asset" in ep else ""
            goal_w = ep["goal_pos"].float()          # (3,) WORLD
            goal_c = transform_points(T_wc, goal_w.unsqueeze(0)).squeeze(0)  # (3,) CAMERA

            # Write to batch
            q_hist[i]       = q
            v_hist[i]       = v
            ee_hist_base[i] = ee_b
            obj_hist_cam[i] = obj_c
            act_hist[i]     = act
            if include_depth:
                depth_hist[i] = dep
            goal_pos[i]   = goal_c
            intrinsics[i] = ep["intrinsics"].float()
            extrinsics[i] = ep["extrinsics"].float()
            grasp_cond[i] = torch.cat([wrist_c, hand_q], dim=0)
            # No future targets
            init_seg[i]   = seg
            object_scale[i] = scale_val
            object_asset.append(asset_val)
            obj_init_pcl_cam[i] = pcl_cam_ep
            

        batch = {
            "q_hist":        (q_hist),              # (B,nhist,31)
            "v_hist":        (v_hist),              # (B,nhist,31)
            "ee_fingers":    (ee_hist_base),        # (B,nhist,6,3) BASE
            "obj_pose_hist": (obj_hist_cam),        # (B,nhist,7)   CAMERA, [x y z w x y z]
            "act_hist":      (act_hist),            # (B,nhist,31)
            "goal_pos":      (goal_pos),            # (B,3)         CAMERA
            "grasp_cond":    (grasp_cond),          # (B,7+22)      CAMERA wrist + hand_q
            "intrinsics":    (intrinsics),          # (B,3,3)
            "extrinsics":    (extrinsics),          # (B,4,4) WORLD->CAMERA
            # NEW meta
            "init_seg":       init_seg,            # (B,H,W,1) uint8
            "object_scale":   object_scale,        # (B,)
            "object_asset":   object_asset,        # list[str] length B
        }

        
        if include_depth:
            batch["depth_hist"] = (depth_hist)      # (B,nhist,H,W,1)
        batch["obj_init_pcl_cam"] = obj_init_pcl_cam
        return batch

    return collate_fn
