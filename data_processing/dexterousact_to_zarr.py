#!/usr/bin/env python3
"""
npz_to_zarr_ragged.py — variable-length trajectories → Zarr (ragged/CSR)

Expected per .npz (world frame for all poses unless noted):
- actions:         [T, 31]
- states:          [T, 31, 2]             # (:,:,0)=q, (:,:,1)=v
- keypoints:       [T, 6, 3]              # world; [fingers..., wrist]
- object_poses:    [T, 7]                 # (x,y,z,qx,qy,qz,qw), world
- depth_image:     [T, H, W]              # kept as-is; may be negative
- seg_image:       [H, W]                 # 0/1 initial mask
- camera_extrinsic:[4, 4]                 # world→camera (stored)
- camera_intrinsic:[3, 3]                 # stored
- goal:            [3]                    # **WORLD** frame (updated)
- grasp:           [29]                   # world; [wrist(7), hand_q(22)] (replaces goal_hand_pose)
- robot_pose:      [7] or [6] (optional)  # world; per-episode (fallback allowed)
- reach_length:    int (per-episode)
- grasp_length:    int (per-episode)
- object_asset:    str (per-episode)
- object_scale:    float (per-episode)
- table_pose:      [7] (per-episode; world)
- table_size:      [3] (per-episode)

Outputs (ragged time-series via traj_ptr):
Time-series (concatenated across episodes):
- q_traj(total_T,31), v_traj(total_T,31), ee_fingers(total_T,6,3),
  obj_pose_traj(total_T,7), act_traj(total_T,31), depth_traj(total_T,H,W)

Per-trajectory (B items):
- init_segmentation(B,H,W), goal_pos(B,3), goal_hand_pose(B,29), grasp_cond(B,29),
  intrinsics(B,3,3), extrinsics(B,4,4), robot_pose(B,7), traj_ptr(B+1),
  reach_length(B,), grasp_length(B,), object_asset(B,), object_scale(B,),
  table_pose(B,7), table_size(B,3)
"""

import argparse, glob, os, shutil
import numpy as np
import zarr
from numcodecs import Blosc, VLenUTF8
from tqdm.auto import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Directory of .npz files')
    ap.add_argument('--dst', required=True, help='Output Zarr directory')
    ap.add_argument('--pattern', default='*.npz', help='Glob pattern (default: *.npz)')
    ap.add_argument('--robot-pose-default', type=float, nargs=7,
                    default=[-0.5, 0, 0, 0, 0, 0, 1],
                    help='Fallback robot base pose if missing in files (WORLD)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched {args.src}/{args.pattern}")

    # First pass: sizes, total_T, infer H/W from data, verify consistency
    B = len(files)
    total_T = 0
    lengths = []
    H = W = None

    for f in files:
        with np.load(f, allow_pickle=False) as d:
            T = int(d['actions'].shape[0])
            reach_length = int(d['reach_length']) if 'reach_length' in d.files else 0
            lengths.append(T-reach_length)
            total_T += T


            if H is None and W is None:
                # infer H,W from depth_image [T,H,W]
                if 'depth_image' not in d.files:
                    raise KeyError(f"{os.path.basename(f)} missing 'depth_image'.")
                _T, _H, _W = d['depth_image'].shape
                H, W = int(_H), int(_W)

    lengths = np.asarray(lengths, dtype=np.int64)
    ptr = np.zeros(B+1, dtype=np.int64)
    np.cumsum(lengths, out=ptr[1:])

    # Prepare Zarr
    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    store = zarr.DirectoryStore(args.dst)
    root = zarr.group(store=store, overwrite=True)
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

    # Time-series (ragged)
    q_traj   = root.create_dataset('q_traj',        shape=(total_T, 31),   chunks=(min(8192,total_T), 31),   dtype='f4', compressor=compressor)
    v_traj   = root.create_dataset('v_traj',        shape=(total_T, 31),   chunks=(min(8192,total_T), 31),   dtype='f4', compressor=compressor)
    ee_fing  = root.create_dataset('ee_fingers',    shape=(total_T, 6, 3), chunks=(min(8192,total_T), 6, 3), dtype='f4', compressor=compressor)
    obj_pose = root.create_dataset('obj_pose_traj', shape=(total_T, 7),    chunks=(min(8192,total_T), 7),    dtype='f4', compressor=compressor)
    act_traj = root.create_dataset('act_traj',      shape=(total_T, 31),   chunks=(min(8192,total_T), 31),   dtype='f4', compressor=compressor)
    depth_ts = root.create_dataset('depth_traj',    shape=(total_T, H, W), chunks=(min(64,total_T), H, W),   dtype='f4', compressor=compressor)

    # Per-trajectory core fields
    init_seg = root.create_dataset('init_segmentation', shape=(B, H, W), chunks=(1, H, W), dtype='u1', compressor=compressor)
    goal_pos = root.create_dataset('goal_pos',          shape=(B, 3),     chunks=(1024, 3), dtype='f4', compressor=compressor)  # **WORLD** frame now
    goal_hp  = root.create_dataset('goal_hand_pose',    shape=(B, 29),    chunks=(1024, 29), dtype='f4', compressor=compressor)  # from 'grasp'
    grasp_cd = root.create_dataset('grasp_cond',        shape=(B, 29),    chunks=(1024, 29), dtype='f4', compressor=compressor)  # alias of goal_hand_pose
    intr     = root.create_dataset('intrinsics',        shape=(B, 3, 3),  chunks=(1024, 3, 3), dtype='f4', compressor=compressor)
    extr     = root.create_dataset('extrinsics',        shape=(B, 4, 4),  chunks=(1024, 4, 4), dtype='f4', compressor=compressor)
    rpose    = root.create_dataset('robot_pose',        shape=(B, 7),     chunks=(1024, 7), dtype='f4', compressor=compressor)
    traj_ptr = root.create_dataset('traj_ptr',          shape=(B+1,),     chunks=(B+1,),   dtype='i8', compressor=compressor)
    traj_ptr[:] = ptr

    # New per-trajectory fields from updated dataset
    reach_len = root.create_dataset('reach_length',     shape=(B,),        chunks=(1024,),  dtype='i4', compressor=compressor)
    grasp_len = root.create_dataset('grasp_length',     shape=(B,),        chunks=(1024,),  dtype='i4', compressor=compressor)
    obj_asset = root.create_dataset('object_asset',     shape=(B,),        chunks=(1024,),  dtype=object, object_codec=VLenUTF8())
    obj_scale = root.create_dataset('object_scale',     shape=(B,),        chunks=(1024,),  dtype='f4', compressor=compressor)
    table_ps  = root.create_dataset('table_pose',       shape=(B, 7),      chunks=(1024,7), dtype='f4', compressor=compressor)
    table_sz  = root.create_dataset('table_size',       shape=(B, 3),      chunks=(1024,3), dtype='f4', compressor=compressor)

    root.attrs.update({
        'layout': 'ragged_concat',
        'frame': 'world',                         # all *poses* in world frame
        'goal_pos_frame': 'world',                # UPDATED (was 'camera')
        'ee_frame': 'world',
        'extrinsics_semantic': 'world_to_camera',
        'depth_convention': 'kept_as_is; may be negative',  # e.g., some sims use negative z-forward
        'B': int(B),
        'total_T': int(total_T),
        'H': int(H), 'W': int(W),
        'ee_order': ['little','ring','middle','fore','thumb','wrist'],
        'grasp_cond_explain': 'Identical to goal_hand_pose (world): [wrist(7), hand_q(22)]',
        'goal_hand_pose_source': "npz['grasp'] if present else npz['goal_hand_pose']",
    })

    rpose_default = np.asarray(args.robot_pose_default, dtype=np.float32)

    # Second pass: write
    for i, f in enumerate(tqdm(files, total=len(files), desc="Writing episodes", unit="ep"), start=0):
        with np.load(f, allow_pickle=False) as d:
            T = lengths[i]
            s, e = ptr[i], ptr[i+1]

            actions = d['actions'].astype(np.float32)             # (T,31)
            states  = d['states'].astype(np.float32)              # (T,31,2)
            obj     = d['object_poses'].astype(np.float32)        # (T,7) world
            depth   = d['depth_image'].astype(np.float32)         # (T,H,W) kept as-is (possibly negative)
            seg     = d['seg_image'].astype(np.uint8)             # (H,W)
            ext     = d['camera_pose_cv'].astype(np.float32)    # (4,4) world→camera
            intri   = d['camera_intrinsic'].astype(np.float32)    # (3,3)
            goal    = d['goal'].astype(np.float32)                # (3,) WORLD
            kps     = d['keypoints'].astype(np.float32)           # (T,6,3) WORLD

            # # Time alignment to T
            # if depth.shape[0] != T:   depth  = depth[:T]
            # if obj.shape[0] != T:     obj    = obj[:T]
            # if states.shape[0] != T:  states = states[:T]
            # if actions.shape[0] != T: actions= actions[:T]
            # if kps.shape[0] != T:     kps    = kps[:T]

            # remove "reach" phase data
            reach_length = int(d['reach_length']) if 'reach_length' in d.files else 0
            actions = actions[reach_length:]
            states  = states[reach_length:]
            obj     = obj[reach_length:]
            depth   = depth[reach_length:]
            kps     = kps[reach_length:]
            print(f"{os.path.basename(f)}: original T={T}, reach_length={reach_length}, writing T={actions.shape[0]}")

            # Goal hand pose (29): prefer 'grasp', fallback to 'goal_hand_pose'
            if 'grasp' in d.files:
                ghp = d['grasp'].astype(np.float32).reshape(-1)
            elif 'goal_hand_pose' in d.files:
                ghp = d['goal_hand_pose'].astype(np.float32).reshape(-1)
            else:
                raise KeyError(f"{os.path.basename(f)}: missing 'grasp' (preferred) or 'goal_hand_pose' (fallback).")
            if ghp.shape[0] != 29:
                raise ValueError(f"{os.path.basename(f)}: 29-dim grasp/goal_hand_pose expected, got {ghp.shape}.")

            # Optional: robot_pose
            robot_pose_in = d['robot_pose']

            # time-series writes
            q_traj[s:e, :]          = states[:, :, 0]
            v_traj[s:e, :]          = states[:, :, 1]
            act_traj[s:e, :]        = actions
            obj_pose[s:e, :]        = obj
            ee_fing[s:e, :, :]      = kps
            depth_ts[s:e, :, :]     = depth

            # per-episode writes
            init_seg[i, :, :]       = seg
            goal_pos[i, :]          = goal
            intr[i, :, :]           = intri
            extr[i, :, :]           = ext
            rpose[i, :]             = robot_pose_in

            goal_hp[i, :]           = ghp
            grasp_cd[i, :]          = ghp  # alias

            # new per-episode fields (robust defaults if missing)
            reach_len[i]            = int(d['reach_length']) if 'reach_length' in d.files else -1
            grasp_len[i]            = int(d['grasp_length']) if 'grasp_length' in d.files else -1
            obj_asset[i]            = str(d['object_asset']) if 'object_asset' in d.files else ''
            obj_scale[i]            = float(d['object_scale']) if 'object_scale' in d.files else 1.0
            table_ps[i, :]          = (d['table_pose'].astype(np.float32).reshape(-1) if 'table_pose' in d.files
                                       else np.array([0,0,0,0,0,0,1], dtype=np.float32))
            table_sz[i, :]          = (d['table_size'].astype(np.float32).reshape(-1) if 'table_size' in d.files
                                       else np.array([0,0,0], dtype=np.float32))

    zarr.convenience.consolidate_metadata(store)
    print(f"Done. Wrote {B} episodes, total_T={total_T}, H={H}, W={W}. Zarr at: {args.dst}")

if __name__ == '__main__':
    main()
