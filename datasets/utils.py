import numpy as np
import torch
import zarr
from zarr.storage import DirectoryStore
from zarr import LRUStoreCache
from typing import Dict
import json
import os

import utils.pytorch3d_transforms as pytorch3d_transforms


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.as_tensor(x)


def read_zarr_with_cache(fname, mem_gb=16):
    # Configure the underlying store
    store = DirectoryStore(fname)

    # Wrap the store with a cache
    cached_store = LRUStoreCache(store, max_size=mem_gb * 2**30)  # GB cache

    # Open Zarr file with caching
    return zarr.open_group(cached_store, mode="r")


def to_relative_action(actions, anchor_action, qform='xyzw'):
    """
    Compute delta actions where the first delta is relative to anchor,
    and subsequent deltas are relative to the previous timestep.

    Args:
        actions: (..., N, 8)  — future trajectory
        anchor_action: (..., 1, 8) — current pose to treat as timestep -1
        qform: 'xyzw' or 'wxyz' — quaternion format

    Returns:
        delta_actions: (..., N, 8)
    """
    assert actions.shape[-1] == 8
    # Stitch anchor in front and shift everything by one
    prev = torch.cat([anchor_action, actions[..., :-1, :]], -2)  # (..., N, 8)

    rel_pos = actions[..., :3] - prev[..., :3]

    if qform == 'xyzw':
        rel_orn = pytorch3d_transforms.quaternion_multiply(
            actions[..., [6, 3, 4, 5]],
            pytorch3d_transforms.quaternion_invert(prev[..., [6, 3, 4, 5]])
        )[..., [1, 2, 3, 0]]
    elif qform == 'wxyz':
        rel_orn = pytorch3d_transforms.quaternion_multiply(
            actions[..., 3:7],
            pytorch3d_transforms.quaternion_invert(prev[..., 3:7])
        )
    else:
        raise ValueError("Invalid quaternion format")

    gripper = actions[..., -1:]

    return torch.cat([rel_pos, rel_orn, gripper], -1)  # (..., N, 8)

def numpy_quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = np.split(quaternions, 4, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1, keepdims=True)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


def load_json(file_path):
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            json_params = json.load(file_p)
    else:
        json_params = file_path
    return json_params


def write_json(data: Dict, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=1)


def load_scene_cfg(scene_path):
    scene_cfg = np.load(scene_path, allow_pickle=True).item()

    def update_relative_path(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                update_relative_path(v)
            elif k.endswith("_path") and isinstance(v, str):
                d[k] = os.path.join(os.path.dirname(scene_path), v)
        return

    update_relative_path(scene_cfg["scene"])

    return scene_cfg

def quat2mat(q):
    w,x,y,z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]
    ])


# =========================
# Quaternion / SE(3) helpers (wxyz)
# =========================

def normalize_quat_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    q: [..., 4] in [w,x,y,z]. Returns normalized quaternion with w >= 0.
    """
    q = q.to(dtype=torch.float32)
    n = torch.clamp(q.norm(dim=-1, keepdim=True), min=eps)
    q = q / n
    # Make the scalar part non-negative for a consistent canonical form
    q = torch.where(q[..., :1] < 0, -q, q)
    return q

def quat_wxyz_to_R(q: torch.Tensor) -> torch.Tensor:
    """
    q: [..., 4] in [w,x,y,z] -> rotation matrix [..., 3, 3].
    """
    q = normalize_quat_wxyz(q)
    w, x, y, z = q.unbind(dim=-1)

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    r00 = 1 - 2*(yy + zz)
    r01 = 2*(xy - wz)
    r02 = 2*(xz + wy)
    r10 = 2*(xy + wz)
    r11 = 1 - 2*(xx + zz)
    r12 = 2*(yz - wx)
    r20 = 2*(xz - wy)
    r21 = 2*(yz + wx)
    r22 = 1 - 2*(xx + yy)

    R = torch.stack([r00, r01, r02,
                     r10, r11, r12,
                     r20, r21, r22], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R

def R_to_quat_wxyz(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    R: [..., 3, 3] -> quaternion [..., 4] in [w,x,y,z].
    Robust branch-based conversion with w>=0.
    """
    R = R.to(dtype=torch.float32)
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    trace = m00 + m11 + m22
    w = torch.zeros_like(trace)
    x = torch.zeros_like(trace)
    y = torch.zeros_like(trace)
    z = torch.zeros_like(trace)

    # Case 1: trace positive
    mask0 = trace > 0
    s0 = torch.sqrt(trace[mask0] + 1.0 + eps) * 2.0  # s = 4*w
    w[mask0] = 0.25 * s0
    x[mask0] = (m21[mask0] - m12[mask0]) / s0
    y[mask0] = (m02[mask0] - m20[mask0]) / s0
    z[mask0] = (m10[mask0] - m01[mask0]) / s0

    # Case 2: m00 largest
    mask1 = (~mask0) & (m00 >= m11) & (m00 >= m22)
    s1 = torch.sqrt(1.0 + m00[mask1] - m11[mask1] - m22[mask1] + eps) * 2.0
    w[mask1] = (m21[mask1] - m12[mask1]) / s1
    x[mask1] = 0.25 * s1
    y[mask1] = (m01[mask1] + m10[mask1]) / s1
    z[mask1] = (m02[mask1] + m20[mask1]) / s1

    # Case 3: m11 largest
    mask2 = (~mask0) & (~mask1) & (m11 >= m22)
    s2 = torch.sqrt(1.0 + m11[mask2] - m00[mask2] - m22[mask2] + eps) * 2.0
    w[mask2] = (m02[mask2] - m20[mask2]) / s2
    x[mask2] = (m01[mask2] + m10[mask2]) / s2
    y[mask2] = 0.25 * s2
    z[mask2] = (m12[mask2] + m21[mask2]) / s2

    # Case 4: m22 largest
    mask3 = (~mask0) & (~mask1) & (~mask2)
    s3 = torch.sqrt(1.0 + m22[mask3] - m00[mask3] - m11[mask3] + eps) * 2.0
    w[mask3] = (m10[mask3] - m01[mask3]) / s3
    x[mask3] = (m02[mask3] + m20[mask3]) / s3
    y[mask3] = (m12[mask3] + m21[mask3]) / s3
    z[mask3] = 0.25 * s3

    q = torch.stack([w, x, y, z], dim=-1)
    return normalize_quat_wxyz(q, eps=eps)

def pose7_wxyz_to_T(pose: torch.Tensor) -> torch.Tensor:
    """
    pose: [..., 7] as [x,y,z, w,x,y,z] (wxyz) -> homogeneous T [..., 4, 4]
    """
    pose = pose.to(dtype=torch.float32)
    t = pose[..., :3]
    q = pose[..., 3:7]
    R = quat_wxyz_to_R(q)
    T = torch.eye(4, dtype=pose.dtype, device=pose.device).expand(pose.shape[:-1] + (4, 4)).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T

def T_to_pose7_wxyz(T: torch.Tensor) -> torch.Tensor:
    """
    T: [..., 4, 4] -> pose [..., 7] as [x,y,z, w,x,y,z] (wxyz)
    """
    T = T.to(dtype=torch.float32)
    t = T[..., :3, 3]
    q = R_to_quat_wxyz(T[..., :3, :3])
    return torch.cat([t, q], dim=-1)

def T_inv(T: torch.Tensor) -> torch.Tensor:
    """
    Inverse of homogeneous transform T: [..., 4, 4]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    RT = R.transpose(-2, -1)
    Ti = torch.zeros_like(T)
    Ti[..., :3, :3] = RT
    Ti[..., :3, 3] = -(RT @ t.unsqueeze(-1)).squeeze(-1)
    Ti[..., 3, 3] = 1.0
    return Ti

def transform_points(T: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    Apply T (4x4, maps 'from'->'to') to points P[..., N, 3] in the 'from' frame.
    Row-vector convention: (x @ R^T + t).
    """
    R = T[..., :3, :3] # [B, 3, 3]
    t = T[..., :3, 3].unsqueeze(-2) # [B, 1, 3]
    # P = P.unsqueeze(-1) if P.dim() == 2 else P  # [..., N, 3]
    # print(P.shape, 'P shape in transform points')
    # print(R.shape, 'R shape in transform points')
    return (P @ R.transpose(-2, -1)) + t

# =========================
# (Optional) reorder helpers for legacy xyzw data
# =========================

def quat_xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    """
    q_xyzw: [...,4] in [x,y,z,w] -> [...,4] in [w,x,y,z]
    """
    q_xyzw = q_xyzw.to(dtype=torch.float32)
    x, y, z, w = q_xyzw.unbind(dim=-1)
    return normalize_quat_wxyz(torch.stack([w, x, y, z], dim=-1))

def pose7_xyzw_to_wxyz(pose_xyzw: torch.Tensor) -> torch.Tensor:
    """
    pose_xyzw: [...,7] as [x,y,z, qx,qy,qz,qw] -> [...,7] as [x,y,z, w,x,y,z]
    """
    pose_xyzw = pose_xyzw.to(dtype=torch.float32)
    xyz = pose_xyzw[..., :3]
    q_xyzw = pose_xyzw[..., 3:7]
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    return torch.cat([xyz, q_wxyz], dim=-1)
