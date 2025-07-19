# from .pk_utils import build_chain_from_mjcf_path
import os
from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_kinematics.chain import Chain
from copy import deepcopy
from torch.amp import autocast
from pytorch3d.transforms import quaternion_to_matrix, quaternion_apply

import mujoco
from pytorch_kinematics import chain, frame
from pytorch_kinematics.mjcf import _build_chain_recurse
import pytorch_kinematics.transforms as tf
from typing import Union


def build_chain_from_mjcf_path(path, body: Union[None, str, int] = None):
    """
    Build a Chain object from MJCF data.

    Parameters
    ----------
    path : str
        MJCF path
    body : str or int, optional
        The name or index of the body to use as the root of the chain. If None, body idx=0 is used.

    Returns
    -------
    chain.Chain
        Chain object created from MJCF.
    """
    m = mujoco.MjModel.from_xml_path(path)
    if body is None:
        root_body = m.body(0)
    else:
        root_body = m.body(body)
    root_frame = frame.Frame(root_body.name,
                             link=frame.Link(root_body.name,
                                             offset=tf.Transform3d(rot=root_body.quat, pos=root_body.pos)),
                             joint=frame.Joint())
    _build_chain_recurse(m, root_frame, root_body)
    return chain.Chain(root_frame)

def batch_forward_kinematics(chain, pose):
    '''
    Differentiable forward kinematics with Pytorch Kinematics

    Parameters
    ----------
    chain: pytorch_kinematics.chain.Chain
        kinamtics chain object for performing differentiable forward kinematics
    pose: torch.tensor BxJ

    Returns
    -------
    A: torch.tensor BxJx4x4
        Affine transformation matrix of all joints
    '''
    tg = chain.forward_kinematics(pose)
    joints = list(tg.keys())
    # print(joints)
    joint_names = [j for j in chain.get_joint_parameter_names()]
    # print(joint_names)
    name_list = []
    param2child = {}
    for joint, child_links in chain.get_joints_and_child_links():  # public API
        param2child[joint.name] = child_links[0].name               # pivot link
    for pname in chain.get_joint_parameter_names():                # 22 × …
        child_link = param2child[pname]
        name_list.append(child_link)
    # print(name_list)
    # joint_transforms = {name: tg[name] for name in joint_names if name in tg}
    # joints = chain.get_frame_names()
    A = torch.stack([tg[joint]._matrix for joint in name_list],dim=1)
    xyz = A[:, :, :3, 3]
    # print(f"Batch forward kinematics: {A.shape}, {xyz.shape}")
    return A, xyz


def joints_to_world(joint_xyz_hand: torch.Tensor,
                    wrist_pos: torch.Tensor,
                    wrist_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Returns the joint positions in the world frame, same leading dims as input.
    """
    # Ensure everything has a batch dimension for broadcasting
    if len(joint_xyz_hand.shape) == 2:                 # (N,3) -> (1,N,3)
        joint_xyz_hand = joint_xyz_hand.unsqueeze(0)
    if len(wrist_pos.shape) == 1:                      # (3,)  -> (1,3)
        wrist_pos = wrist_pos.unsqueeze(0)
    if len(wrist_quat_wxyz.shape) == 1:                # (4,)  -> (1,4)
        wrist_quat_wxyz = wrist_quat_wxyz.unsqueeze(0)

    # Rotate the local points
    joint_xyz_rot = quaternion_apply(
        wrist_quat_wxyz.unsqueeze(1).expand(-1, joint_xyz_hand.size(1), -1),
        joint_xyz_hand
    )                                             # (B,N,3)

    # Translate
    joint_xyz_world = joint_xyz_rot + wrist_pos.unsqueeze(1)  # (B,N,3)
    return joint_xyz_world.squeeze(0)  


def get_joint_positions(chain, grasp_qpos: torch.Tensor, pose_normalized=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get joint positions in the world frame from the chain and grasp qpos.

    Parameters
    ----------
    chain : pytorch_kinematics.chain.Chain
        The kinematic chain.
    grasp_qpos : torch.Tensor
        The batched grasp qpos tensor, shape [B, 29]

    Returns
    -------
    torch.Tensor
        Joint positions in the world frame.
    """
    # print(f"Grasp qpos: {grasp_qpos}")
    pose = grasp_qpos[:, 7:]  # Exclude the first 7 elements (translation and quaternion)
    trans = grasp_qpos[:, :3]  # First 3 elements are translation
    quat = grasp_qpos[:, 3:7]  # Next 4 elements are quaternion
    
    if pose_normalized:
        lower_limits, upper_limits = chain.get_joint_limits()
        pose = (pose + 1) * (upper_limits - lower_limits) / 2 + lower_limits
        
    A, xyz = batch_forward_kinematics(chain, pose)
    xyz = joints_to_world(xyz, trans, quat)
    
    return xyz
    
    
if __name__ == "__main__":

    dtype = torch.float32
    device = 'cuda'

    chain = build_chain_from_mjcf_path('/data/user_data/austinz/Robots/DexGraspBench/assets/hand/shadow/right_hand.xml')
    num_frames = len(chain.joint_type_indices) #might be different from num_joints, includes world frame too
    print(f'Number of frames in chain: {num_frames}')
    chain = chain.to(dtype=dtype, device=device)

    lower_limits, upper_limits = chain.get_joint_limits()
    print("lower limits", lower_limits)
    print("upper limits", upper_limits)
    lower_limits = torch.tensor(lower_limits, dtype=dtype, device=device)
    upper_limits = torch.tensor(upper_limits, dtype=dtype, device=device)

    pose = torch.rand((1, 22), dtype=dtype, device=device) * (upper_limits - lower_limits) + lower_limits
    pose_normalized = False

    input_npz_path = '/data/user_data/austinz/Robots/manipulation/analogical_manipulation/train_logs/DEXONOMY_7k_pcdcentric/run_Jul13_grasp_denoiser-Dexonomy-lr1e-4-constant-rectified_flow-B32-Bval8-DT10/visualization_denoise_process_batch2.npz'
    data = np.load(input_npz_path, allow_pickle=True)
    grasps = data['grasps'] # [B, T, 29]
    print(grasps.shape)
    partial_points = data['partial_points'] # [B, 4096, 3]

    for i in range(grasps.shape[0]):
        for j in range(grasps.shape[1]):
            grasp_data = {}
            grasp_data['grasp_qpos'] = grasps[i, :, :]
            grasp_data['pregrasp_qpos'] = grasps[i, :, :]
            grasp_data['squeeze_qpos'] = grasps[i, :, :]
            grasp_data['partial_points'] = partial_points[i]

            
            # pose = torch.tensor(grasp_data['grasp_qpos'][7:], dtype=dtype, device=device)
            # trans = torch.tensor(grasp_data['grasp_qpos'][:3], dtype=dtype, device=device)
            # quat = torch.tensor(grasp_data['grasp_qpos'][3:7], dtype=dtype, device=device)
            grasp = torch.tensor(grasp_data['grasp_qpos'], dtype=dtype, device=device)
                        
            xyz = get_joint_positions(chain, grasp, pose_normalized=pose_normalized)
                        
            # if pose_normalized:
            #     pose = (pose + 1) * (upper_limits - lower_limits) / 2 + lower_limits
            #     print(pose)
                
            # A, xyz = batch_forward_kinematics(chain, pose)
            # print(A.shape)
            # print(xyz)
            # print(grasp_data['grasp_qpos'][:3])
            # xyz = joints_to_world(xyz, trans, quat)
            # print(xyz)
            import torch, open3d as o3d
            points = xyz[0]
            print(points.shape)
            points = torch.cat([points, torch.tensor(grasp_data['partial_points'], dtype=dtype, device=device)], dim=0)  # Add homogeneous coordinate
            points_np = points.detach().cpu().numpy()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            o3d.io.write_point_cloud("tmp/joint_positions.ply", pcd, write_ascii=True)
            break
        break