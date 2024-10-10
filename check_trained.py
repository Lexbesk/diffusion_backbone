"""
pcd_obs_env with:
1. object/background segmentation
2. object registration
3. goal sampling
4. reward calculation
"""

import numpy as np
from PIL import Image as im 
import os
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import numpy as np
from ctypes import * # convert float to uint32
# from matplotlib import pyplot as plt
import copy

# import rospy
# import rosbag
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation
import torch
np.set_printoptions(suppress=True,precision=4)

def get_transform( trans_7D):
    trans = trans_7D[0:3]
    quat = trans_7D[3:7]
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t

def visualize_pcd(pcd, lefts = None, rights = None, curr_pose = None):
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.1, center=(0., 0., 0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.) )
    if(lefts is not None):
        for left in lefts:
            left_mesh = copy.deepcopy(mesh).transform(left)
            vis.add_geometry(left_mesh)

    if(rights is not None):
        for right in rights:
            right_mesh = copy.deepcopy(mesh).transform(right)
            vis.add_geometry(right_mesh)

    if(curr_pose is not None):
        for pose in curr_pose:
            curr_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            curr_mesh.scale(0.2, center=(0., 0., 0.) )
            curr_mesh = curr_mesh.transform(pose)
            vis.add_geometry(curr_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def main():
    
    sample = np.load("./trained_model.npy" , allow_pickle = True)

    sample = sample.item()
    rgbs = sample["rgbs"]
    # print("rgb: ", rgb.shape)
    rgbs = rgbs.numpy()
    pcds = sample["pcds"].numpy()
    # actions = sample["action"].numpy()
    abs_actions = sample["abs_action"].numpy()    
    curr_gripper = sample['curr_gripper'].numpy()
    gts = sample["abs_sample_trajectory"].numpy()

    length = rgbs.shape[0]

    print("actions: ", abs_actions.shape)
    print("gts: ", gts.shape)
    for step_id in range(length):
        rgb = np.transpose(rgbs[step_id,0], (1, 2, 0) ).astype(float) # (0,1)
        xyz = np.transpose(pcds[step_id,0], (1, 2, 0) ).astype(float) # (0,1)

        pcd_rgb = rgb.reshape(-1, 3)
        pcd_xyz = xyz.reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector( pcd_rgb )
        pcd.points = o3d.utility.Vector3dVector( pcd_xyz )
        gt = gts[step_id]
        # action = actions[step_id]
        abs_action = abs_actions[step_id]
        left = []
        right = []
        
        curr_pose = [ get_transform(curr_gripper[step_id][0,0:7]), get_transform(curr_gripper[step_id][1,0:7]) ]
        
        print("gt: ", gt.shape)
        for action_idx in range(gt.shape[0]):
            
            left.append( get_transform( abs_action[action_idx,0,0:7]))
            right.append( get_transform( abs_action[action_idx,1,0:7]))

            diff = np.abs( abs_action[action_idx,1,0:3] - gt[action_idx,1,0:3])
            if( np.max(diff) > 0.005 ):
                print("step idx: ", action_idx)
                print("gt: ", gt[action_idx,1,0:3], " action: ", abs_action[action_idx,1,0:3])
                print("diff: ", diff)
        visualize_pcd(pcd, left, right, curr_pose)


if __name__ == "__main__":
    main()


    # [frame_ids],  # we use chunk and max_episode_length to index it
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (1, 8)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (1, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 8)
    # List of tensors
