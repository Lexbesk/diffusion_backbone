"""Main script for trajectory optimization."""

import os
from pathlib import Path
import random
from typing import Optional

import numpy as np
import torch

from datasets.dataset_mobaloha import MobileAlohaDataset
from diffuser_actor.trajectory_optimization.bimanual_diffuser_actor_mobaloha import BiManualDiffuserActor

from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)
from main_trajectory import Arguments as BaseArguments
from main_trajectory import TrainTester as BaseTrainTester
from main_trajectory import traj_collate_fn

# ROS2 related
import math
import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Joy, PointCloud2, PointField
from geometry_msgs.msg import PointStamped, TwistStamped, Quaternion, Vector3, TransformStamped
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32, Header
import numpy as np 
import time

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2 
from rclpy.qos import QoSProfile
from rclpy.clock import Clock
from message_filters import Subscriber, ApproximateTimeSynchronizer

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
import argparse

import open3d as o3d
import torch
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

####################################################################
#
# everything is in robot frame, (middle point of two robot arm base)
#
####################################################################
class bi_3dda_node(Node):
    def __init__(self, args):
        super().__init__('bi_3dda_node')

        self.left_hand_frame = "follower_left/ee_gripper_link"
        self.right_hand_frame = "follower_right/ee_gripper_link"

        self.left_hand_gripper_frames = ["follower_left/left_finger_link", "follower_left/right_finger_link"]
        self.right_hand_gripper_frames = ["follower_right/left_finger_link", "follower_right/right_finger_link"]
        # self.left_base_frame = "follower_left/base_link"
        # self.right_base_frame = "follower_right/base_link"
        self.left_base_frame = "world"
        self.right_base_frame = "world"

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.left_hand_transform = TransformStamped()
        self.right_hand_transform = TransformStamped()

        self.lh_gripper_left_transform = TransformStamped()
        self.lh_gripper_right_transform = TransformStamped()
        self.rh_gripper_left_transform = TransformStamped()
        self.rh_gripper_right_transform = TransformStamped()

        # Todo: use yaml files
        self.cam_extrinsic = self.get_transform( [-0.13913296, 0.053, 0.43643044, -0.63127772, 0.64917582, -0.31329509, 0.28619116])
        self.o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)

        self.resized_img_size = (256,256)
        self.original_image_size = (1080, 1920) #(h,)
        fxfy = 256.0
        self.resized_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(256, 256, fxfy, fxfy, 128.0, 128.0)
        self.resized_intrinsic_np = np.array([
            [fxfy, 0., 128.0],
            [0. ,fxfy,  128.0],
            [0., 0., 1.0]
        ])

        self.last_data_time = time.time()

        queue_size = 1000
        max_delay = 0.01 #10ms
        self.time_diff = 0.05
        self.tf_broadcaster = TransformBroadcaster(self)

        self.bgr_sub = Subscriber(self, Image, "/camera_1/left_image")
        self.depth_sub = Subscriber(self, Image, "/camera_1/depth")
        self.left_hand_sub = Subscriber(self, JointState, "/follower_left/joint_states")
        self.right_hand_sub = Subscriber(self, JointState, "/follower_right/joint_states")

        self.time_sync = ApproximateTimeSynchronizer([self.bgr_sub, self.depth_sub, self.left_hand_sub, self.right_hand_sub],
                                                     queue_size, max_delay)
        self.time_sync.registerCallback(self.SyncCallback)

        timer_period = 0.01 #100hz
        self.timer = self.create_timer(timer_period, self.update_hand_transform)

    def get_transform( transf_7D):
        trans = transf_7D[0:3]
        quat = transf_7D[3:7]
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
        t[:3, 3] = trans
        return t

    def transform_to_numpy(self, ros_transformation):
        x = ros_transformation.transform.translation.x
        y = ros_transformation.transform.translation.y
        z = ros_transformation.transform.translation.z
        
        qx = ros_transformation.transform.rotation.x
        qy = ros_transformation.transform.rotation.y
        qz = ros_transformation.transform.rotation.z
        qw = ros_transformation.transform.rotation.w

        return np.array( [x, y, z, qx, qy, qz, qw] )

    def xyz_from_depth(depth_image, depth_intrinsic, depth_extrinsic, depth_scale=1000.):
        # Return X, Y, Z coordinates from a depth map.
        # This mimics OpenCV cv2.rgbd.depthTo3d() function
        fx = depth_intrinsic[0, 0]
        fy = depth_intrinsic[1, 1]
        cx = depth_intrinsic[0, 2]
        cy = depth_intrinsic[1, 2]
        # Construct (y, x) array with pixel coordinates
        y, x = np.meshgrid(range(depth_image.shape[0]), range(depth_image.shape[1]), sparse=False, indexing='ij')

        X = (x - cx) * depth_image / (fx * depth_scale)
        Y = (y - cy) * depth_image / (fy * depth_scale)
        ones = np.ones( ( depth_image.shape[0], depth_image.shape[1], 1) )
        xyz = np.stack([X, Y, depth_image / depth_scale], axis=2)
        xyz[depth_image == 0] = 0.0

        # print("xyz: ", xyz.shape)
        # print("ones: ", ones.shape)
        # print("depth_extrinsic: ", depth_extrinsic.shape)
        xyz = np.concatenate([xyz, ones], axis=2)
        xyz =  xyz @ np.transpose( depth_extrinsic)
        xyz = xyz[:,:,0:3]
        return xyz

    def image_process( bgr, depth, intrinsic_np, original_img_size, resized_intrinsic_np, resized_img_size):

        # print("bgr: ", bgr.shape)
        # print("depth: ", depth.shape)
        # print("intrinsic_np: ", intrinsic_np)
        # print("resized_intrinsic_np: ", resized_intrinsic_np)
        
        cx = intrinsic_np[0,2]
        cy = intrinsic_np[1,2]

        fx_factor = resized_intrinsic_np[0,0] / intrinsic_np[0,0]
        fy_factor = resized_intrinsic_np[1,1] / intrinsic_np[1,1]

        raw_fx = resized_intrinsic_np[0,0] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
        raw_fy = resized_intrinsic_np[1,1] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]
        raw_cx = resized_intrinsic_np[0,2] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
        raw_cy = resized_intrinsic_np[1,2] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]

        width = resized_img_size[0] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
        height = resized_img_size[0] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]
        
        half_width = int( width / 2.0 )
        half_height = int( height / 2.0 )

        cropped_bgr = bgr[round(cy-half_height) : round(cy + half_height), round(cx - half_width) : round(cx + half_width), :]
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.resize(cropped_rgb, resized_img_size)

        cropped_depth = depth[round(cy-half_height) : round(cy + half_height), round(cx - half_width) : round(cx + half_width)]
        processed_depth = cv2.resize(cropped_depth, resized_img_size, interpolation =cv2.INTER_NEAREST)

        # print("processed_rgb: ", processed_rgb.shape)
        # print("width: ", width)
        # print("height: ", height)
        # print("raw_fx: ", raw_fx)
        # print("raw_fy: ", raw_fy)
        # print("raw_cx: ", raw_cx)
        # print("raw_cy: ", raw_cy)

        return processed_rgb, processed_depth

    def SyncCallback(self, bgr, depth, left_hand_joints, right_hand_joints):

        try:
            self.left_hand_transform = self.tf_buffer.lookup_transform(
                    self.left_base_frame,
                    self.left_hand_frame,
                    bgr.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=0.01)
                )
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.left_base_frame} to {self.left_hand_frame}: {ex}'
            )
            return
        self.left_hand_transform_7D = self.transform_to_numpy( self.left_hand_transform )
        try:
            self.right_hand_transform = self.tf_buffer.lookup_transform(
                    self.right_base_frame,
                    self.right_hand_frame,
                    bgr.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=0.01)
                )
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.right_base_frame} to {self.right_hand_frame}: {ex}'
            )
            return
        self.right_hand_transform_7D = self.transform_to_numpy( self.right_hand_transform )
        
        bgr_np = np.array(self.br.imgmsg_to_cv2(bgr))[:,:,:3]
        depth_np = np.array(self.br.imgmsg_to_cv2(depth, desired_encoding="mono16"))

        rgb, depth = self.image_process(bgr, 
                                        depth, 
                                        self.o3d_intrinsic.intrinsic_matrix, 
                                        self.original_image_size, 
                                        self.resized_intrinsic_o3d.intrinsic_matrix,
                                        self.resized_image_size 
                                        )
        # print("rgb: ", type(rgb))
        im_color = o3d.geometry.Image(rgb)
        im_depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth, depth_scale=1000, depth_trunc=2000, convert_rgb_to_intensity=False)
        
        # original_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #         rgbd,
        #         o3d_intrinsic
        #         # resized_intrinsic
        #     )
        # original_pcd = original_pcd.transform(cam_extrinsic)
        # xyz = np.array(original_pcd.points)
        # rgb = np.array(original_pcd.colors)
        # valid_xyz, valid_rgb, valid_label, cropped_pcd = cropping( xyz, rgb, bound_box )

        all_valid_resized_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.resized_intrinsic_o3d,
        )
        all_valid_resized_pcd.transform( self.cam_extrinsic )

        # visualize_pcd(all_valid_resized_pcd)
        xyz = self.xyz_from_depth(depth, self.resized_intrinsic_o3d.intrinsic_matrix, self.cam_extrinsic )

        if( len( np.where( np.isnan(xyz))[0] ) >0 ):
            print(np.where( np.isnan(xyz)))
            print(" x y z has invalid point !!!!!")
            print(" x y z has invalid point !!!!!")
            print(" x y z has invalid point !!!!!")
            raise

        # xyz_rgb_validation(rgb, xyz)

        resized_img_data = np.transpose(rgb, (2, 0, 1) ).astype(float)
        resized_img_data = resized_img_data / 255.0
        # print("resized_img_data: ", resized_img_data.shape)
        resized_xyz = np.transpose(xyz, (2, 0, 1) ).astype(float)
        # print("resized_xyz: ", resized_xyz.shape)
        n_cam = 1
        obs = np.zeros( (n_cam, 2, 3, 256, 256) )
        obs[0][0] = resized_img_data
        obs[0][1] = resized_xyz

        self.left_hand_transform_7D
        self.right_hand_transform_7D
        
        # current_state["left_pos"] = np.array(left_hand_joints.position) 
        # current_state["left_vel"] = np.array(left_hand_joints.velocity) 
        # current_state["right_pos"] = np.array(right_hand_joints.position) 
        # current_state["right_vel"] = np.array(right_hand_joints.velocity)
        # history


        
       

def main(main_args):
   
    rclpy.init(args=None)
    node = bi_3dda_node(main_args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, default="/data/images/human1.png", required=True)
    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--conv-mode", type=str, default=None)
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--max-new-tokens", type=int, default=512)
    # parser.add_argument("--load-8bit", action="store_true")
    # parser.add_argument("--load-4bit", action="store_true")
    # parser.add_argument("--debug", action="store_true")

    # parser.add_argument("--ros_namespace", type=str, default="" )   
    args = parser.parse_args()
    main(args)

