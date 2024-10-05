"""Main script for trajectory optimization."""

import os
from pathlib import Path
import random
from typing import Optional

import numpy as np
np.set_printoptions(suppress=True,precision=4)

import torch

from datasets.dataset_mobaloha import MobileAlohaDataset
from diffuser_actor.trajectory_optimization.bimanual_diffuser_actor_mobaloha import BiManualDiffuserActor

from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)
from main_trajectory import Arguments as BaseArguments
from main_trajectory import TrainTester as BaseTrainTester
from main_trajectory import traj_collate_fn

from realworld_trajectory_mobaloha import Tester

# ROS2 related
import math
import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Joy, PointCloud2, PointField
from geometry_msgs.msg import PointStamped, TwistStamped, Quaternion, Vector3, TransformStamped
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32, Header, Float32MultiArray, MultiArrayDimension
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
        self.args = args

        ########################################################## 3dda model
        self.network = Tester(args)
        ##########################################################
        # self.file_dir = "/ws/data/mobile_aloha_debug/20240827_plate+0/ep4.npy"
        self.file_dir = "./trained_model.npy"
        self.sample = np.load( self.file_dir, allow_pickle=True)
        self.sample = self.sample.item()

        self.file_dir2 = "/ws/data/mobile_aloha_debug/20240827_plate+0/ep41.npy"
        self.episode = np.load( self.file_dir2, allow_pickle=True)
        # self.episode = self.episode.item()

        self.frame_idx = -1
        self.inference_action = []

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


        self.br = CvBridge()
        # Todo: use yaml files
        self.cam_extrinsic = self.get_transform( [-0.13913296, 0.053, 0.43643044, -0.63127772, 0.64917582, -0.31329509, 0.28619116])
        self.o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)

        self.resized_image_size = (256,256)
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

        timer_period = 0.01 #100hz
        self.timer = self.create_timer(timer_period, self.publish_tf)
        self.bimanual_ee_publisher = self.create_publisher(Float32MultiArray, "bimanual_ee_cmd", 1)

        self.bgr_sub = Subscriber(self, Image, "/camera_1/left_image")
        self.depth_sub = Subscriber(self, Image, "/camera_1/depth")
        self.left_hand_sub = Subscriber(self, JointState, "/follower_left/joint_states")
        self.right_hand_sub = Subscriber(self, JointState, "/follower_right/joint_states")

        # self.time_sync = ApproximateTimeSynchronizer([self.bgr_sub, self.depth_sub, self.left_hand_sub, self.right_hand_sub],
                                                    #  queue_size, max_delay)
        self.time_sync = ApproximateTimeSynchronizer([self.bgr_sub, self.depth_sub],
                                                     queue_size, max_delay)
        self.time_sync.registerCallback(self.SyncCallback)
        print("init finished !!!!!!!!!!!")

    
    def publish_tf(self):
        left_t = TransformStamped()
        right_t = TransformStamped()
        master_cam_t = TransformStamped()
        
        # # Read message content and assign it to
        # # corresponding tf variables
        ros_time = self.get_clock().now()
        left_t.header.stamp = ros_time.to_msg()
        right_t.header.stamp = ros_time.to_msg()
        master_cam_t.header.stamp = ros_time.to_msg()

        left_t.header.frame_id = 'world'
        left_t.child_frame_id = "follower_left/base_link"
        left_t.transform.translation.x = 0.0
        left_t.transform.translation.y = 0.315
        left_t.transform.translation.z = 0.0
        left_t.transform.rotation.x = 0.0
        left_t.transform.rotation.y = 0.0
        left_t.transform.rotation.z = 0.0
        left_t.transform.rotation.w = 1.0

        right_t.header.frame_id = 'world'
        right_t.child_frame_id = "follower_right/base_link"
        right_t.transform.translation.x = 0.0
        right_t.transform.translation.y = -0.315
        right_t.transform.translation.z = 0.0
        right_t.transform.rotation.x = 0.0
        right_t.transform.rotation.y = 0.0
        right_t.transform.rotation.z = 0.0
        right_t.transform.rotation.w = 1.0

        master_cam_t.header.frame_id = 'world'
        master_cam_t.child_frame_id = "master_camera"
        master_cam_t.transform.translation.x = -0.1393031
        master_cam_t.transform.translation.y = 0.0539
        master_cam_t.transform.translation.z = 0.43911375

        master_cam_t.transform.rotation.x = -0.61860094
        master_cam_t.transform.rotation.y = 0.66385477
        master_cam_t.transform.rotation.z = -0.31162288
        master_cam_t.transform.rotation.w = 0.2819945

        # cam_t.header.frame_id = 'master_camera'
        # cam_t.child_frame_id = "zed_left_camera_frame"
        # cam_t.transform.translation.x = 0.0
        # cam_t.transform.translation.y = 0.0
        # cam_t.transform.translation.z = 0.0
        # cam_t.transform.rotation.x = -0.4996018
        # cam_t.transform.rotation.y =  -0.4999998
        # cam_t.transform.rotation.z = 0.4999998
        # cam_t.transform.rotation.w = 0.5003982

        # # Send the transformation
        self.tf_broadcaster.sendTransform(left_t)
        self.tf_broadcaster.sendTransform(right_t)
        self.tf_broadcaster.sendTransform(master_cam_t)
        # self.tf_broadcaster.sendTransform(cam_t)
    def get_transform(self, transf_7D):
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

    def xyz_from_depth(self, depth_image, depth_intrinsic, depth_extrinsic, depth_scale=1000.):
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

    def image_process(self, bgr, depth, intrinsic_np, original_img_size, resized_intrinsic_np, resized_img_size):

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

    def xyz_rgb_validation(self, rgb, xyz):
        # verify xyz and depth value
        valid_pcd = o3d.geometry.PointCloud()
        xyz = xyz.reshape(-1,3)
        rgb = (rgb/255.0).reshape(-1,3)
        valid_pcd.points = o3d.utility.Vector3dVector( xyz )
        valid_pcd.colors = o3d.utility.Vector3dVector( rgb )
        # visualize_pcd(valid_pcd)


    # [frame_ids],  # we use chunk and max_episode_length to index it
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (1, 8)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (1, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 8)
    # List of tensors


    # def SyncCallback(self, bgr, depth, left_hand_joints, right_hand_joints):
    def SyncCallback(self, bgr, depth):
        self.frame_idx += 1
        length = self.sample["rgbs"].shape[0]

        print("in callback")
        if(self.frame_idx >= length):
            # self.episode.append(self.inference_action)
            np.save('debug_result', self.inference_action)
            return
        
        start = time.time()
 
        instr = torch.zeros((1, 53, 512))
        rgbs = self.sample["rgbs"][self.frame_idx]
        pcds = self.sample["pcds"][self.frame_idx]

        rgbs = rgbs[None, :,:,:,:]
        pcds = pcds[None, :,:,:,:]

        curr_gripper = self.sample['curr_gripper'][self.frame_idx]
        curr_gripper = curr_gripper[None, None, :,:]
        print("rgbs: ",rgbs.shape)
        print("curr_gripper: ", curr_gripper.shape)
        action = self.network.run( rgbs, pcds, curr_gripper, instr)
        checked_action = self.sample["abs_action"][self.frame_idx].numpy()
        # print("diff", np.abs(checked_action - action[1:]))

        obs = self.episode[1][ self.frame_idx ].numpy()
        rgbs = torch.from_numpy(obs[0:,0])
        pcds = torch.from_numpy(obs[0:,1])
        rgbs = rgbs[None, :,:,:,:]
        pcds = pcds[None, :,:,:,:]
        curr_gripper = self.episode[4][ self.frame_idx ]
        curr_gripper = curr_gripper[None, None, :,:]
        action2 = self.network.run( rgbs, pcds, curr_gripper, instr)
        print("diff", np.abs(action - action2))

        current_data = {}
        current_data['rgb'] = rgbs
        current_data['xyz'] = pcds
        current_data['curr_gripper'] = curr_gripper
        current_data['action'] = action
        current_data['gt'] = self.episode[5][ self.frame_idx ].numpy()       
        np.save('step_{}'.format(self.frame_idx), current_data, allow_pickle = True)
        # self.step_idx += 1

        end = time.time()
        print("3dda took: ", end - start)


class Arguments(BaseArguments):
    instructions: Optional[Path] = None
       

def main(main_args):
   
    rclpy.init()
    node = bi_3dda_node( main_args )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)

    # [frame_ids],  # we use chunk and max_episode_length to index it
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (1, 8)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (1, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 8)
    # List of tensors
