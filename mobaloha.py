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
from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Path

from geometry_msgs.msg import PoseStamped, PointStamped, TwistStamped, Quaternion, Vector3, TransformStamped

from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32, Header, Float32MultiArray, MultiArrayDimension

from sklearn.neighbors import NearestNeighbors

import numpy as np 
import time


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

from math_tools import *

# from utils.ros2_o3d_utils import *


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

        self.left_hand_frame = "follower_left/ee_gripper_link"
        self.right_hand_frame = "follower_right/ee_gripper_link"

        self.left_hand_gripper_frames = ["follower_left/left_finger_link", "follower_left/right_finger_link"]
        self.right_hand_gripper_frames = ["follower_right/left_finger_link", "follower_right/right_finger_link"]

        self.left_base_frame = "follower_left/base_link"
        self.right_base_frame = "follower_right/base_link"

        # self.left_base_frame = "world"
        # self.right_base_frame = "world"

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

        self.bound_box = np.array( [ [0.05, 0.65], [ -0.5 , 0.5], [ -0.1 , 0.6] ] )
        # self.left_bias = self.get_transform( [ -0.11, 0.015, 0.010 ,0., 0., 0., 1.] )
        # self.right_bias = self.get_transform( [-0.06, 0.005, -0.005, 0., 0., 0., 1.] )
        
        self.left_bias = self.get_transform(   [ -0.01, 0.365, -0.0 ,0., 0.,0.02617695, 0.99965732] )
        self.left_tip_bias = self.get_transform( [-0.028, 0.01, 0.01,      0., 0., 0., 1.] ) @ self.get_transform([0.087, 0, 0., 0., 0., 0., 1.] )

        self.right_bias = self.get_transform(   [ 0.01, -0.315, 0.00, 0., 0., 0., 1.0] )
        self.right_tip_bias = self.get_transform( [-0.035, 0.01, -0.008,      0., 0., 0., 1.] ) @ self.get_transform([0.087, 0, 0., 0., 0., 0., 1.] )

        self.last_data_time = time.time()

        max_delay = 0.01 #10ms
        self.time_diff = 0.05
        self.tf_broadcaster = TransformBroadcaster(self)

        self.step_idx = 0        

        timer_period = 0.01 #100hz
        # self.timer = self.create_timer(timer_period, self.publish_tf)

        self.bimanual_ee_publisher = self.create_publisher(Float32MultiArray, "bimanual_ee_cmd", 1)

        self.left_hand_ee_publisher = self.create_publisher(Path, "left_ee_goal", 1)
        self.right_hand_ee_publisher = self.create_publisher(Path, "right_ee_goal", 1)

        self.bgr_sub = Subscriber(self, Image, "/camera_1/left_image")
        self.depth_sub = Subscriber(self, Image, "/camera_1/depth")

        self.left_hand_joints = None
        self.right_hand_joints = None

        self.left_hand_sub = self.create_subscription(JointState, "/follower_left/joint_states",self.left_hand_callback,1)
        self.right_hand_sub = self.create_subscription(JointState, "/follower_right/joint_states",self.right_hand_callback,1)

        self.low_level_finsihed = True
        self.controller_sub = self.create_subscription(Bool, "controller_finished",self.controller_callback,1)

        # self.time_sync = ApproximateTimeSynchronizer([self.bgr_sub, self.depth_sub, self.left_hand_sub, self.right_hand_sub],
                                                    #  queue_size, max_delay)
        queue_size = 1
        self.time_sync = ApproximateTimeSynchronizer([self.bgr_sub, self.depth_sub],
                                                     queue_size, max_delay)
        self.time_sync.registerCallback(self.SyncCallback)

        print("init finished !!!!!!!!!!!")

    def left_hand_callback(self, JointState_msg):
        self.left_hand_joints = JointState_msg

    def right_hand_callback(self, JointState_msg):
        self.right_hand_joints = JointState_msg

    def controller_callback(self, bool_msg):
        self.low_level_finsihed = True

    def print_action(self, action):
        action = action.reshape(-1, 2, 8)
        left_path_np = action[:, 0, :]
        right_path_np = action[:, 1, :]
        start = time.time()

        header = Header()
        header.frame_id = "world"
        ros_time = self.get_clock().now()
        header.stamp = ros_time.to_msg()

        left_path = Path()
        left_path.header = header
        for i in range(action.shape[0]):
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = float(action[i,0,0])
            pose.pose.position.y = float(action[i,0,1])
            pose.pose.position.z = float(action[i,0,2])
            pose.pose.orientation.x = float(action[i,0,3])
            pose.pose.orientation.y = float(action[i,0,4])
            pose.pose.orientation.z = float(action[i,0,5])
            pose.pose.orientation.w = float(action[i,0,6])
            left_path.poses.append(pose)

        right_path = Path()
        right_path.header = header
        for i in range(action.shape[0]):
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = float(action[i,1,0])
            pose.pose.position.y = float(action[i,1,1])
            pose.pose.position.z = float(action[i,1,2])
            pose.pose.orientation.x = float(action[i,1,3])
            pose.pose.orientation.y = float(action[i,1,4])
            pose.pose.orientation.z = float(action[i,1,5])
            pose.pose.orientation.w = float(action[i,1,6])
            right_path.poses.append(pose)

        end = time.time()
        self.left_hand_ee_publisher.publish(left_path)
        self.right_hand_ee_publisher.publish(right_path)
        print( "took: ", end - start)

    def get_transform(self, transf_7D):
        trans = transf_7D[0:3]
        quat = transf_7D[3:7]
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
        t[:3, 3] = trans
        return t

    def get_7D_transform(self, transf):
        trans = transf[0:3,3]
        trans = trans.reshape(3)
        quat = Rotation.from_matrix( transf[0:3,0:3] ).as_quat()
        quat = quat.reshape(4)
        return np.concatenate( [trans, quat])

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

        xyz = np.concatenate([xyz, ones], axis=2)
        xyz =  xyz @ np.transpose( depth_extrinsic)
        xyz = xyz[:,:,0:3]
        return xyz

    def denoise(self, rgb, xyz, debug = False):

        start = time.time()
        x = xyz[:,:,0]
        z = xyz[:,:,2]
        # print("minz: ", np.min(z))
        valid_idx = np.where( (z > -0.08) & (x > 0.05))
        # print("points: ", len( valid_idx[0]))
        if(len( valid_idx[0]) < 10):
            return rgb, xyz

        test_xyz = xyz [ valid_idx ]
        X = test_xyz.reshape(-1,3)

        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = distances[:,2]
        invalid_idx = np.where(distances > 0.01)
        
        if( len(valid_idx[0]) < 10):
            return rgb, xyz

        xs = valid_idx[0]
        ys = valid_idx[1]

        xs = xs[invalid_idx]
        ys = ys[invalid_idx]    
        rgb[xs,ys] = np.array([0., 0., 0.])
        xyz[xs,ys] = np.array([0., 0., 0.])
        end = time.time()
        if(debug):
            print("time cost: ", end - start)
        
        return rgb, xyz


    def image_process(self, bgr, depth, intrinsic_np, original_img_size, resized_intrinsic_np, resized_img_size):
        
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

        return processed_rgb, processed_depth

    def cropping(self, rgb, xyz, bound_box, label = None, return_image = True):

        x = xyz[:,:,0]
        y = xyz[:,:,1]
        z = xyz[:,:,2]

        # print("bound_box: ", bound_box)
        valid_idx = np.where( (x>=bound_box[0][0]) & (x <=bound_box[0][1]) & (y>=bound_box[1][0]) & (y<=bound_box[1][1]) & (z>=bound_box[2][0]) & (z<=bound_box[2][1]) )

        if(return_image):

            cropped_rgb = np.zeros(rgb.shape)
            cropped_xyz = np.zeros(xyz.shape) 
            cropped_rgb[valid_idx] = rgb[valid_idx]
            cropped_xyz[valid_idx] = xyz[valid_idx]

            return cropped_rgb, cropped_xyz

        valid_xyz = xyz[valid_idx]
        valid_rgb = rgb[valid_idx]
        valid_label = None
        if(label is not None):
            valid_label = label[valid_idx]
                
        valid_pcd = o3d.geometry.PointCloud()
        valid_pcd.points = o3d.utility.Vector3dVector( valid_xyz)
        if(np.max(valid_rgb) > 1.0):
            valid_pcd.colors = o3d.utility.Vector3dVector( valid_rgb/255.0 )
        else:
            valid_pcd.colors = o3d.utility.Vector3dVector( valid_rgb )
        return valid_xyz, valid_rgb, valid_label, valid_pcd

    def xyz_rgb_validation(self, rgb, xyz):
        # verify xyz and depth value
        valid_pcd = o3d.geometry.PointCloud()
        xyz = xyz.reshape(-1,3)
        rgb = (rgb/255.0).reshape(-1,3)
        valid_pcd.points = o3d.utility.Vector3dVector( xyz )
        valid_pcd.colors = o3d.utility.Vector3dVector( rgb )
        # visualize_pcd(valid_pcd)

    # def SyncCallback(self, bgr, depth, left_hand_joints, right_hand_joints):
    def SyncCallback(self, bgr, depth):
        # print("in call back")
        if(self.low_level_finsihed == False):
            return
        if(self.left_hand_joints is None):
            return 
        if(self.right_hand_joints is None):
            return
        print("now: ", time.time())
        print("joint time: ", self.left_hand_joints.header.stamp)
        print("joint time: ", self.right_hand_joints.header.stamp)
        print("image_time: ", bgr.header.stamp) 
        
        bgr_np = np.array(self.br.imgmsg_to_cv2(bgr))[:,:,:3]
        depth_np = np.array(self.br.imgmsg_to_cv2(depth, desired_encoding="mono16"))

        rgb, depth = self.image_process(bgr_np, 
                                        depth_np, 
                                        self.o3d_intrinsic.intrinsic_matrix, 
                                        self.original_image_size, 
                                        self.resized_intrinsic_o3d.intrinsic_matrix,
                                        self.resized_image_size 
                                        )
        print("rgb: ", rgb.shape)
        # print("rgb: ", type(rgb))
   
        # print("image_time: ", bgr.header.stamp)
        # print("left hand pose: ", self.left_hand_transform_7D)

        # print("right hand pose: ", self.right_hand_transform_7D)

        im_color = o3d.geometry.Image(rgb)
        im_depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth, depth_scale=1000, depth_trunc=2000, convert_rgb_to_intensity=False)

        all_valid_resized_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.resized_intrinsic_o3d,
        )
        all_valid_resized_pcd.transform( self.cam_extrinsic )

        # visualize_pcd(all_valid_resized_pcd)
        xyz = self.xyz_from_depth(depth, self.resized_intrinsic_o3d.intrinsic_matrix, self.cam_extrinsic )

        # sample = np.load("/ws/bimanual/39/39_bound/step_0.npy", allow_pickle=True)
        # sample = sample.item()

        # sample_rgb  = sample["rgb"]
        # sample_rgb = sample_rgb[0,0].numpy()*255
        # sample_rgb = np.transpose(sample_rgb, (1, 2, 0) ).astype(float) # (0,1)

        # rgb[105:150, 70:130] = sample_rgb[105:150, 70:130]

        # sample_xyz  = sample["xyz"]
        # sample_xyz = sample_xyz[0,0].numpy()
        # sample_xyz = np.transpose(sample_xyz, (1, 2, 0) ).astype(float) # (0,1)

        # xyz[105:150, 70:130] = sample_xyz[105:150, 70:130]
        # # xyz[105:150, 70:130, 2] = -0.05
        
        cropped_rgb, cropped_xyz = self.cropping( rgb, xyz, self.bound_box)
        filtered_rgb, filtered_xyz = self.denoise(cropped_rgb, cropped_xyz, debug= True)

        if( len( np.where( np.isnan(xyz))[0] ) > 0 ):
            print(np.where( np.isnan(xyz)))
            print(" x y z has invalid point !!!!!")
            print(" x y z has invalid point !!!!!")
            print(" x y z has invalid point !!!!!")
            raise

        resized_img_data = np.transpose(filtered_rgb, (2, 0, 1) ).astype(float)
        resized_img_data = (resized_img_data / 255.0 ).astype(float)
        resized_xyz = np.transpose(filtered_xyz, (2, 0, 1) ).astype(float)
        # print("resized_xyz: ", resized_xyz.shape)
        n_cam = 1
        obs = np.zeros( (1, n_cam, 2, 3, 256, 256) ).astype(float)
        obs[0][0][0] = resized_img_data
        obs[0][0][1] = resized_xyz
        # print("rgb_shape: ", obs[0:,0:,0].shape)
        
        left_pos = np.array(self.left_hand_joints.position) 
        right_pos = np.array(self.right_hand_joints.position) 

        left_transform = self.left_bias @ FwdKin(left_pos[0:6])  @ self.left_tip_bias
        left_hand_transform_7D = self.get_7D_transform( left_transform )

        right_transform = self.right_bias @ FwdKin(right_pos[0:6]) @ self.right_tip_bias
        right_hand_transform_7D = self.get_7D_transform( right_transform )

        # trans = FwdKin(left_pos[0:6])
        # trans[1,3] += 0.315
        # left_fk = self.get_7D_transform( trans)
        # print("left_fk: ", left_fk)
        # print("diff: ", left_fk - self.left_hand_transform_7D)
        left_min_joint = 0.62
        left_max_joint = 1.62

        right_min_joint = 0.62
        right_max_joint = 1.62

        curr_gripper_np = np.zeros( (1,1,2,8)).astype(float)
        curr_gripper_np[0,0,0,0:7] = left_hand_transform_7D
        curr_gripper_np[0,0,0,7] = (left_pos[6] - left_min_joint ) / (left_max_joint -  left_min_joint)
        curr_gripper_np[0,0,0,7] = 0 if curr_gripper_np[0,0,0,7] < 0.1 else 1

        curr_gripper_np[0,0,1,0:7] = right_hand_transform_7D
        curr_gripper_np[0,0,1,7] = (right_pos[6] - right_min_joint ) / (right_max_joint -  right_min_joint)
        curr_gripper_np[0,0,1,7] = 0 if curr_gripper_np[0,0,1,7] < 0.1 else 1

        start = time.time()
 
        # instr = torch.zeros((1, 53, 512))
        task = self.args.current_task
        rgbs = torch.from_numpy(obs[0:,0:,0])
        pcds = torch.from_numpy(obs[0:,0:,1])
        curr_gripper = torch.from_numpy(curr_gripper_np)

        action = self.network.run( rgbs, pcds, curr_gripper, task)
        end = time.time()
        print("3dda took: ", end - start)
        # print("action: ", action.shape)
        # print("action: ", action[0:5, :,:])
        # action[:,0,1] -= 0.02
        # action[:,1,1] += 0.02
        self.print_action(action)

        current_data = {}
        current_data['rgb'] = rgbs
        current_data['xyz'] = pcds
        current_data['curr_gripper'] = curr_gripper
        current_data['left_joints'] = left_pos
        current_data['right_joints'] = right_pos
        current_data['action'] = action
        np.save('step_{}'.format(self.step_idx), current_data, allow_pickle = True)
        self.step_idx += 1


        array_msg = Float32MultiArray()
        
        array_msg.layout.dim.append(MultiArrayDimension())
        array_msg.layout.dim.append(MultiArrayDimension())
        array_msg.layout.dim.append(MultiArrayDimension())

        array_msg.layout.dim[0].label = "steps"
        array_msg.layout.dim[1].label = "hands"
        array_msg.layout.dim[2].label = "pose"

        array_msg.layout.dim[0].size = action.shape[0]
        array_msg.layout.dim[1].size = action.shape[1]
        array_msg.layout.dim[1].size = action.shape[2]
        array_msg.layout.data_offset = 0

        array_msg.data = action.reshape([1, -1])[0].tolist();
        # array_msg.layout.dim[0].stride = width*height
        # array_msg.layout.dim[1].stride = width
        self.low_level_finsihed = False
        self.bimanual_ee_publisher.publish(array_msg)

        print()
        # self.left_hand_transform_7D
        # self.right_hand_transform_7D

class Arguments(BaseArguments):
    instructions: Optional[str] = None
       

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

