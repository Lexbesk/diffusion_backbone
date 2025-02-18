import os
import glob
import random
from typing import List

import open3d
import traceback
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from rlbench.demo import Demo
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from online_evaluation_rlbench.get_stored_demos import get_stored_demos

# ??? pick up a plate, put item in drawer is missing from this.
# ??? bimanual_sweep_to_dustpan and coordinated_lift_tray are interchanged in the download links, tell Markus
ALL_RLBENCH_TASKS = [
    'coordinated_push_box', 'coordinated_lift_ball', 'dual_push_buttons', 'bimanual_pick_plate', 
    'coordinated_put_bottle_in_fridge', 'handover_item', 'bimanual_pick_laptop', 'bimanual_straighten_rope', 
    'coordinated_lift_tray', 'bimanual_sweep_to_dustpan', 'handover_item_easy',
    'coordinated_take_tray_out_of_oven',
    'coordinated_put_item_in_drawer',
]
TASK_TO_ID = {task: i for i, task in enumerate(ALL_RLBENCH_TASKS)}


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.bimanual_tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class Mover:

    def __init__(self, task, disabled=False, max_tries=1):
        self._task = task
        self._last_action = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action, collision_checking=False):
        # if self._disabled:
        #     return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[:, 7] = self._last_action[:, 7].copy()

        images = []
        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            # We need to fix this, Peract2 takes (right, left) action, but we 
            # process it in terms of (left ,right) action
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            obs, reward, terminate = self._task.step(action_collision)

            l_pos = obs.left.gripper_pose[:3]
            r_pos = obs.right.gripper_pose[:3]
            l_dist_pos = np.sqrt(np.square(target[0, :3] - l_pos).sum())
            r_dist_pos = np.sqrt(np.square(target[1, :3] - r_pos).sum())
            criteria = (l_dist_pos < 5e-3, r_dist_pos < 5e-3)

            if all(criteria) or reward == 1:
                break

            print(
                f"Too far away (l_pos: {l_dist_pos:.3f}, r_pos: {r_dist_pos:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and (action[0, 7] != self._last_action[0, 7] or action[1, 7] != self._last_action[1, 7])
        ):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            # We need to fix this, Peract2 takes (right, left) action, but we 
            # process it in terms of (left ,right) action
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            obs, reward, terminate = self._task.step(action_collision)

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        return obs, reward, terminate, images


class Actioner:

    def __init__(
        self,
        policy=None,
        instructions=None,
        apply_cameras=("over_shoulder_left", "over_shoulder_right", "wrist_left", "wrist_right" "front")
    ):
        self._policy = policy
        self._instructions = instructions
        self._apply_cameras = apply_cameras

        self._actions = {}
        self._instr = None
        self._task_str = None

        self._policy.eval()

    def load_episode(self, task_str, variation):
        self._task_str = task_str
        instructions = list(self._instructions[task_str][variation])
        self._instr = random.choice(instructions).unsqueeze(0)
        self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def predict(self, rgbs, pcds, gripper,
                interpolation_length=None):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            interpolation_length: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        output = {"action": None}

        rgbs = rgbs / 2 + 0.5  # in [0, 1]

        if self._instr is None:
            raise ValueError()

        self._instr = self._instr.to(rgbs.device)
        self._task_id = self._task_id.to(rgbs.device)

        gripper = gripper.unflatten(-1, (2, -1))

        # Predict trajectory
        fake_traj = torch.full(
            [1, interpolation_length - 1, gripper.shape[-1]], 0
        ).to(rgbs.device)
        traj_mask = torch.full(
            [1, interpolation_length - 1], False
        ).to(rgbs.device)
        output["action"] = self._policy(
            fake_traj,
            traj_mask,
            rgbs,
            pcds,
            self._instr,
            gripper[..., :7],
            run_inference=True
        )

        return output

    @property
    def device(self):
        return next(self._policy.parameters()).device


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("over_shoulder_left", "over_shoulder_right", "wrist_left", "wrist_right", "front"),
        collision_checking=False
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        # ??? Change this to BiManual
        # 1. Select a control mode for the arm (arm_action_mode) - right now it is running a planner to reach a pose
        # 2. Select a control mode for the gripper (gripper_action_mode) - it is discrete open or close here
        # These are given to an overall action mode that only moves the gripper after moving the arm.
        self.action_mode = BimanualMoveArmThenGripper(
            arm_action_mode=BimanualEndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=BimanualDiscrete()
        )
        # The right and left actions are appended together like from :7 and 7:

        # Define a headless environment (this creates the rl bench scene) with the given action mode.
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless, robot_setup="dual_panda"
        )
        self.image_size = image_size

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = obs.perception_data["{}_rgb".format(cam)]
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = obs.perception_data["{}_depth".format(cam)]
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = obs.perception_data["{}_point_cloud".format(cam)]
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.left.gripper_pose,
                                 [obs.left.gripper_open],
                                 obs.right.gripper_pose,
                                 [obs.right.gripper_open]])

        # action is an array of length 16 = (7+1)*2; 7 is pose and 1 is open flag, 2 is for 2 arms

        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        return rgb, pcd, gripper

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False
        )
        return demos

    def evaluate_task_on_multiple_variations(
        self,
        task_str: str,
        max_steps: int,
        num_variations: int,  # -1 means all variations
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        interpolation_length=100,
        num_history=1
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = task.variation_count()

        if num_variations > 0:
            task_variations = np.minimum(num_variations, task_variations)
            task_variations = range(task_variations)
        else:
            task_variations = glob.glob(os.path.join(self.data_path, task_str, "variation*"))
            task_variations = [int(n.split('/')[-1].replace('variation', '')) for n in task_variations]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in task_variations:
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    num_demos=num_demos // len(task_variations) + 1,
                    actioner=actioner,
                    max_tries=max_tries,
                    verbose=verbose,
                    interpolation_length=interpolation_length,
                    num_history=num_history
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str: str,
        task: TaskEnvironment,
        max_steps: int,
        variation: int,
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        interpolation_length=50,
        num_history=0
    ):
        device = actioner.device

        success_rate = 0
        num_valid_demos = 0
        total_reward = 0

        for demo_id in range(num_demos):
            if verbose:
                print()
                print(f"Starting demo {demo_id}")

            try:
                demo = get_stored_demos(
                    amount=1,
                    image_paths=False,
                    dataset_root=self.data_path,
                    variation_number=variation,
                    task_name=task_str,
                    obs_config=self.obs_config,
                    random_selection=False,
                    from_episode_number=demo_id
                )[0]
                num_valid_demos += 1
            except:
                continue

            grippers = torch.Tensor([]).to(device)

            # descriptions, obs = task.reset()
            descriptions, obs = task.reset_to_demo(demo)

            actioner.load_episode(task_str, variation)

            move = Mover(task, max_tries=max_tries)
            reward = 0.0
            max_reward = 0.0

            for step_id in range(max_steps):

                # Fetch the current observation, and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgbs_input = rgb.to(device)
                pcds_input = pcd.to(device)
                gripper = gripper.to(device)

                grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

                # Prepare proprioception history
                if num_history < 1:
                    gripper_input = grippers[:, -1]
                else:
                    gripper_input = grippers[:, -num_history:]
                    npad = num_history - gripper_input.shape[1]
                    gripper_input = F.pad(
                        gripper_input, (0, 0, npad, 0), mode='replicate'
                    )

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    interpolation_length=interpolation_length
                )

                if verbose:
                    print(f"Step {step_id}")

                terminate = True

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    actions = output["action"][-1].cpu().numpy()
                    actions[..., -1] = actions[..., -1].round()

                    # execute
                    for action in actions:
                        collision_checking = self._collision_checking(task_str, step_id)
                        obs, reward, terminate, _ = move(action, collision_checking=collision_checking)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                    if terminate:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0

            total_reward += max_reward
            if reward == 0:
                step_id += 1

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", num_valid_demos,
            )

        # Compensate for failed demos
        if num_valid_demos == 0:
            assert success_rate == 0
            valid = False
        else:
            valid = True

        return success_rate, valid, num_valid_demos

    def _collision_checking(self, task_str, step_id):
        """Collision checking for planner."""
        collision_checking = False
        return collision_checking

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        
        # Define a config for an unused camera with all rgb, depth and point cloud applications as False.
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        
        # Define a config for a used camera with the given image size and flags
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs,
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
        # For each camera name(str), assign the used camera config class to it (defined above)
        cameras = {}
        for name in camera_names:
            cameras[name] = used_cams

        obs_config = ObservationConfig(
            camera_configs = cameras,
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )

        return obs_config


# Identify way-point in each RLBench Demo by checking where the manipulator stops
def _left_is_stopped(demo, i, obs, stopped_buffer, delta):

    next_is_not_final = i == (len(demo) - 2) # Check if next step is not the final one
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )

    # Check if the gripper state hasn't changed from (i-2) to (i+1)
    gripper_state_no_change = i < (len(demo) - 2) and (     # Not the second last or last step
        obs.left.gripper_open == demo[i + 1].left.gripper_open        # if the current and next gripper states are both same
        and obs.left.gripper_open == demo[max(0, i - 1)].left.gripper_open    # if the current and previous gripper states are same
        and demo[max(0, i - 2)].left.gripper_open == demo[max(0, i - 1)].left.gripper_open    # if the (i-1)th and (i-2)th gripper states are same
    )
    
    # Check if the current velocities are zero to some delta tolerance.
    small_delta = np.allclose(obs.left.joint_velocities, 0, atol=delta)
    
    # It is a stopping state if 1) the next is not final, 2) no change in gripper state, 3) vels are almost zero
    stopped = (
        stopped_buffer <= 0     # this is set as 4 at every stop point, so that there won't be less than 4 steps gap between stop points
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )

    return stopped

def _right_is_stopped(demo, i, obs, stopped_buffer, delta):

    next_is_not_final = i == (len(demo) - 2) # Check if next step is not the final one
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )

    # Check if the gripper state hasn't changed from (i-2) to (i+1)
    gripper_state_no_change = i < (len(demo) - 2) and (     # Not the second last or last step
        obs.right.gripper_open == demo[i + 1].right.gripper_open        # if the current and next gripper states are both same
        and obs.right.gripper_open == demo[max(0, i - 1)].right.gripper_open    # if the current and previous gripper states are same
        and demo[max(0, i - 2)].right.gripper_open == demo[max(0, i - 1)].right.gripper_open    # if the (i-1)th and (i-2)th gripper states are same
    )
    
    # Check if the current velocities are zero to some delta tolerance.
    small_delta = np.allclose(obs.right.joint_velocities, 0, atol=delta)
    
    # It is a stopping state if 1) the next is not final, 2) no change in gripper state, 3) vels are almost zero
    stopped = (
        stopped_buffer <= 0     # this is set as 4 at every stop point, so that there won't be less than 4 steps gap between stop points
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )

    return stopped

def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    
    episode_keypoints_left = []
    episode_keypoints_right = []
    
    prev_left_gripper_open = demo[0].left.gripper_open
    prev_right_gripper_open = demo[0].right.gripper_open
    left_stopped_buffer = 0
    right_stopped_buffer = 0

    for i, obs in enumerate(demo):
        
        # Check if current state obs is a stopped state:
        left_stopped = _left_is_stopped(demo, i, obs, left_stopped_buffer, stopping_delta)  # returns bool
        right_stopped = _right_is_stopped(demo, i, obs, right_stopped_buffer, stopping_delta)  # returns bool

        # Set as 4 at every stop point, so that there won't be less than 4 steps gap between stop points
        left_stopped_buffer = 4 if left_stopped else left_stopped_buffer - 1
        right_stopped_buffer = 4 if right_stopped else right_stopped_buffer - 1
        
        last = i == (len(demo) - 1)     # Check if end of episode
        
        # If change in gripper, stopped state, or end of episode, append the keypoint to the episode
        if i != 0 and (obs.left.gripper_open != prev_left_gripper_open or last or left_stopped):   
            episode_keypoints_left.append(i)
        if i != 0 and (obs.right.gripper_open != prev_right_gripper_open or last or right_stopped):   
            episode_keypoints_right.append(i)
        
        prev_left_gripper_open = obs.left.gripper_open
        prev_right_gripper_open = obs.right.gripper_open

    # If the last and second last keypoints are the same, pop it out.
    if (
        len(episode_keypoints_left) > 1
        and (episode_keypoints_left[-1] - 1) == episode_keypoints_left[-2]
    ):
        episode_keypoints_left.pop(-2)
    if (
        len(episode_keypoints_right) > 1
        and (episode_keypoints_right[-1] - 1) == episode_keypoints_right[-2]
    ):
        episode_keypoints_right.pop(-2)

    return episode_keypoints_left, episode_keypoints_right


def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc
    return torch.cat(obs, dim=0)
