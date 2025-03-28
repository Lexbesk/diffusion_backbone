import os
import glob
import random

import open3d
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
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from online_evaluation_rlbench.get_stored_demos import get_stored_demos


ALL_RLBENCH_TASKS = [
    'bimanual_push_box',
    'bimanual_lift_ball',
    'bimanual_dual_push_buttons',
    'bimanual_pick_plate',
    'bimanual_put_item_in_drawer',
    'bimanual_put_bottle_in_fridge',
    'bimanual_handover_item',
    'bimanual_pick_laptop',
    'bimanual_straighten_rope',
    'bimanual_sweep_to_dustpan',
    'bimanual_lift_tray',
    'bimanual_handover_item_easy',
    'bimanual_take_tray_out_of_oven'
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

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._step_id = 0
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        target = action.copy()
        if self._last_action is not None:
            action[:, 7] = self._last_action[:, 7].copy()

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

        return obs, reward, terminate


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
        self._instr = [random.choice(self._instructions[task_str][str(variation)])]
        self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def predict(self, rgbs, pcds, gripper,
                prediction_len=None):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            prediction_len: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        output = {"action": None}

        rgbs = rgbs / 2 + 0.5  # in [0, 1]

        if self._instr is None:
            raise ValueError()

        # self._instr = self._instr.to(rgbs.device)
        self._task_id = self._task_id.to(rgbs.device)

        gripper = gripper.unflatten(-1, (2, -1))

        # Predict trajectory
        fake_traj = torch.full(
            [1, prediction_len, gripper.shape[-1]], 0
        ).to(rgbs.device)
        traj_mask = torch.full(
            [1, prediction_len, 2], False
        ).to(rgbs.device)
        # import pickle
        # with open('first.pkl', 'wb') as f:
        #     pickle.dump([rgbs.cpu(), pcds.cpu(), self._instr], f)
        # jkjk
        output["action"] = self._policy(
            None,
            traj_mask,
            rgbs,
            None,
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
        state = transform(state_dict)
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

    def evaluate_task_on_multiple_variations(
        self,
        task_str: str,
        max_steps: int,
        num_variations: int,  # -1 means all variations
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        prediction_len=100,
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

        for variation in tqdm(task_variations):
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
                    prediction_len=prediction_len,
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
        prediction_len=50,
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
                    prediction_len=prediction_len
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
                        obs, reward, terminate = move(action, collision_checking=False)

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
            render_mode=RenderMode.OPENGL3,
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


def transform(obs_dict):
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
