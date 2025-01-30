from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path
from time import time
import json
import pickle

import numpy as np
import torch
import einops

import utils.pytorch3d_transforms as pytorch3d_transforms
from .utils import loader, TrajectoryInterpolator
from .dataset_base import BaseDataset


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.as_tensor(x)


def to_relative_action(actions, anchor_actions):
    assert actions.shape[-1] == 8

    rel_pos = actions[..., :3] - anchor_actions[..., :3]

    # pytorch3d takes wxyz quaternion, the input is xyzw
    rel_orn = pytorch3d_transforms.quaternion_multiply(
        actions[..., [6, 3, 4, 5]],
        pytorch3d_transforms.quaternion_invert(anchor_actions[..., [6,3,4,5]])
    )[..., [1, 2, 3, 0]]

    gripper = actions[..., -1:]
    rel_actions = torch.concat([rel_pos, rel_orn, gripper], dim=-1)

    return rel_actions


class RLBenchDataset(BaseDataset):
    """RLBench dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,
        precompute_instruction_encodings,
        # dataset specification
        taskvar=[('TASK_NAME', 0)],  # a list of (task, variation) tuples
        max_episode_length=5,  # maximum chunk length
        cache_size=0,  # number of episodes to cache
        max_episodes_per_task=100,  # maximum number of episodes per task
        cameras=("CAMERA_NAME", ),  # camera names
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        color_aug=False,
        # for trajectories
        dense_interpolation=False,  # whether to interpolate trajectories
        interpolation_length=100,  # length of interpolated trajectory
        bimanual=False,  # whether to return bimanual actions
        relative_action=False,  # whether to return relative actions
    ):
        if isinstance(root, (Path, str)):
            root = [Path(root)]

        super().__init__(
            root=root,
            training=training,
            image_rescale=image_rescale,
            relative_action=relative_action,
            color_aug=color_aug
        )

        # For RLBench datset, we save in the data structure as follows:
        # ROOT/
        #   TASK_NAME+VARIANCE_ID/
        #       ep_EP_ID.npy 
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._taskvar = taskvar
        self._bimanual = bimanual

        # For trajectory optimization, initialize interpolation tools
        self._interpolate_traj = TrajectoryInterpolator(
            use=dense_interpolation,
            interpolation_length=interpolation_length
        )

        # Keep variations and useful instructions
        self._precompute_instruction_encodings = precompute_instruction_encodings
        if precompute_instruction_encodings:
            self._instructions = pickle.load(open(instructions, "rb")) if instructions else None
        else:
            self._instructions = json.load(open(instructions)) if instructions else None
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                self._num_vars[task] += 1

        # File-names of episodes per task and variation
        episodes_by_task = defaultdict(list)  # {task: [(task, var, filepath)]}
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                # print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            # Split episodes equally into task variations
            if max_episodes_per_task > -1:
                episodes = episodes[
                    :max_episodes_per_task // self._num_vars[task] + 1
                ]
            episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
                eps = random.sample(eps, max_episodes_per_task)
            episodes_by_task[task] = sorted(
                eps, key=lambda t: int(str(t[2]).split('/')[-1][2:-4])
            )
            self._episodes += eps
            self._num_episodes += len(eps)
        print(f"Created dataset from {root} with {self._num_episodes}")
        self._episodes_by_task = episodes_by_task

    def read_from_cache(self, args):
        if self._cache_size == 0:
            return loader(args)

        if args in self._cache:
            return self._cache[args]

        value = loader(args)

        if len(self._cache) == self._cache_size:
            key = list(self._cache.keys())[int(time()) % self._cache_size]
            del self._cache[key]

        if len(self._cache) < self._cache_size:
            self._cache[args] = value

        return value

    @staticmethod
    def _unnormalize_rgb(rgb):
        # (from [-1, 1] to [0, 1]) to feed RGB to pre-trained backbone
        return rgb / 2 + 0.5

    def __newgetitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # Dynamic chunking so as not to overload GPU memory
        st_frame = random.randint(
            0, max(len(episode[0]) - self._max_episode_length, 0)
        )
        ed_frame = min(st_frame + self._max_episode_length, len(episode[0]))

        # Get the image tensors for the frame ids we got
        states = to_tensor(episode[1][st_frame:ed_frame])
        if states.dtype == torch.float64:
            states = states.float()

        # Camera ids
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            # Re-map states based on camera ids
            states = states[:, index]

        # Augmentations
        if self._resize is not None:
            states = einops.rearrange(states, "t n m c h w -> t n (m c) h w")
            states = self._resize(states=states)["states"]
            states = einops.rearrange(states, "t n (m c) h w -> t n m c h w", m=2)

        # Split RGB and XYZ
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]
        rgbs = self._unnormalize_rgb(rgbs)

        if self._color_aug is not None:
            rgbs = rgbs.mul(255).byte()
            rgbs = self._color_aug(rgbs)
            rgbs = rgbs.float().div(255)

        # Sample one instruction feature
        if self._precompute_instruction_encodings:
            if self._instructions:
                instr_ind = [
                    np.random.randint(len(self._instructions[task][variation]))
                    for _ in range(st_frame, ed_frame)
                ]
                instr = self._instructions[task][variation][instr_ind]
            else:
                instr = torch.zeros((rgbs.shape[0], 53, 512))
        else:
            if self._instructions:
                instr = [
                    random.choice(self._instructions[task][str(variation)])
                    for _ in range(st_frame, ed_frame)
                ]
            else:
                instr = [""] * (ed_frame - st_frame)

        # Get gripper tensors for respective frame ids
        action = torch.cat([
            to_tensor(episode[2][i]) for i in range(st_frame, ed_frame)
        ])
        gripper = torch.cat([
            to_tensor(episode[4][i]) for i in range(st_frame, ed_frame)
        ])

        # gripper history
        gripper_history = torch.stack([
            torch.cat([
                to_tensor(episode[4][max(0, i-2)])
                for i in range(st_frame, ed_frame)
            ]),
            torch.cat([
                to_tensor(episode[4][max(0, i-1)])
                for i in range(st_frame, ed_frame)
            ]),
            gripper
        ], dim=1)

        # Low-level trajectory
        if len(episode) > 5:
            traj_items = [
                self._interpolate_traj(to_tensor(episode[5][i]))
                for i in range(st_frame, ed_frame)
            ]
        else:
            traj_items = torch.cat([gripper, action], dim=0)
            traj_items = [
                self._interpolate_traj(traj_items[i])
                for i in range(len(gripper))
            ]
        max_l = max(len(item) for item in traj_items)
        traj = torch.zeros(len(traj_items), max_l, gripper.shape[-1])
        traj_lens = torch.as_tensor(
            [len(item) for item in traj_items]
        )
        for i, item in enumerate(traj_items):
            traj[i, :len(item)] = item
        traj_mask = torch.zeros(traj.shape[:-1])
        for i, len_ in enumerate(traj_lens.long()):
            traj_mask[i, len_:] = 1

        # Compute relative action
        if self._relative_action:
            traj = to_relative_action(traj, traj[:, :1])

        ret_dict = {
            "task": [task for _ in range(st_frame, ed_frame)],
            "instr": instr,
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "proprioception": gripper_history,
            "action": traj,  # e.g. tensor (n_frames, T, 8)
            "action_mask": traj_mask.bool()  # tensor (n_frames, T)
        }
        
        if self._bimanual:
            ret_dict["proprioception"] = ret_dict["proprioception"].unflatten(-1, (2, -1))
            ret_dict["action"] = ret_dict["action"].unflatten(-1, (2, -1))

        return ret_dict

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # Dynamic chunking so as not to overload GPU memory
        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        )

        # Get frame ids for this chunk
        frame_ids = episode[0][
            chunk * self._max_episode_length:
            (chunk + 1) * self._max_episode_length
        ]

        # Get the image tensors for the frame ids we got
        # states = torch.stack([
        #     episode[1][i] if isinstance(episode[1][i], torch.Tensor)
        #     else torch.from_numpy(episode[1][i])
        #     for i in frame_ids
        # ])
        states = torch.from_numpy(episode[1][frame_ids])

        # Camera ids
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            # Re-map states based on camera ids
            states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]
        rgbs = self._unnormalize_rgb(rgbs)

        if self._color_aug is not None:
            rgbs = rgbs.mul(255).byte()
            rgbs = self._color_aug(rgbs)
            rgbs = rgbs.float().div(255)

        # Get action tensors for respective frame ids
        action = torch.cat([
            episode[2][i] if isinstance(episode[2][i], torch.Tensor)
            else torch.from_numpy(episode[2][i])
            for i in frame_ids
        ])

        # Sample one instruction feature
        if self._instructions:
            instr = random.choice(self._instructions[task][variation])
            instr = instr[None].repeat(len(rgbs), 1, 1)
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))

        # Get gripper tensors for respective frame ids
        gripper = torch.cat([
            episode[4][i] if isinstance(episode[4][i], torch.Tensor)
            else torch.from_numpy(episode[4][i])
            for i in frame_ids
        ])

        # gripper history
        gripper_history = torch.stack([
            torch.cat([
                episode[4][max(0, i-2)] if isinstance(episode[4][i], torch.Tensor)
                else torch.from_numpy(episode[4][max(0, i-2)])
                for i in frame_ids
            ]),
            torch.cat([
                episode[4][max(0, i-1)] if isinstance(episode[4][i], torch.Tensor)
                else torch.from_numpy(episode[4][max(0, i-1)])
                for i in frame_ids
            ]),
            gripper
        ], dim=1)

        # Low-level trajectory
        traj, traj_lens = None, 0
        if len(episode) > 5:
            traj_items = [
                self._interpolate_traj(
                    episode[5][i] if isinstance(episode[5][i], torch.Tensor)
                    else torch.from_numpy(episode[5][i])
                )
                for i in frame_ids
            ]
        else:
            traj_items = [
                self._interpolate_traj(
                    torch.cat([
                        torch.from_numpy(episode[4][i]),
                        torch.from_numpy(episode[2][i])
                    ], dim=0)
                ) for i in frame_ids
            ]
        max_l = max(len(item) for item in traj_items)
        traj = torch.zeros(len(traj_items), max_l, gripper.shape[-1])
        traj_lens = torch.as_tensor(
            [len(item) for item in traj_items]
        )
        for i, item in enumerate(traj_items):
            traj[i, :len(item)] = item
        traj_mask = torch.zeros(traj.shape[:-1])
        for i, len_ in enumerate(traj_lens.long()):
            traj_mask[i, len_:] = 1

        # Augmentations
        if self._training:
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        ret_dict = {
            "task": [task for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            # "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            # "curr_gripper": gripper,
            "proprioception": gripper_history
        }
        ret_dict.update({
            "action": traj,  # e.g. tensor (n_frames, T, 8)
            "action_mask": traj_mask.bool()  # tensor (n_frames, T)
        })
        
        if self._bimanual:
            ret_dict['action'] = ret_dict['action'].unflatten(-1, (2, -1))
            ret_dict['curr_gripper'] = ret_dict['curr_gripper'].unflatten(-1, (2, -1))
            ret_dict['curr_gripper_history'] = ret_dict['curr_gripper_history'].unflatten(-1, (2, -1))
            ret_dict['trajectory'] = ret_dict['trajectory'].unflatten(-1, (2, -1))

        return ret_dict

    def __len__(self):
        return self._num_episodes


class GNFactorDataset(RLBenchDataset):
    """RLBench dataset under GNFactor setup."""
    tasks = [
        "close_jar", "open_drawer", "sweep_to_dustpan_of_size",
        "meat_off_grill", "turn_tap", "slide_block_to_color_target",
        "put_item_in_drawer", "reach_and_drag", "push_buttons",
        "stack_blocks"
    ]
    variations = range(0, 199)
    cameras=("front", )

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # the path to the instruction file
        precompute_instruction_encodings,  # whether instruction is latent encoded
        training,  # whether in training mode
        image_rescale=(0.75, 1.25),  # rescale factor for images
        dense_interpolation=True,  # whether to interpolate trajectories
        interpolation_length=2,  # length of interpolated trajectory
        relative_action=False,  # whether to return relative actions
    ):
        taskvar = [(task, var) for task in self.tasks for var in self.variations]
        cache_size = 200 if training else 0
        max_episode_length = 100
        max_episodes_per_task = 20
        color_aug = False
        bimanual = False
        # relative_action = False

        super().__init__(
            root=root,
            instructions=instructions,
            precompute_instruction_encodings=precompute_instruction_encodings,
            taskvar=taskvar,
            max_episode_length=max_episode_length,
            cache_size=cache_size,
            max_episodes_per_task=max_episodes_per_task,
            cameras=self.cameras,
            training=training,
            image_rescale=image_rescale,
            color_aug=color_aug,
            dense_interpolation=dense_interpolation,
            interpolation_length=interpolation_length,
            bimanual=bimanual,
            relative_action=relative_action
        )


class PeractDataset(RLBenchDataset):
    """RLBench dataset under Peract setup."""
    tasks = [
        "place_cups", "close_jar", "insert_onto_square_peg",
        "light_bulb_in", "meat_off_grill", "open_drawer",
        "place_shape_in_shape_sorter", "place_wine_at_rack_location",
        "push_buttons", "put_groceries_in_cupboard",
        "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
        "slide_block_to_color_target", "stack_blocks", "stack_cups",
        "sweep_to_dustpan_of_size", "turn_tap"
    ]
    variations = range(0, 199)
    cameras = ("left_shoulder", "right_shoulder", "wrist", "front")

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # the path to the instruction file
        precompute_instruction_encodings,  # whether instruction is latent encoded
        training,  # whether in training mode
        image_rescale,  # rescale factor for images
        dense_interpolation=False,  # whether to interpolate trajectories
        interpolation_length=100,  # length of interpolated trajectory
        relative_action=False,  # whether to return relative actions
    ):
        taskvar = [(task, var) for task in self.tasks for var in self.variations]
        cache_size = 0
        max_episode_length = 100
        max_episodes_per_task = -1
        color_aug = False
        bimanual = False
        # relative_action = False

        super().__init__(
            root=root,
            instructions=instructions,
            precompute_instruction_encodings=precompute_instruction_encodings,
            taskvar=taskvar,
            max_episode_length=max_episode_length,
            cache_size=cache_size,
            max_episodes_per_task=max_episodes_per_task,
            cameras=self.cameras,
            training=training,
            image_rescale=image_rescale,
            color_aug=color_aug,
            dense_interpolation=dense_interpolation,
            interpolation_length=interpolation_length,
            bimanual=bimanual,
            relative_action=relative_action
        )


class DebugPeractDataset(RLBenchDataset):
    """RLBench dataset under Peract setup."""
    tasks = [
        "place_cups",
    ]
    variations = range(0, 199)
    cameras = ("left_shoulder", "right_shoulder", "wrist", "front")

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # the path to the instruction file
        precompute_instruction_encodings,  # whether instruction is latent encoded
        training,  # whether in training mode
        image_rescale,  # rescale factor for images
        dense_interpolation=False,  # whether to interpolate trajectories
        interpolation_length=100,  # length of interpolated trajectory
        relative_action=False,  # whether to return relative actions
    ):
        taskvar = [(task, var) for task in self.tasks for var in self.variations]
        cache_size = 0
        max_episode_length = 100
        max_episodes_per_task = -1
        color_aug = False
        bimanual = False

        super().__init__(
            root=root,
            instructions=instructions,
            precompute_instruction_encodings=precompute_instruction_encodings,
            taskvar=taskvar,
            max_episode_length=max_episode_length,
            cache_size=cache_size,
            max_episodes_per_task=max_episodes_per_task,
            cameras=self.cameras,
            training=training,
            image_rescale=image_rescale,
            color_aug=color_aug,
            dense_interpolation=dense_interpolation,
            interpolation_length=interpolation_length,
            bimanual=bimanual,
            relative_action=relative_action
        )


class Peract2Dataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = [
        "bimanual_pick_laptop", "bimanual_pick_plate",
        "bimanual_straighten_rope", "bimanual_sweep_to_dustpan",
        "coordinated_lift_ball", "coordinated_lift_tray",
        "coordinated_push_box", "coordinated_put_bottle_in_fridge",
        "coordinated_put_item_in_drawer", "coordinated_take_tray_out_of_oven",
        "dual_push_buttons", "handover_item_easy", "handover_item"
    ]
    variations = range(0, 199)
    cameras = (
        "over_shoulder_left", "over_shoulder_right",
        "wrist_left", "wrist_right", "front"
    )

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # the path to the instruction file
        precompute_instruction_encodings,  # whether instruction is latent encoded
        training,  # whether in training mode
        image_rescale,  # rescale factor for images
        dense_interpolation=False,  # whether to interpolate trajectories
        interpolation_length=100,  # length of interpolated trajectory
        relative_action=False,  # whether to return relative actions
    ):
        taskvar = [(task, var) for task in self.tasks for var in self.variations]
        cache_size = 0
        max_episode_length = 100
        max_episodes_per_task = -1
        color_aug = False
        bimanual = True
        # relative_action = False

        super().__init__(
            root=root,
            instructions=instructions,
            precompute_instruction_encodings=precompute_instruction_encodings,
            taskvar=taskvar,
            max_episode_length=max_episode_length,
            cache_size=cache_size,
            max_episodes_per_task=max_episodes_per_task,
            cameras=self.cameras,
            training=training,
            image_rescale=image_rescale,
            color_aug=color_aug,
            dense_interpolation=dense_interpolation,
            interpolation_length=interpolation_length,
            bimanual=bimanual,
            relative_action=relative_action
        )