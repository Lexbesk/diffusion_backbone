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

from .utils import loader, TrajectoryInterpolator
from utils.utils_with_calvin import to_relative_action, convert_rotation
from .dataset_base import BaseDataset


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.as_tensor(x)


class CalvinDataset(BaseDataset):
    """CALVIN dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,
        precompute_instruction_encodings,
        # dataset specification
        scene=["SCENE_NAME", ],  # a list of scene names
        max_episode_length=5,  # maximum chunk length
        cache_size=0,  # number of episodes to cache
        max_episodes_per_scene=100,  # maximum number of episodes per scene
        cameras=("CAMERA_NAME", ),  # camera names
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        color_aug=False,
        # for trajectories
        dense_interpolation=False,  # whether to interpolate trajectories
        interpolation_length=100,  # length of interpolated trajectory
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

        # For CALVIN datset, we save in the data structure as follows:
        # ROOT/
        #   SCENE_NAME/
        #       ep_EP_ID.npy 
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._scene = scene

        # For trajectory optimization, initialize interpolation tools
        self._interpolate_traj = TrajectoryInterpolator(
            use=dense_interpolation,
            interpolation_length=interpolation_length
        )

        # Keep useful instructions
        self._precompute_instruction_encodings = precompute_instruction_encodings
        if precompute_instruction_encodings:
            self._instructions = pickle.load(open(instructions, "rb")) if instructions else None
        else:
            self._instructions = json.load(open(instructions)) if instructions else None
        self._instructions = None

        # File-names of episodes per scene
        episodes_by_scene = defaultdict(list)
        for root, scene in itertools.product(self._root, scene):
            data_dir = root / f"{scene}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(scene, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(scene, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(scene, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            # Split episodes equally
            if max_episodes_per_scene > -1:
                episodes = episodes[:max_episodes_per_scene]
            episodes_by_scene[scene] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for scene, eps in episodes_by_scene.items():
            if len(eps) > max_episodes_per_scene and max_episodes_per_scene > -1:
                eps = random.sample(eps, max_episodes_per_scene)
            self._episodes += eps
            self._num_episodes += len(eps)
        print(f"Created dataset from {root} with {self._num_episodes}")
        self._episodes_by_scene = episodes_by_scene

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
        scene, file = self._episodes[episode_id]

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
        rgbs = states[:, :, 0, :, 20:180, 20:180]
        pcds = states[:, :, 1, :, 20:180, 20:180]
        rgbs = self._unnormalize_rgb(rgbs)

        if self._color_aug is not None:
            rgbs = rgbs.mul(255).byte()
            rgbs = self._color_aug(rgbs)
            rgbs = rgbs.float().div(255)

        # Sample one instruction feature
        if self._precompute_instruction_encodings:
            if self._instructions:
                instr_ind = episode[6][0]
                instr = self._instructions[instr_ind]
                instr = instr.repeat(len(rgbs), 1, 1)
            else:
                instr = torch.zeros((rgbs.shape[0], 53, 512))
        else:
            if self._instructions:
                instr_ind = episode[6][0]
                instr = [
                    random.choice(self._instructions[instr_ind])
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
            rel_traj = torch.zeros_like(traj)
            for i in range(traj.shape[0]):
                for j in range(traj.shape[1]):
                    rel_traj[i, j] = torch.as_tensor(to_relative_action(
                        traj[i, j].numpy(), traj[i, 0].numpy(), clip=False
                    ))
            traj = rel_traj

        # Convert Euler angles to Quarternion
        gripper_history = torch.cat([
            gripper_history[..., :3],
            torch.as_tensor(convert_rotation(gripper_history[..., 3:6])),
            gripper_history[..., 6:]
        ], dim=-1)
        traj = torch.cat([
            traj[..., :3],
            torch.as_tensor(convert_rotation(traj[..., 3:6])),
            traj[..., 6:]
        ], dim=-1)

        ret_dict = {
            "task": [scene for _ in range(st_frame, ed_frame)],
            "instr": instr,
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "proprioception": gripper_history,
            "action": traj,  # e.g. tensor (n_frames, T, 8)
            "action_mask": traj_mask.bool()  # tensor (n_frames, T)
        }
        

        return ret_dict

    def __len__(self):
        return self._num_episodes


class TrainABCTestD_CalvinDataset(CalvinDataset):
    """CALVIN dataset under the setup of training with scene A, B, C
    and testing with scene D."""
    scenes = [
        "A", "B", "C", "D",
    ]
    cameras=("front", "wrist")

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
        scene = self.scenes
        cache_size = 0
        max_episode_length = 100
        max_episodes_per_scene = -1
        color_aug = False
        instructions = (
            f"{instructions}/training.pkl" if training
            else f"{instructions}/validation.pkl"
        )

        super().__init__(
            root=root,
            instructions=instructions,
            precompute_instruction_encodings=precompute_instruction_encodings,
            scene=scene,
            max_episode_length=max_episode_length,
            cache_size=cache_size,
            max_episodes_per_scene=max_episodes_per_scene,
            cameras=self.cameras,
            training=training,
            image_rescale=image_rescale,
            color_aug=color_aug,
            dense_interpolation=dense_interpolation,
            interpolation_length=interpolation_length,
            relative_action=relative_action
        )
