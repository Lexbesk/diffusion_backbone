import random
import itertools
from typing import Tuple
import pickle
from pathlib import Path
# from ipdb import set_trace as st

import blosc
from tqdm import tqdm
import tap
import torch
import numpy as np
import einops

from utils.utils_with_rlbench import (
    RLBenchEnv,
    keypoint_discovery,
    transform
)


class Arguments(tap.Tap):
    data_dir: Path = Path("/home/sirdome/katefgroup/pushkal_pc/raw_256/18_peract_tasks_train_new/")
    seed: int = 2
    tasks: Tuple[str, ...] = ("close_jar",)
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front")
    image_size: str = "256,256"
    output: Path = Path(__file__).parent / "datasets"
    max_variations: int = 199
    offset: int = 0
    num_workers: int = 0
    store_intermediate_actions: int = 1


def get_observation(task_str: str, variation: int,
                    episode: int, env: RLBenchEnv,
                    store_intermediate_actions: bool):
    demos = env.get_demo(task_str, variation, episode)
    demo = demos[0]

    key_frame = keypoint_discovery(demo)

    # Every five frames + keyframes
    inds = np.arange(0, len(demo._observations), 5).tolist()[:-1] + sorted(key_frame)[:-1]
    inds = sorted(list(set(inds)))

    # Target keyframe (the next keyframe or last frame)
    k_i = 0
    target_inds = []
    prev_inds = []
    prev_prev_inds = []
    for i in range(len(inds)):
        while k_i < len(key_frame) and key_frame[k_i] < inds[i]:
            k_i += 1
        if k_i == len(key_frame):
            target_inds.append(len(demo._observations) - 1)
        else:
            target_inds.append(key_frame[k_i])
        prev_inds.append(key_frame[k_i - 1] if k_i > 0 else 0)
        prev_prev_inds.append(key_frame[k_i - 2] if k_i > 1 else 0)

    # Get observations and states for all inds
    ind2obs_state = {ind: env.get_obs_action(demo._observations[ind]) for ind in inds}
    for ind in target_inds:
        if ind not in ind2obs_state:
            ind2obs_state[ind] = env.get_obs_action(demo._observations[ind])

    # Store observations, proprioceptive states and actions
    obs_ls = []
    prop_ls = []
    action_ls = []

    for i in range(len(inds)):
        obs, state = ind2obs_state[inds[i]]
        # Observation
        obs = transform(obs)
        obs_ls.append(obs.unsqueeze(0))
        # For proprioception, add history
        _, prev_prev = ind2obs_state[prev_prev_inds[i]]
        _, prev = ind2obs_state[prev_inds[i]]
        prop_ls.append(torch.stack([prev_prev, prev, state]))
        # Action
        _, action = ind2obs_state[target_inds[i]]
        action_ls.append(action.unsqueeze(0))

    return demo, obs_ls, prop_ls, action_ls


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args: Arguments):
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=args.data_dir,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_depth=False,
            apply_pc=True,
            apply_cameras=args.cameras,
        )

        tasks = args.tasks
        variations = range(args.offset, args.max_variations)
        self.items = []
        for task_str, variation in itertools.product(tasks, variations):
            episodes_dir = args.data_dir / task_str / f"variation{variation}" / "episodes"
            episodes = [
                (task_str, variation, int(ep.stem[7:]))
                for ep in episodes_dir.glob("episode*")
            ]
            self.items += episodes

        self.num_items = len(self.items)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        taskvar_dir = args.output / f"{task}+{variation}"
        taskvar_dir.mkdir(parents=True, exist_ok=True)

        (demo,
         state_ls,
         prop_ls,
         action_ls) = get_observation(
            task, variation, episode, self.env,
            bool(args.store_intermediate_actions)
        )

        state_ls = einops.rearrange(
            state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(args.cameras),
            m=2
        )

        frame_ids = list(range(len(state_ls)))

        state_dict = []
        print("Demo {}".format(episode))
        state_dict.append(frame_ids)
        state_dict.append(state_ls.numpy())
        state_dict.append(prop_ls)
        state_dict.append(action_ls)

        with open(taskvar_dir / f"ep{episode}.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))


if __name__ == "__main__":
    args = Arguments().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    for _ in tqdm(dataloader):
        continue
