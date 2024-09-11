import math
import random

import torch

from utils.utils_with_mobaloha import to_relative_action
from .dataset_engine import RLBenchDataset


class MobileAlohaDataset(RLBenchDataset):
    """Dataset class for Mobile Aloha."""

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
        episode[0] = list(range(len(episode[0])))
        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        )

        # Get frame ids for this chunk
        frame_ids = episode[0][
            chunk * self._max_episode_length:
            (chunk + 1) * self._max_episode_length
        ]

        # Get the image tensors for the frame ids we got
        states = torch.stack([
            episode[1][i] if isinstance(episode[1][i], torch.Tensor)
            else torch.from_numpy(episode[1][i])
            for i in frame_ids
        ])

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
        # rgbs = self._unnormalize_rgb(rgbs)

        # Get action tensors for respective frame ids
        # action = torch.cat([torch.from_numpy(episode[2][i]) for i in frame_ids])
        action = torch.cat([episode[2][i].view(1, -1) for i in frame_ids])

        # Sample one instruction feature
        if self._instructions:
            instr = random.choice(self._instructions[task][variation])
            instr = instr[None].repeat(len(rgbs), 1, 1)
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))

        # Get gripper tensors for respective frame ids
        # gripper = torch.cat([torch.from_numpy(episode[4][i]) for i in frame_ids])
        gripper = torch.cat([episode[4][i].view(1, -1) for i in frame_ids])

        # gripper history
        gripper_history = torch.stack([
            # torch.cat([torch.from_numpy(episode[4][max(0, i-2)]) for i in frame_ids]),
            # torch.cat([torch.from_numpy(episode[4][max(0, i-1)]) for i in frame_ids]),
            torch.cat([episode[4][max(0, i-2)].view(1, -1) for i in frame_ids]),
            torch.cat([episode[4][max(0, i-1)].view(1, -1) for i in frame_ids]),
            gripper
        ], dim=1)

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            if len(episode) > 5:
                try:
                    traj_items = [
                        # self._interpolate_traj(torch.from_numpy(episode[5][i]))
                        self._interpolate_traj(episode[5][i].flatten(1, -1))
                        for i in frame_ids
                    ]
                except:
                    import ipdb; ipdb.set_trace()
            else:
                traj_items = [
                    self._interpolate_traj(
                        # torch.cat([torch.from_numpy(episode[4][i]),
                        #            torch.from_numpy(episode[2][i])], dim=0)
                        torch.cat([episode[4][i].view(1, -1),
                                   episode[2][i].view(1, -1)], dim=0)
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

        # Compute relative action
        if self._relative_action and traj is not None:
            rel_traj = torch.zeros_like(traj)
            for i in range(traj.shape[0]):
                for j in range(traj.shape[1]):
                    rel_traj[i, j] = torch.as_tensor(to_relative_action(
                        traj[i, j], traj[i, 0]
                    ))
            traj = rel_traj

        ret_dict = {
            "task": [task for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper,
            "curr_gripper_history": gripper_history
        }
        if self._return_low_lvl_trajectory:
            ret_dict.update({
                "trajectory": traj,  # e.g. tensor (n_frames, T, 8)
                "trajectory_mask": traj_mask.bool()  # tensor (n_frames, T)
            })
        
        if self._bimanual:
            ret_dict['action'] = ret_dict['action'].unflatten(-1, (2, -1))
            ret_dict['curr_gripper'] = ret_dict['curr_gripper'].unflatten(-1, (2, -1))
            ret_dict['curr_gripper_history'] = ret_dict['curr_gripper_history'].unflatten(-1, (2, -1))
            ret_dict['trajectory'] = ret_dict['trajectory'].unflatten(-1, (2, -1))

        return ret_dict
