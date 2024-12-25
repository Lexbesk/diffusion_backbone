import json
import pickle
import random

import torch

from .utils import to_tensor, read_zarr_with_cache, to_relative_action


class RLBenchDataset:
    """RLBench dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,
        precompute_instruction_encodings,
        relative_action=False  # whether to return relative actions
    ):
        self._relative_action = relative_action

        # Load instructions
        self._precompute_instr_encs = precompute_instruction_encodings
        if instructions:
            if precompute_instruction_encodings:
                self._instructions = pickle.load(open(instructions, "rb"))
            else:
                self._instructions = json.load(open(instructions))
        else:
            instructions = None

        # Load all annotations lazily
        self.annos = read_zarr_with_cache(root)

    def __getitem__(self, idx):
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
        # Split RGB and XYZ
        rgbs = to_tensor(self.annos['rgb'][idx])
        pcds = to_tensor(self.annos['depth'][idx])

        # Sample one instruction feature
        t_ = int(self.annos['task_id'][idx])
        v_ = int(self.annos['variation'][idx])
        task = self.tasks[t_]
        if self._precompute_instr_encs:
            if self._instructions:
                instr = random.choice(self._instructions[task][v_])
            else:
                instr = torch.zeros((53, 512))
        else:
            if self._instructions:
                instr = [random.choice(self._instructions[task][str(v_)])]
            else:
                instr = [""]

        # Get gripper tensors for respective frame ids
        action = to_tensor(self.annos['action'][idx])
        gripper_history = to_tensor(self.annos['proprioception'][idx])

        # Compute relative action
        if self._relative_action:
            action = to_relative_action(action, action[:1])

        ret_dict = {
            "task": [task],
            "instr": instr,  # [str] or tensor(53, 512)
            "rgbs": rgbs,  # tensor(n_cam, 3, H, W)
            "pcds": pcds,  # tensor(n_cam, H, W)
            "proprioception": gripper_history,  # tensor(1, 8)
            "action": action,  # tensor(T, 8)
            "action_mask": torch.zeros(action.shape[:-1]).bool()  # tensor (T,)
        }

        return ret_dict

    def __len__(self):
        return len(self.annos['variation'])


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
    cameras = ("left_shoulder", "right_shoulder", "wrist", "front")


class GNFactorDataset(RLBenchDataset):
    """RLBench dataset under GNFactor setup."""
    tasks = [
        "close_jar", "open_drawer", "sweep_to_dustpan_of_size",
        "meat_off_grill", "turn_tap", "slide_block_to_color_target",
        "put_item_in_drawer", "reach_and_drag", "push_buttons",
        "stack_blocks"
    ]
    cameras = ("front",)
