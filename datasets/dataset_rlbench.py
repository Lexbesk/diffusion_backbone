import json
import pickle
import random

import torch

from .dataset_base import BaseDataset
from .utils import to_tensor


class RLBenchDataset(BaseDataset):
    """RLBench dataset."""
    quat_format= 'xyzw'

    def __init__(
        self,
        root,
        instructions,
        precompute_instruction_encodings,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            precompute_instruction_encodings=precompute_instruction_encodings,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only
        )

    def _load_instructions(self, instruction_file=None):
        if instruction_file:
            if self._precompute_instr_encs:
                instructions = pickle.load(open(instruction_file, "rb"))
            else:
                instructions = json.load(open(instruction_file))
        else:
            instructions = None
        return instructions

    def _get_task(self, idx):
        return [self.tasks[int(self.annos['task_id'][idx])]]

    def _get_instr(self, idx):
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
        return instr

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            pcd: (N, n_cam, H, W) float16 (depth)
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
        }
        """
        return super().__getitem__(idx)


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
    train_copies = 1  # how many copies of the dataset to load


class GNFactorDataset(RLBenchDataset):
    """RLBench dataset under GNFactor setup."""
    tasks = [
        "close_jar", "open_drawer", "sweep_to_dustpan_of_size",
        "meat_off_grill", "turn_tap", "slide_block_to_color_target",
        "put_item_in_drawer", "reach_and_drag", "push_buttons",
        "stack_blocks"
    ]
    variations = range(0, 199)
    cameras = ("front",)
    train_copies = 2000  # how many copies of the dataset to load


class PeractTwoCamDataset(RLBenchDataset):
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
    cameras = ("wrist", "front")
    train_copies = 10  # how many copies of the dataset to load

    def _get_rgb(self, idx):
        return to_tensor(self.annos['rgb'][idx])[-2:]

    def _get_pcd(self, idx):
        return to_tensor(self.annos['depth'][idx])[-2:]


class PeractSingleCamDataset(RLBenchDataset):
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
    cameras = ("front",)
    train_copies = 10  # how many copies of the dataset to load

    def _get_rgb(self, idx):
        return to_tensor(self.annos['rgb'][idx])[-1:]

    def _get_pcd(self, idx):
        return to_tensor(self.annos['depth'][idx])[-1:]


class Peract2Dataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = [
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
    variations = range(0, 199)
    # cameras = (
    #     "front", "over_shoulder_left", "over_shoulder_right",
    #     "wrist_left", "wrist_right"
    # )
    cameras = ("front",)
    train_copies = 10  # how many copies of the dataset to load

    def _get_rgb(self, idx):
        return to_tensor(self.annos['rgb'][idx])[:1]

    def _get_pcd(self, idx):
        return to_tensor(self.annos['depth'][idx])[:1]

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
        }
        In addition self.annos may contain fields for task/instruction ids
        """
        idx = idx % len(self.annos['rgb'])
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),  # [str] or tensor(53, 512)
            "rgb": self._get_rgb(idx),  # tensor(n_cam, 3, H, W)
            "depth": self._get_pcd(idx),  # tensor(n_cam, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
            "extrinsics": to_tensor(self.annos['extrinsics'][idx])[:1],
            "intrinsics": to_tensor(self.annos['intrinsics'][idx])[:1]
        }
