import numpy as np
import torch

from .utils import to_tensor, read_zarr_with_cache, to_relative_action


class RLBenchCompDataset:
    """RLBench compositional dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,
        precompute_instruction_encodings=False,
        copies=1,  # how many copies of the dataset to load
        relative_action=False  # whether to return relative actions
    ):
        self.copies = copies
        self._relative_action = relative_action

        # Load instructions
        instructions = str(root)[:-5] + '_subgoals.npz'
        self._instructions = np.load(instructions)['subgoals']

        # Load all annotations lazily
        self.annos = read_zarr_with_cache(root)

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, 256, 256) float16
            lang_id: (N,) int
            proprioception: (N, 1, 8) float
            rgb: (N, n_cam, 3, 256, 256) uint8
        }
        """
        idx = idx % len(self.annos['action'])

        # Visual observations
        rgbs = to_tensor(self.annos['rgb'][idx])
        pcds = to_tensor(self.annos['depth'][idx])

        # Instruction
        instr = [str(self._instructions[int(self.annos['lang_id'][idx])])]

        # Get gripper tensors for respective frame ids
        action = to_tensor(self.annos['action'][idx])
        gripper_history = to_tensor(self.annos['proprioception'][idx])

        # Compute relative action
        if self._relative_action:
            action = to_relative_action(action, action[:1])

        ret_dict = {
            "task": [instr[0].split()[0]],
            "instr": instr,  # [str] or tensor(53, 512)
            "rgbs": rgbs,  # tensor(n_cam, 3, H, W)
            "pcds": pcds,  # tensor(n_cam, H, W)
            "proprioception": gripper_history,  # tensor(1, 8)
            "action": action,  # tensor(T, 8)
            "action_mask": torch.zeros(action.shape[:-1]).bool()  # tensor (T,)
        }

        return ret_dict

    def __len__(self):
        return self.copies * len(self.annos['action'])
