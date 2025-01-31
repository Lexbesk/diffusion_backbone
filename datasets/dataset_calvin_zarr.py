import json
import pickle

import torch

from .utils import to_tensor, read_zarr_with_cache, to_relative_action


class CalvinDataset:
    """CALVIN dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,
        precompute_instruction_encodings,
        copies=None,
        relative_action=False  # whether to return relative actions
    ):
        self.copies = self.train_copies if copies is None else copies
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

    def _get_rgb(self, idx):
        return to_tensor(self.annos['rgb'][idx])

    def _get_pcd(self, idx):
        return to_tensor(self.annos['pcd'][idx])

    def _get_instr(self, idx):
        t_ = int(self.annos['instr_id'][idx])
        if self._precompute_instr_encs:
            if self._instructions:
                instr = self._instructions['embeddings'][t_].squeeze(1)
            else:
                instr = torch.zeros((53, 512))
        else:
            if self._instructions:
                instr = [self._instructions[t_]]
            else:
                instr = [""]
        return instr

    def _get_proprioception(self, idx):
        return to_tensor(self.annos['proprioception'][idx])

    def _get_action(self, idx):
        action = to_tensor(self.annos['action'][idx])
        if self._relative_action:
            action = to_relative_action(action, action[:1])
        return action

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            pcd: (N, n_cam, 3, 160, 256) float16
            instr_id: (N,) int
            proprioception: (N, 1, 8) float
            rgb: (N, n_cam, 3, 256, 256) uint8
        }
        """
        idx = idx % len(self.annos['rgb'])
        ret_dict = {
            "task": ['calvin'],
            "instr": self._get_instr(idx),  # [str] or tensor(53, 512)
            "rgbs": self._get_rgb(idx),  # tensor(n_cam, 3, H, W)
            "pcds": self._get_pcd(idx),  # tensor(n_cam, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx)  # tensor(T, 8)
        }
        ret_dict["action_mask"] = torch.zeros(
            ret_dict["action"].shape[:-1], dtype=bool
        )
        return ret_dict

    def __len__(self):
        return self.copies * len(self.annos['rgb'])


class ABC_DDataset(CalvinDataset):
    """CALVIN dataset under Peract setup."""
    cameras = ("front", "wrist")
    train_copies = 1  # how many copies of the dataset to load
