import json
import pickle

import torch

from .dataset_base_zarr import BaseDataset


class CALVINDataset(BaseDataset):
    """CALVIN dataset."""
    quat_format= 'wxyz'

    def __init__(
        self,
        root,
        instructions,
        precompute_instruction_encodings,
        copies=None,
        relative_action=False,
        mem_limit=8
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            precompute_instruction_encodings=precompute_instruction_encodings,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit
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
        return ['calvin']

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

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            pcd: (N, n_cam, 3, 160, 160) float16
            instr_id: (N,) int
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, 160, 160) uint8
        }
        """
        return super().__getitem__(idx)


class ABC_DDataset(CALVINDataset):
    """CALVIN dataset under ABC_D setup."""
    cameras = ("front", "wrist")
    train_copies = 1  # how many copies of the dataset to load
