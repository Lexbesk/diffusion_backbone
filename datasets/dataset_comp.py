import numpy as np

from .dataset_base_zarr import BaseDataset


class RLBenchCompDataset(BaseDataset):
    """RLBench compositional dataset."""
    quat_format= 'xyzw'

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

    def _load_instructions(self, instruction_file):
        # str(root)[:-5] + '_subgoals.npz'
        return np.load(instruction_file)['subgoals']

    def _get_task(self, idx):
        instr = self._get_instr(idx)
        return [instr[0].split()[0]]

    def _get_instr(self, idx):
        return [str(self._instructions[int(self.annos['lang_id'][idx])])]

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
        return super().__getitem__(idx)
