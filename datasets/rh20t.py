import json

from .base import BaseDataset
from .utils import to_tensor


class RH20TDataset(BaseDataset):
    """RH20T dataset."""
    quat_format= 'wxyz'
    train_copies = 1
    camera_inds = None

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only
        )

    def _load_instructions(self, instruction_file):
        return json.load(open(instruction_file))

    def _get_task(self, idx):
        return ['rh20t']

    def _get_instr(self, idx):
        t_ = int(self.annos['instr_id'][idx])
        return [self._instructions[t_]]

    def _get_rgb2d(self, idx):
        return to_tensor(self.annos['rgb2d'][idx])

    def _get_extrinsics(self, idx):
        return to_tensor(self.annos['extrinsics'][idx])

    def _get_intrinsics(self, idx):
        return to_tensor(self.annos['intrinsics'][idx])

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        idx = idx % len(self.annos['action'])
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),  # [str]
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam3d, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam3d, H, W)
            "rgb2d": self._get_rgb2d(idx),  # tensor(n_cam2d, 3, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
            "extrinsics": self._get_extrinsics(idx),  # tensor(n_cam3d, 4, 4)
            "intrinsics": self._get_intrinsics(idx)  # tensor(n_cam3d, 3, 3)
        }
