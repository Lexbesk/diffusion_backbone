import json

from .base import BaseDataset
from .utils import to_tensor


class CALVINDataset(BaseDataset):
    """CALVIN dataset."""
    cameras = ("front", "wrist")
    train_copies = 1
    quat_format= 'wxyz'

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
        return ['calvin']

    def _get_pcd(self, idx):
        return to_tensor(self.annos['pcd'][idx])

    def _get_instr(self, idx):
        t_ = int(self.annos['instr_id'][idx])
        return [self._instructions[t_]]

    def _get_rgb(self, idx):
        return to_tensor(self.annos['rgb_front'][idx])

    def _get_depth(self, idx):
        return to_tensor(self.annos['depth_front'][idx])

    def _get_rgb2d(self, idx):
        return to_tensor(self.annos['rgb_wrist'][idx])

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam3d, 160, 160) float16
            instr_id: (N,) int
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam3d, 3, 160, 160) uint8
            rgb2d: (N, n_cam2d, 3, 160, 160) uint8
        }
        """
        idx = idx % len(self.annos['rgb'])
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam3d, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam3d, H, W)
            "rgb2d": self._get_rgb(idx),  # tensor(n_cam3d, 3, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx)  # tensor(T, 8)
        }


class ABC_DDataset(CALVINDataset):
    """CALVIN dataset under ABC_D setup."""
    cameras = ("front", "wrist")
    train_copies = 1  # how many copies of the dataset to load


class ABC_DSingleCamDataset(CALVINDataset):
    """CALVIN dataset under ABC_D setup."""
    cameras = ("front")
    train_copies = 1  # how many copies of the dataset to load

    def _get_pcd(self, idx):
        return to_tensor(self.annos['pcd'][idx, :1])

    def _get_rgb(self, idx):
        return to_tensor(self.annos['rgb'][idx, :1])
