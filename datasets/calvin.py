from .base import BaseDataset


class CALVINDataset(BaseDataset):
    """CALVIN dataset."""
    cameras = ("front", "wrist")
    train_copies = 1
    quat_format= 'wxyz'
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

    def _get_task(self, idx):
        return ['calvin']

    def _get_instr(self, idx):
        t_ = int(self.annos['instr_id'][idx])
        return [self._instructions[t_]]

    def _get_extrinsics_wrist(self, idx):
        return self._get_attr_by_idx(idx, 'extrinsics_wrist')

    def __getitem__(self, idx):
        idx = idx % len(self.annos['action'])
        if self._actions_only:
            return {"action": self._get_action(idx)[:, None]}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx, 'rgb_front'),  # n_cam3d, 3, H, W
            "depth": self._get_depth(idx, 'depth_front'),  # n_cam3d H W
            "rgb2d": self._get_rgb(idx, 'rgb_wrist'),  # n_cam2d 3 H W
            "wrist_depth": self._get_depth(idx, 'depth_wrist'),  # n_cam2d H, W
            "extrinsics_wrist": self._get_extrinsics_wrist(idx),  # tensor(4, 4)
            # Unsqueeze action and proprio to include an "nhand" dim
            "proprioception": self._get_proprioception(idx)[:, None],  # 1 1 8
            "action": self._get_action(idx)[:, None]  # tensor(T, 1, 8)
        }
