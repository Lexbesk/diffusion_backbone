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
        actions_only=False,
        chunk_size=4
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=False,
            mem_limit=mem_limit,
            actions_only=actions_only,
            chunk_size=chunk_size
        )

    def _get_task(self, idx):
        return ['calvin'] * self.chunk_size

    def _get_instr(self, idx):
        return [self._instructions[int(t_)] for t_ in self.annos['instr_id'][idx:idx + self.chunk_size]]

    def __getitem__(self, idx):
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size - 1)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)[:, None]}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_attr_by_idx(idx, 'rgb_front'),  # n_cam3d, 3, H, W
            "depth": self._get_attr_by_idx(idx, 'depth_front'),  # n_cam3d H W
            "rgb2d": self._get_attr_by_idx(idx, 'rgb_wrist'),  # n_cam2d 3 H W
            "wrist_depth": self._get_attr_by_idx(idx, 'depth_wrist'),  # n_cam2d H, W
            "extrinsics_wrist": self._get_attr_by_idx(idx, 'extrinsics_wrist'),  # tensor(4, 4)
            # Unsqueeze action and proprio to include an "nhand" dim
            "proprioception": self._get_attr_by_idx(idx, 'proprioception')[:, :, None],  # 1 1 8
            "action": self._get_attr_by_idx(idx, 'action')[:, 1:, None]  # tensor(T, 1, 8)
        }
