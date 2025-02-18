import torch

from .utils import to_tensor, read_zarr_with_cache, to_relative_action


class IsaacDataset:
    """IsaacGym-collected dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        copies=1,  # how many copies of the dataset to load
        relative_action=False,  # whether to return relative actions
        mem_limit=8,
        actions_only=False
    ):
        self.copies = copies
        self._relative_action = relative_action
        self._actions_only = actions_only

        # Load all annotations lazily
        self.annos = read_zarr_with_cache(root, mem_limit)

    def __getitem__(self, idx):
        idx = idx % len(self.annos['rgb'])

        # Split RGB and XYZ
        rgbs = to_tensor(self.annos['rgb'][idx])
        segs = to_tensor(self.annos['seg'][idx])
        pcds = to_tensor(self.annos['depth'][idx])
        proj_matrix = to_tensor(self.annos['proj_matrix'][idx])
        extrinsics = to_tensor(self.annos['extrinsics'][idx])

        # Get gripper tensors for respective frame ids
        action = to_tensor(self.annos['action'][idx])
        gripper_history = to_tensor(self.annos['proprioception'][idx])

        # Compute relative action
        if self._relative_action:
            action = to_relative_action(action, action[:1])
        if self._actions_only:
            return {"action": self._get_action(idx)}

        ret_dict = {
            "rgbs": rgbs,  # tensor(n_cam, 3, H, W)
            "segs": segs,  # tensor(n_cam, H, W)
            "pcds": pcds,  # tensor(n_cam, H, W)
            "proj_matrix": proj_matrix,  # tensor(n_cam, 4, 4)
            "extrinsics": extrinsics,  # tensor(n_cam, 4, 4)
            "proprioception": gripper_history,  # tensor(1, 8)
            "action": action,  # tensor(T, 8)
            "action_mask": torch.zeros(action.shape[:-1]).bool()  # tensor (T,)
        }

        return ret_dict

    def __len__(self):
        return self.copies * len(self.annos['rgb'])
