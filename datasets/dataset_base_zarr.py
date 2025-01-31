from torch.utils.data import Dataset

from .utils import to_tensor, read_zarr_with_cache, to_relative_action


class BaseDataset(Dataset):
    """Base dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # path to instruction file
        precompute_instruction_encodings,  # if true, load tensors, else str
        copies=None,  # copy the dataset for less loader restarts
        relative_action=False,  # whether to return relative actions
        mem_limit=8  # cache limit per dataset class in GigaBytes
    ):
        self._precompute_instr_encs = precompute_instruction_encodings
        self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action

        # Load instructions
        self._instructions = self._load_instructions(instructions)

        # Load all annotations lazily
        self.annos = read_zarr_with_cache(root, mem_gb=mem_limit)

    def _load_instructions(self, instruction_file):
        return None

    def _get_task(self, idx):
        return ["task"]

    def _get_instr(self, idx):
        return [""]

    def _get_rgb(self, idx):
        return to_tensor(self.annos['rgb'][idx])

    def _get_pcd(self, idx):
        return to_tensor(self.annos['pcd'][idx])

    def _get_proprioception(self, idx):
        return to_tensor(self.annos['proprioception'][idx])

    def _get_action(self, idx):
        action = to_tensor(self.annos['action'][idx])
        if self._relative_action:
            action = to_relative_action(action, action[:1], self.quat_format)
        return action

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            pcd: (N, n_cam, H, W) float16 (depth)
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
        }
        In addition self.annos may contain fields for task/instruction ids
        """
        idx = idx % len(self.annos['rgb'])
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),  # [str] or tensor(53, 512)
            "rgbs": self._get_rgb(idx),  # tensor(n_cam, 3, H, W)
            "pcds": self._get_pcd(idx),  # tensor(n_cam, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx)  # tensor(T, 8)
        }

    def __len__(self):
        return self.copies * len(self.annos['rgb'])
