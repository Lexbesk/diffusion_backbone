# dex_zarr_dataset.py
import json
from torch.utils.data import Dataset
from .utils import to_tensor, read_zarr_with_cache
from .base import BaseDataset

REQUIRED_KEYS = (
    "partial_points",
    "full_points",        # NEW (N_full, 3) float32
    "pregrasp_qpos",
    "grasp_qpos",
    "squeeze_qpos",
    "grasp_type_id",
    "anchor_visible",     # NEW (bool / uint8)
    "obj_pose",           # NEW (7,)
    "obj_scale",          # NEW scalar
    "obj_path",           # NEW UTF-8 string
)

class DexZarrDataset(BaseDataset):
    """
    Dataset for the zarr created with dex2zarr.py.
    Each index i corresponds to exactly one grasp trial.
    """
    train_copies = 1

    def __init__(
        self,
        root,                     # /path/to/<split>.zarr
        mem_limit=8,             # GB to cache with zarr's LRU
        copies=None,
        chunk_size=1
    ):
        super().__init__(       
            root=root,
            instructions=None,
            copies=copies,
            relative_action=False,
            mem_limit=mem_limit,
            actions_only=False,
            chunk_size=chunk_size)
        
    def _get_partial_points(self, idx, key='partial_points'):
        return self._get_attr_by_idx(idx, key, False)
    
    def _get_pregrasp_qpos(self, idx, key='pregrasp_qpos'):
        return self._get_attr_by_idx(idx, key, False)
    
    def _get_grasp_qpos(self, idx, key='grasp_qpos'):
        return self._get_attr_by_idx(idx, key, False)
    
    def _get_squeeze_qpos(self, idx, key='squeeze_qpos'):
        return self._get_attr_by_idx(idx, key, False)
    
    def _get_grasp_type_id(self, idx, key='grasp_type_id'):
        return self._get_attr_by_idx(idx, key, False)

    def _get_obj_pose(self, idx, key='obj_pose'):
        return self._get_attr_by_idx(idx, key, False)
    
    def _get_obj_scale(self, idx, key='obj_scale'):
        return self._get_attr_by_idx(idx, key, False)
    
    def _get_obj_path(self, idx):
        # """
        # Returns a plain Python str (chunk_size==1) or list[str] (chunk_size>1).
        # We intentionally avoid `to_tensor` because torch Tensors cannot hold
        # variable-length strings.
        # """
        # dset = self.annos["obj_path"]            # Zarr object array
        # if self.chunk_size == 1:
        #     raw = dset[idx]
        #     return raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        # else:
        #     raw_slice = dset[idx : idx + self.chunk_size]
        #     return [
        #         r.decode() if isinstance(r, (bytes, bytearray)) else str(r)
        #         for r in raw_slice.tolist()
        #     ]
        return [
            tid
            for tid in self.annos['obj_path'][idx:idx + self.chunk_size]
        ]

    def __getitem__(self, idx):
        """
        Returns a dict:
        {
            partial_points : (4096, 3)  float32
            pregrasp_qpos  : (29,)      float32
            grasp_qpos     : (29,)      float32
            squeeze_qpos   : (29,)      float32
            grasp_type_id  : int
            (optional) instr : str
        }
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['grasp_qpos']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        sample = {
            "partial_points": self._get_partial_points(idx),
            "pregrasp_qpos": self._get_pregrasp_qpos(idx),
            "grasp_qpos": self._get_grasp_qpos(idx),
            "squeeze_qpos": self._get_squeeze_qpos(idx),
            "grasp_type_id": self._get_grasp_type_id(idx),
            "anchor_visible": self._get_attr_by_idx(idx, 'anchor_visible', False),
            "obj_pose"      : self._get_obj_pose(idx),         # (7,) float32
            "obj_scale"     : self._get_obj_scale(idx),        # float
            "obj_path"      : self._get_obj_path(idx),         # str
        }


        return sample


# grasp_dataset.py
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
import random
import numpy as np
import torch

_DEFAULT_POSE = np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
_ZEROS29      = np.zeros((1, 29), dtype=np.float32)

class GraspXLDataset(Dataset):
    def __init__(
        self,
        root,
        mem_limit=8,
        copies=None,
        chunk_size=1,
        partial_choice=0,
    ):
        """
        Parameters
        ----------
        root : str | Path
            Folder that contains the GraspXL object sub-directories (e.g. .../mixed_train).
        partial_choice : "random" | int | callable
            • "random"   choose a random `view_*.npy` each time __getitem__ is called
            • int        always choose that fixed index (0-8)
            • callable   a function that maps `List[Path]` → chosen Path
        """
        self.root = Path(root).expanduser().resolve()
        self.obj_dirs: List[Path] = [
            d for d in sorted(self.root.iterdir()) if (d / "combined.obj").exists()
        ]
        if not self.obj_dirs:
            raise RuntimeError(f"No object folders with combined.obj found in {self.root}")

        self.partial_choice = partial_choice

    def _pick_partial(self, view_files: List[Path]) -> Path:
        if callable(self.partial_choice):
            return self.partial_choice(view_files)
        if self.partial_choice == "random":
            return random.choice(view_files)
        # assume int
        idx = int(self.partial_choice) % len(view_files)
        return view_files[idx]

    def __len__(self) -> int:
        return len(self.obj_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        folder = self.obj_dirs[idx]
        view_files = sorted((folder / "partials").glob("view_*.npy"))
        if not view_files:
            raise FileNotFoundError(f"No partials in {folder}")
        pcd = np.load(self._pick_partial(view_files)).astype(np.float32)[None]   # [4096,3]

        sample = {
            "grasp_qpos":    torch.from_numpy(_ZEROS29.copy()),    # placeholders
            "pregrasp_qpos": torch.from_numpy(_ZEROS29.copy()),
            "squeeze_qpos":  torch.from_numpy(_ZEROS29.copy()),

            "partial_points": torch.from_numpy(pcd),               # [4096,3]
            "anchor_visible": torch.tensor([0], dtype=torch.uint8),
            "grasp_type_id":  torch.tensor([10], dtype=torch.uint8),

            "obj_path":  [str(folder / "combined.obj")],             # keep as str
            "obj_scale": torch.tensor([1.0], dtype=torch.float32),
            "obj_pose":  torch.from_numpy(_DEFAULT_POSE.copy()),
        }
        return sample

