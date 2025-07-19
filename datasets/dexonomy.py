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

