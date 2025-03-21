import numpy as np
import torch
import zarr
from zarr.storage import DirectoryStore
from zarr import LRUStoreCache

import utils.pytorch3d_transforms as pytorch3d_transforms


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.as_tensor(x)


def read_zarr_with_cache(fname, mem_gb=16):
    # Configure the underlying store
    store = DirectoryStore(fname)

    # Wrap the store with a cache
    cached_store = LRUStoreCache(store, max_size=mem_gb * 2**30)  # GB cache

    # Open Zarr file with caching
    return zarr.open_group(cached_store, mode="r")


def to_relative_action(actions, anchor_actions, qform='xyzw'):
    assert actions.shape[-1] == 8

    rel_pos = actions[..., :3] - anchor_actions[..., :3]

    if qform == 'xyzw':
        # pytorch3d takes wxyz quaternion, the input is xyzw
        rel_orn = pytorch3d_transforms.quaternion_multiply(
            actions[..., [6, 3, 4, 5]],
            pytorch3d_transforms.quaternion_invert(anchor_actions[..., [6,3,4,5]])
        )[..., [1, 2, 3, 0]]
    elif qform == 'wxyz':
        # pytorch3d takes wxyz quaternion, the input is xyzw
        rel_orn = pytorch3d_transforms.quaternion_multiply(
            actions[..., 3:7],
            pytorch3d_transforms.quaternion_invert(anchor_actions[..., 3:7])
        )
    else:
        assert False

    gripper = actions[..., -1:]
    rel_actions = torch.concat([rel_pos, rel_orn, gripper], dim=-1)

    return rel_actions
