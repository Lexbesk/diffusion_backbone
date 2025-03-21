import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import torch
import zarr
from zarr.storage import DirectoryStore
from zarr import LRUStoreCache

from modeling.utils.utils import normalise_quat
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


class TrajectoryInterpolator:
    """Interpolate a trajectory to have fixed length."""

    def __init__(self, use=False, interpolation_length=50):
        self._use = use
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        if not self._use:
            return trajectory
        trajectory = trajectory.numpy()
        # Calculate the current number of steps
        old_num_steps = len(trajectory)

        # Create a 1D array for the old and new steps
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Interpolate each dimension separately
        # resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        # for i in range(trajectory.shape[1]):
        #     if i == (trajectory.shape[1] - 1):  # gripper opening
        #         interpolator = interp1d(old_steps, trajectory[:, i])
        #     else:
        #         interpolator = CubicSpline(old_steps, trajectory[:, i])

        #     resampled[:, i] = interpolator(new_steps)
        resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        interpolator = CubicSpline(old_steps, trajectory[:, :-1])
        resampled[:, :-1] = interpolator(new_steps)
        last_interpolator = interp1d(old_steps, trajectory[:, -1])
        resampled[:, -1] = last_interpolator(new_steps)

        resampled = torch.tensor(resampled)
        if trajectory.shape[1] == 8:
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        elif trajectory.shape[1] == 16:
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
            resampled[:, 11:15] = normalise_quat(resampled[:, 11:15])
        return resampled


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
