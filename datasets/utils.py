import blosc
import pickle

import einops
from pickle import UnpicklingError
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2 as tfv2
import torchvision.transforms.functional as transforms_f

from diffuser_actor.utils.utils import normalise_quat
from utils.pytorch3d_transforms import euler_angles_to_matrix


def loader(file):
    if str(file).endswith(".npy"):
        try:
            content = np.load(file, allow_pickle=True)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".dat"):
        try:
            with open(file, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".pkl"):
        try:
            with open(file, 'rb') as f:
                content = pickle.load(f)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    return None


class ColorAugmentation:
    """Color-wise augmentation"""
    def __init__(self, ratio = 0.5 ):
        self.transforms = tfv2.Compose([
            tfv2.RandomChoice([
                tfv2.GaussianBlur(9),
                tfv2.RandomAdjustSharpness(sharpness_factor=4)
            ]),
            tfv2.RandomAutocontrast(),
            tfv2.RandomEqualize(),
            tfv2.RandomPosterize(2)
        ])
        self.ratio = ratio

    def __call__(self, image, **kwargs):
        image = self.transforms(image)
        random_value = np.random.uniform()
        if(random_value > self.ratio):
            brightness = np.random.uniform(0.5, 1.5)
            image = tfv2.functional.adjust_brightness(
                image, brightness
            )
        return image


class Resize:
    """Resize and pad/crop the image and aligned point cloud."""

    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs):
        """Accept tensors as T, N, C, H, W."""
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Sample resize scale from continuous range
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize
        kwargs = {
            n: transforms_f.resize(
                arg,
                resized_size,
                transforms.InterpolationMode.NEAREST
            )
            for n, arg in kwargs.items()
        }

        # If resized image is smaller than original, pad it with a reflection
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = max(raw_w - resized_size[1], 0), max(
                raw_h - resized_size[0], 0
            )
            kwargs = {
                n: transforms_f.pad(
                    arg,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        # If resized image is larger than original, crop it
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )
        kwargs = {
            n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()
        }

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


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

def calibration_augmentation(pcd, roll_range, pitch_range, yaw_range,
                             translate_range):
    """Apply random rotation and translation to the given pcd

    Args:
        pcd: A tensor of shape (B, ..., 3)
        roll_range: An integer or a list indicates the (min, max) of the roll
        pitch_range: An integer or a list indicates the (min, max) of the pitch
        yaw_range: An integer or a list indicates the (min, max) of the yaw
        translate_range: An integer or a list indicates the (min, max) of the
            tranlation
    
    Returns:
        pcd: A tensor of shape (B, ...., 3)
    """
    if isinstance(roll_range, int):
        roll_range = [min(-roll_range, roll_range),
                      max(-roll_range, roll_range)]

    if isinstance(pitch_range, int):
        pitch_range = [min(-pitch_range, pitch_range),
                       max(-pitch_range, pitch_range)]

    if isinstance(yaw_range, int):
        yaw_range = [min(-yaw_range, yaw_range),
                     max(-yaw_range, yaw_range)]

    if isinstance(translate_range, int):
        translate_range = [min(-translate_range, translate_range),
                           max(-translate_range, translate_range)]
    
    bs = pcd.shape[0]
    euler_angles = torch.rand((bs, 3), dtype=pcd.dtype, device=pcd.device)
    euler_angles *= torch.Tensor([
        roll_range[1] - roll_range[0],
        pitch_range[1] - pitch_range[0],
        yaw_range[1] - yaw_range[0]
    ], device=pcd.device)
    euler_angles += torch.Tensor([
        roll_range[0], pitch_range[0], yaw_range[0]
    ], device=pcd.device)

    R = euler_angles_to_matrix(euler_angles, convention='XYZ')
    T = torch.rand((bs, 1, 3), dtype=pcd.dtype, device=pcd.device)
    T = T * (translate_range[1] - translate_range[0])
    T = T + translate_range[0]

    flat_pcd = pcd.reshape(bs, -1, 3)
    flat_pcd = flat_pcd @ R + T
    pcd = flat_pcd.reshape(pcd.shape)

    return pcd
