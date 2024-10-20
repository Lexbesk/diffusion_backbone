from typing import Tuple, Union, Optional
from dataclasses import dataclass

import torch
import numpy as np
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


class RFSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class RFScheduler(SchedulerMixin, ConfigMixin):
    """The code is heavily based on:
        https://github.com/cloneofsimo/minRF/blob/main/advanced/main_t2i.py
    
    We refactor the scheduler to be more compatible with the convention of
    diffusers.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
    ):
        """
        Args:
            num_train_timesteps (`int`, defaults to 1000):
                The number of diffusion steps to train the model.
            timestep_spacing (`str`, defaults to `"leading"`):
                The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
                Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
            steps_offset (`int`, defaults to 0):
                An offset added to the inference steps, as required by some model families
        """
        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy()
        )

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """We use the same convetion of timesteps as DDPMScheduler

        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )
        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:

        x = original_samples
        z1 = noise
        # DDPM uses timesteps from the range of 0, ..., T - 1
        # RectifiedFlow seems to uses timesteps from the range of
        # 0, 1/ΔT, 2/ΔT, ..., 1, where `0` denotes clean sample and `1` denotes
        # pure noise
        t = (1 + timesteps).to(x.device).float() / self.config.num_train_timesteps
        assert ((t >= 0) & (t <= 1)).all()
        b = x.size(0)
        
        # Interpolate between Z0 and Z1 (Eqn 1 in the paper)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        zt = (1 - texp) * x + texp * z1

        # make t, zt into same dtype as x
        zt = zt.to(x.dtype)

        return zt

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[RFSchedulerOutput, Tuple]:
        zt = sample
        vc = model_output

        # DDPM uses timesteps from the range of 0, ..., T - 1
        # RectifiedFlow seems to uses timesteps from the range of
        # 0, 1/ΔT, 2/ΔT, ..., 1, where `0` denotes clean sample and `1` denotes
        # pure noise
        dt_to_0 = float(timestep + 1) / self.config.num_train_timesteps
        pred_original_sample = zt - dt_to_0 * vc # z_0

        prev_t = self.previous_timestep(timestep)
        if prev_t < 0:
            # timestep seems to be the last step
            pred_prev_sample = pred_original_sample
        else:
            dt = float(timestep - prev_t) / self.config.num_train_timesteps
            pred_prev_sample = zt - dt * vc # z_t'

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )
        return RFSchedulerOutput(prev_sample=pred_prev_sample,
                                 pred_original_sample=pred_original_sample)

    def previous_timestep(self, timestep):
        num_inference_steps = (
            self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
        )
        prev_t = timestep - self.config.num_train_timesteps // num_inference_steps

        return prev_t