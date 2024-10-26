import math
from typing import Tuple, Union, Optional, Dict, Any
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
        timestep_spacing: str = "linspace",
        noise_sampler: str = "log_normal",
        noise_sampler_config: Dict[str, Any] = {},
    ):
        """
        Args:
            num_train_timesteps (`int`, defaults to 1000):
                The number of diffusion steps to train the model.
            timestep_spacing (`str`, defaults to `"leading"`):
                The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
                Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        """
        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(
            np.linspace(0, 1, num_train_timesteps)[::-1].copy()
        )[:-1].float()

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
            timesteps = np.linspace(0, 1, num_inference_steps,
                                    dtype=np.float32)[::-1][:-1].copy()
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def sample_noise_step(self, num_noise, device):
        if self.config.noise_sampler == "uniform":
            # Might be buggy, do we need to consider 1?
            samples = torch.full((num_noise,), 0, dtype=torch.float,
                                 device=device)
            timesteps = samples.uniform_()
        elif self.config.noise_sampler == "logit_normal":
            samples = torch.full((num_noise,), 0, dtype=torch.float,
                                 device=device)
            samples = samples.normal_(
                mean=self.config.noise_sampler_config['mean'],
                std=self.config.noise_sampler_config['std']
            )
            timesteps = torch.sigmoid(samples)
        elif self.config.noise_sampler == "stratified":
            # Stratified sampling
            b = num_noise
            quantiles = torch.linspace(0, 1, b + 1).to(device)
            z = quantiles[:-1] + torch.rand((b,)).to(device) / b
            z = torch.erfinv(2 * z - 1) * math.sqrt(2)
            timesteps = torch.sigmoid(z)  # Stratified sigmoid
        else:
            raise NotImplementedError(f"sampler: {self.noise_sampler} not implemented")

        return timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:

        x = original_samples
        z1 = noise
        t = timesteps
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
        timestep: float,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[RFSchedulerOutput, Tuple]:
        zt = sample
        vc = model_output
        dt_to_0 = timestep.to(vc.device)
        pred_original_sample = zt - dt_to_0 * vc # z_0

        curr_ind = (self.timesteps == timestep).flatten().nonzero()[0]
        if curr_ind == self.timesteps.shape[0] - 1:
            # timestep seems to be the last step
            pred_prev_sample = pred_original_sample
        else:
            prev_t = self.timesteps[curr_ind + 1].to(vc.device)
            dt = timestep - prev_t
            pred_prev_sample = zt - dt * vc # z_t'

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )
        return RFSchedulerOutput(prev_sample=pred_prev_sample,
                                 pred_original_sample=pred_original_sample)
