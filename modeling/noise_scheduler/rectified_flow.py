import torch
import numpy as np


class RFScheduler:
    """
    Code based on:
    https://github.com/cloneofsimo/minRF/blob/main/advanced/main_t2i.py
    """

    def __init__(self, noise_sampler="logit_normal", noise_sampler_config={}):
        self.noise_sampler = noise_sampler
        self.noise_sampler_config = noise_sampler_config

    def set_timesteps(self, num_inference_steps, device='cpu'):
        self.timesteps = torch.from_numpy(np.linspace(
            0, 1, num_inference_steps,
            dtype=np.float32
        )[::-1][:-1].copy()).to(device)

    def sample_noise_step(self, num_noise, device):
        if self.noise_sampler == "uniform":
            samples = torch.full((num_noise,), 0, dtype=torch.float32,
                                 device=device)
            timesteps = samples.uniform_()
        elif self.noise_sampler == "logit_normal":
            samples = torch.full((num_noise,), 0, dtype=torch.float32,
                                 device=device)
            samples = samples.normal_(
                mean=self.noise_sampler_config['mean'],
                std=self.noise_sampler_config['std']
            )
            timesteps = torch.sigmoid(samples)
        elif self.noise_sampler == "pi0":
            alpha, beta = 1.5, 1.0
            timesteps = torch.distributions.Beta(alpha, beta).sample((num_noise,)).to(device).clamp(max=0.999)
        else:
            raise NotImplementedError(f"{self.noise_sampler} not implemented")

        return timesteps

    def add_noise(self, original_samples, noise, timesteps):
        x = original_samples
        z1 = noise
        t = timesteps
        b = x.size(0)
        
        # Interpolate between Z0 and Z1
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        zt = (1 - texp) * x + texp * z1
        return zt.to(x.dtype)

    def get_scalings(self, sigma):
        return (
            torch.zeros_like(sigma),
            torch.ones_like(sigma),
            torch.ones_like(sigma)
        )

    def step(self, model_output, timestep, sample):
        zt = sample
        vc = model_output

        curr_ind = (self.timesteps == timestep).flatten().nonzero()[0]
        if curr_ind == len(self.timesteps) - 1:
            prev_t = 0
        else:
            prev_t = self.timesteps[curr_ind + 1].to(vc.device)
        dt = timestep - prev_t
        pred_prev_sample = zt - dt * vc # z_t'

        return DummyClass(prev_sample=pred_prev_sample)

    def prepare_target(self, noise, gt, noised_input, timesteps):
        return noise - gt


class DummyClass:

    def __init__(self, prev_sample):
        self.prev_sample = prev_sample
