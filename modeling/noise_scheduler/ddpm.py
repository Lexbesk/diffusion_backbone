import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as BaseScheduler


class DDPMScheduler(BaseScheduler):
    """A wrapper class for DDPM which handles sampling noise for training."""

    def sample_noise_step(self, num_noise, device):
        timesteps = torch.randint(
            0,
            self.config.num_train_timesteps,
            (num_noise,), device=device
        ).long()

        return timesteps

    def get_scalings(self, sigma):
        return (
            torch.zeros_like(sigma),
            torch.ones_like(sigma),
            torch.ones_like(sigma)
        )

    def prepare_target(self, noise, gt, noised_input, timesteps):
        if self.config.prediction_type == "epsilon":
            return noise
        elif self.config.prediction_type == "sample":
            return gt
        else:
            raise NotImplementedError
