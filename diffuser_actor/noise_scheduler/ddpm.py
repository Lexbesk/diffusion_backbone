import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as BaseScheduler


class DDPMScheduler(BaseScheduler):
    """A wrapper class for DDPM which handles sampling noise for training.
    """

    def sample_noise_step(self, num_noise, device):
        timesteps = torch.randint(
            0,
            self.config.num_train_timesteps,
            (num_noise,), device=device
        ).long()

        return timesteps

    def prepare_target(self, noise, gt):
        if self.config.prediction_type == "epsilon":
            return noise
        elif self.config.prediction_type == "sample":
            return gt
        else:
            raise NotImplementedError
