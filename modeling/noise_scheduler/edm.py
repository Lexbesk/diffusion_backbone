import math

import torch


class EDMScheduler:
    """
    Adapted code from:
    https://github.com/intuitive-robots/MoDE_Diffusion_Policy/blob/main/mode/models/edm_diffusion/score_wrappers.py
    """

    def __init__(
        self,
        noise_scheduler='exponential',
        sigma_data=0.5,
        sigma_min=0.001,
        sigma_max=80
    ):
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.noise_scheduler = noise_scheduler

    def set_timesteps(self, num_inference_steps, device='cpu'):
        if self.noise_scheduler == 'exponential':
            sigmas = get_sigmas_exponential(
                num_inference_steps,
                self.sigma_min, self.sigma_max,
                device
            )
        elif self.noise_scheduler == 'linear':
            sigmas = get_sigmas_linear(
                num_inference_steps,
                self.sigma_min, self.sigma_max,
                device
            )
        else:
            assert False, 'wrong noise scheduler for EDM'
        self.timesteps = sigmas

    def sample_noise_step(self, num_noise, device):
        return rand_log_logistic(
            shape=(num_noise,),
            loc=math.log(self.sigma_data),
            scale=0.5,
            min_value=self.sigma_min,
            max_value=self.sigma_max,
            device=device
        )

    def add_noise(self, original_samples, noise, timesteps):
        x = original_samples
        z1 = noise
        t = timesteps
        b = x.size(0)

        zt = x + z1 * t.view([b, *([1] * len(x.shape[1:]))])
        return zt.to(x.dtype)

    def get_scalings(self, sigma):
        """
        Compute the scalings for the denoising process.

        Args:
            sigma: The input sigma.
        Returns:
            The computed scalings for skip connections, output, and input.
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def step(self, model_output, timestep, sample):
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        i = (self.timesteps == timestep).flatten().nonzero()[0]
        t = t_fn(self.timesteps[i])
        if i == len(self.timesteps) - 1:
            t_next = torch.zeros_like(t)
        else:
            t_next = t_fn(self.timesteps[i + 1])
        h = t_next - t
        prev_sample = (
            (sigma_fn(t_next) / sigma_fn(t)) * sample
            - (-h).expm1() * model_output
        )

        return DummyClass(prev_sample=prev_sample)

    def prepare_target(self, noise, gt, noised_input, timesteps):
        c_skip, c_out, _ = self.get_scalings(timesteps)
        b = len(gt)
        c_skip = c_skip.view([b, *([1] * len(gt.shape[1:]))])
        c_out = c_out.view([b, *([1] * len(gt.shape[1:]))])
        return (gt - c_skip * noised_input) / c_out


def rand_log_logistic(shape, loc=0., scale=1.,
                      min_value=0., max_value=float('inf'),
                      device='cpu', dtype=torch.float32):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = (
        torch.rand(shape, device=device, dtype=torch.float64)
        * (max_cdf - min_cdf)
        + min_cdf
    )
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return sigmas


def get_sigmas_linear(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an linear noise schedule."""
    sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    return sigmas


class DummyClass:

    def __init__(self, prev_sample):
        self.prev_sample = prev_sample
