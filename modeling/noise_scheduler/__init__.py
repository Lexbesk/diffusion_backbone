from .ddpm import DDPMScheduler
from .edm import EDMScheduler
from .rectified_flow import RFScheduler


def fetch_schedulers(denoise_model, denoise_timesteps):
    if denoise_model == "ddpm":
        position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
    elif denoise_model in ("rectified_flow", "moritz", "pi0", "flow_uniform"):
        noise_sampler_config = {"mean": 0, "std": 1.5}
        if denoise_model == "moritz":
            noise_sampler_config = {"mean": 0, "std": 1.0}
        samplers = {
            "rectified_flow": "logit_normal",
            "moritz": "logit_normal",
            "pi0": "pi0",
            "flow_uniform": "uniform"
        }
        position_noise_scheduler = RFScheduler(
            noise_sampler=samplers[denoise_model],
            noise_sampler_config=noise_sampler_config
        )
        rotation_noise_scheduler = RFScheduler(
            noise_sampler=samplers[denoise_model],
            noise_sampler_config=noise_sampler_config
        )
    elif denoise_model in ('edm', 'beso'):
        if denoise_model == 'edm':
            noise_scheduler = 'exponential'
            sigma_data = 0.5
            sigma_min = 0.005
            sigma_max = 1.0
        elif denoise_model == 'beso':
            noise_scheduler = 'exponential'
            sigma_data = 0.5
            sigma_min = 0.001
            sigma_max = 80.0
        position_noise_scheduler = EDMScheduler(
            noise_scheduler=noise_scheduler,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
        rotation_noise_scheduler = EDMScheduler(
            noise_scheduler=noise_scheduler,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
    return position_noise_scheduler, rotation_noise_scheduler
