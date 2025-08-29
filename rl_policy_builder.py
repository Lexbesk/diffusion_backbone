from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np

from modeling.policy.grasp_trajectory_denoiser import DexterousActor
from argparse import Namespace


DIFFUSION_CONFIG = {'train_data_dir': '/data/user_data/austinz/Robots/manipulation/zarr_datasets/dexterousact/train.zarr', 
        'eval_data_dir': ('/data/user_data/austinz/Robots/manipulation/zarr_datasets/dexterousact/train.zarr'), 
        'dataset': 'DexterousAct', 
        'num_workers': 4, 'batch_size': 16, 
        'batch_size_val': 8, 'chunk_size': 1, 'memory_limit': 8, 'base_log_dir': ('train_logs'), 
        'exp_log_dir': ('DexterousAct_debug'), 'run_log_dir': ('run_Aug24-B16-lv2bs4-Bval8-DT10-nhist8-nfuture32-K4'), 
        'checkpoint': '/data/user_data/austinz/Robots/manipulation/diffusion_backbone/train_logs/DexterousAct_debug/run_Aug24-B16-lv2bs4-Bval8-DT10-nhist4-nfuture4-K4/last.pth', 
        'val_freq': 1000, 'eval_only': False, 'eval_overfit': False, 'lr': 1e-06, 'lr_scheduler': 'constant', 'wd': 0.005, 
        'train_iters': 600000, 'use_compile': False, 'use_ema': False, 'lv2_batch_size': 4, 'model_type': 'dexterousactor', 
        'bimanual': False, 'keypose_only': True, 'pre_tokenize': True, 'custom_img_size': None, 'backbone': 'clip', 
        'output_level': 'res3', 'upsample': False, 'finetune_backbone': False, 'finetune_text_encoder': False, 
        'fps_subsampling_factor': 5, 'embedding_dim': 256, 'num_attn_heads': 8, 'num_vis_instr_attn_layers': 3, 
        'num_history': 1, 'num_shared_attn_layers': 30, 'workspace_normalizer_buffer': 0.04, 'relative_action': False, 
        'rotation_format': 'quat_wxyz', 'denoise_timesteps': 10, 'denoise_model': 'rectified_flow', 'visualize_denoising_steps': False, 
        'accurate_joint_pos': True, 'save_for_mujoco': False, 'test_mujoco': True, 'vis_freq': 10000000000, 'val_set_all_anchor': True, 
        'condition_on_grasp_type_id': True, 'guidance_weight': 1.5, 'sample_times': 1, 'nhist': 8, 'nfuture': 32, 'K': 4, 
        'log_dir': ('analogical_manipulation/train_logs/DexterousAct_debug/run_Aug24-B16-lv2bs4-Bval8-DT10-nhist8-nfuture32-K4'), 'local_rank': 0}

DIFFUSION_CONFIG = Namespace(**DIFFUSION_CONFIG)

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class DiffusionPlayer(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.model = DexterousActor(
            embedding_dim=DIFFUSION_CONFIG.embedding_dim,
            num_attn_heads=DIFFUSION_CONFIG.num_attn_heads,
            nhist=DIFFUSION_CONFIG.num_history,
            nhand=2 if DIFFUSION_CONFIG.bimanual else 1,
            num_shared_attn_layers=DIFFUSION_CONFIG.num_shared_attn_layers,
            relative=DIFFUSION_CONFIG.relative_action,
            rotation_format=DIFFUSION_CONFIG.rotation_format,
            denoise_timesteps=DIFFUSION_CONFIG.denoise_timesteps,
            denoise_model=DIFFUSION_CONFIG.denoise_model,
            lv2_batch_size=DIFFUSION_CONFIG.lv2_batch_size,
            visualize_denoising_steps=DIFFUSION_CONFIG.visualize_denoising_steps,
            accurate_joint_pos=DIFFUSION_CONFIG.accurate_joint_pos,
            guidance_weight=DIFFUSION_CONFIG.guidance_weight,
        )
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        # self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()


    def get_action(self, obs, is_deterministic = False):
        print(obs.shape, 'obs sum before preproc')
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        print(obs, 'obs after preproc')
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        # checkpoint = torch_ext.load_checkpoint(fn)
        model_dict = torch.load(
            DIFFUSION_CONFIG.checkpoint,
            map_location="cpu",
            weights_only=True
        )
        # self.model.load_state_dict(checkpoint['model'])
        msn, unxpct = self.model.load_state_dict(model_dict["weight"], strict=False)
        if msn:
            print(f"Missing keys (not found in checkpoint): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("All keys matched successfully!")
        # if self.normalize_input and 'running_mean_std' in checkpoint:
        #     self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = model_dict.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)
        self.is_rnn = False

    def reset(self):
        # self.init_rnn()
        pass