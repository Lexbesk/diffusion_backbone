import logging

import torch
from torch.nn import functional as F

from modeling.policy import fetch_model_class
from modeling.encoder.text import fetch_tokenizers
from utils.pytorch3d_transforms import relative_to_absolute
from online_evaluation_calvin.utils_with_calvin import convert_action


logger = logging.getLogger(__name__)


class Model:

    def __init__(self, args):
        self.args = args
        self.policy = self.get_policy()
        self.reset()

    def get_policy(self):
        """Initialize the model."""
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        model_class = fetch_model_class(self.args.model_type)
        return model_class(
            backbone=self.args.backbone,
            num_vis_instr_attn_layers=self.args.num_vis_instr_attn_layers,
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            nhist=self.args.num_history,
            nhand=1,
            relative=self.args.relative_action,
            quaternion_format=self.args.quaternion_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model
        )

    def reset(self):
        """Set model to evaluation mode."""
        device = self.args.device
        self.policy.eval()
        self.policy = self.policy.to(device)

    def load_pretrained_weights(self):
        state_dict = torch.load(self.args.checkpoint, map_location="cpu")["weight"]
        model_weights = {}
        for key in state_dict:
            _key = key[7:]
            model_weights[_key] = state_dict[key]
        print(f'Loading weights from {self.args.checkpoint}')
        self.policy.load_state_dict(model_weights)

    @torch.no_grad()
    def step(self, obs, instruction):
        """
        Args:
            obs: {
                rgb_obs: {rgb_static: (H, W, 3), rgb_gripper: (h, w, 3)},
                pcd_obs: {pcd_static: (H, W, 3)},
                proprio: (nhist, 8)
            }
            instruction: str, the instruction of the task

        Returns:
            action: predicted action
        """
        device = self.args.device

        # Process inputs
        trajectory_mask = torch.full([self.args.pred_len], False).to(device)
        trajectory_mask = trajectory_mask[None, :, None]

        # Static camera
        rgb3d = obs["rgb_obs"]["rgb_static"].transpose(2, 0, 1)[None, None]
        rgb3d = torch.from_numpy(rgb3d).to(device).float()
        rgb3d = rgb3d[..., 20:180, 20:180]

        # Merge wrist camera
        h, w = rgb3d.shape[-2:]
        rgb2d = obs["rgb_obs"]["rgb_gripper"].transpose(2, 0, 1)[None]
        rgb2d = torch.from_numpy(rgb2d).to(device).float()
        rgb2d = F.interpolate(rgb2d, (h, w), mode='bilinear')[:, None]
        rgbs = torch.cat((rgb3d, rgb2d), 1)

        # Static camera point cloud
        pcds = obs["pcd_obs"]["pcd_static"].transpose(2, 0, 1)[None, None]
        pcds = torch.from_numpy(pcds).to(device).float()
        pcds = pcds[..., 20:180, 20:180]

        # Merge wrist camera point cloud
        h, w = pcds.shape[-2:]
        pcd2d = obs["pcd_obs"]["pcd_gripper"].transpose(2, 0, 1)[None]
        pcd2d = torch.from_numpy(pcd2d).to(device).float()
        pcd2d = F.interpolate(pcd2d, (h, w), mode='bilinear')[:, None]
        pcds = torch.cat((pcds, pcd2d), 1)

        # Proprioception
        proprio = torch.from_numpy(obs["proprio"]).to(device).float()
        proprio = proprio[None, :self.args.num_history, None]

        # Forward pass
        trajectory = self.policy(
            None,
            trajectory_mask,  # (1, T, 1)
            rgbs,  # (1, 2, 3, 160, 160)
            None,  # (1, 1, 3, 84, 84)
            pcds,  # (1, 2, 3, 160, 160)
            self.tokenizer([instruction,]).cuda(non_blocking=True),
            proprio[..., :7].float(),  # (1, nhist, 1, 7)
            run_inference=True
        )  # (1, T, 1, 8)
        trajectory = trajectory.view(1, self.args.pred_len, 8)  # (1, T, 8)

        # Back to absolute actions
        if bool(self.args.relative_action):
            trajectory = relative_to_absolute(trajectory, proprio[:, [-1], 0])

        # Convert quaternion to Euler angles
        trajectory = convert_action(trajectory)

        return trajectory


def create_model(args):
    model = Model(args)
    model.load_pretrained_weights()
    return model
