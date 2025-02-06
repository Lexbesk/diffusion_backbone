import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np

from .denoise_actor import DenoiseActor as BaseActor


class DenoiseActor(BaseActor):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 num_attn_heads=8,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 rotation_parametrization='6D',
                 quaternion_format='xyzw',
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 nhist=3,
                 relative=False):
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            num_attn_heads=num_attn_heads,
            use_instruction=use_instruction,
            fps_subsampling_factor=fps_subsampling_factor,
            rotation_parametrization=rotation_parametrization,
            quaternion_format=quaternion_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            nhist=nhist,
            relative=relative
        )
        self.fg_emb = nn.Embedding(1, embedding_dim)

    def encode_inputs(self, visible_rgb, visible_pcd, seg_mask,
                      curr_gripper):
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(
            visible_rgb, visible_pcd
        )
        # Keep only low-res scale
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0],
            "b ncam c h w -> b (ncam h w) c"
        )
        context = pcd_pyramid[0]

        if self._mae and self.training:
            drop_ratio = np.random.uniform(0.3, 0.6)
            # drop_ratio = np.random.uniform(0.2, 0.4)
            keep_num = int(context_feats.shape[1] * (1 - drop_ratio))
            device = context_feats.device
            keep_inds = [
                torch.randperm(context_feats.shape[1], device=device)[:keep_num]
                for _ in range(context_feats.shape[0])
            ]
            keep_inds = torch.stack(keep_inds, dim=0)
            context_feats = torch.gather(
                context_feats, 1,
                keep_inds.unsqueeze(-1).repeat(1, 1, context_feats.shape[-1])
            )
            context = torch.gather(
                context, 1,
                keep_inds.unsqueeze(-1).repeat(1, 1, context.shape[-1])
            )

        # Add binary indicator for tokens of the target object
        feat_h, feat_w = rgb_feats_pyramid[0].shape[-2:]
        seg_mask = F.interpolate(
            seg_mask.float(),
            (feat_h, feat_w),
            mode='nearest'
        ).reshape(len(seg_mask), -1)[..., None]  # B (nc*h*w) 1
        emb_mask = seg_mask * self.fg_emb.weight[None]  # B (nc*h*w) F
        context_feats = context_feats + emb_mask

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            self.encoder.relative_pe_layer(context)
        )

        return (
            context_feats, context,  # contextualized visual features
            emb_mask,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats, fps_pos  # sampled visual features
        )
