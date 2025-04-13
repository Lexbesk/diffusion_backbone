import torch

from .denoise_actor_3d import DenoiseActor as BaseDenoiseActor
from .denoise_actor_3d import TransformerHead as BaseTransformerHead


class DenoiseActor(BaseDenoiseActor):

    def __init__(self,
                 # Encoder arguments
                 backbone="clip",
                 output_level="res3",
                 upsample=False,
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=9,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 relative=False,
                 quaternion_format='xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm"):
        super().__init__(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            nhand=nhand,
            relative=relative,
            quaternion_format=quaternion_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads
        )


class TransformerHead(BaseTransformerHead):

    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos
    ):
        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(rgb3d_pos)
        rel_fps_pos = self.relative_pe_layer(fps_scene_pos)
        rel_instr_pos = self.relative_pe_layer(instr_pos)
        rel_pos = torch.cat([rel_traj_pos, rel_fps_pos, rel_instr_pos], 1)
        # Also use PE for 2D cameras if available
        if rgb2d_feats is not None:
            rel_2d_pos = self.relative_pe_layer(rgb2d_pos)
            rel_pos = torch.cat([rel_pos, rel_2d_pos], 1)
        return rel_traj_pos, rel_scene_pos, rel_pos

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        features = torch.cat([traj_feats, fps_scene_feats, instr_feats], 1)
        if rgb2d_feats is not None:
            features = torch.cat([features, rgb2d_feats], 1)
        return features
