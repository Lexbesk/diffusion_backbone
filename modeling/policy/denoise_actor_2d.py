import torch

from ..encoder.multimodal.encoder2d import Encoder
from ..utils.position_encodings import SinusoidalPosEmb

from .base_denoise_actor import DenoiseActor as BaseDenoiseActor
from .base_denoise_actor import TransformerHead as BaseTransformerHead


class DenoiseActor(BaseDenoiseActor):

    def __init__(self,
                 # Encoder arguments
                 backbone="clip",
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

        # Vision-language encoder, runs only once
        self.encoder = Encoder(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads
        )


class TransformerHead(BaseTransformerHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 rotary_pe=False):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            rotary_pe=False
        )
        # Positional embeddings
        self.pos_embed_2d = SinusoidalPosEmb(embedding_dim)

    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos
    ):
        _traj_pos = torch.zeros_like(traj_feats)
        _scene_pos = self.pos_embed_2d(
            torch.arange(0, fps_scene_feats.size(1), device=traj_feats.device)
        )[None].repeat(traj_feats.size(0), 1, 1)
        _instr_pos = torch.zeros_like(instr_feats)
        _pos = torch.cat([_traj_pos, _scene_pos, _instr_pos], 1)
        if rgb2d_feats is not None:
            _2d_pos = self.pos_embed_2d(
                torch.arange(0, rgb2d_feats.size(1), device=traj_feats.device)
            )[None].repeat(traj_feats.size(0), 1, 1)
            _pos = torch.cat([_pos, _2d_pos], 1)
        return _pos

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        features = torch.cat([traj_feats, fps_scene_feats, instr_feats], 1)
        if rgb2d_feats is not None:
            features = torch.cat([features, rgb2d_feats])
        return features
