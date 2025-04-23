import einops
import torch

from ..utils.layers import DoubleCrossSelfAttentionModule
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

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 rotary_pe=True):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            rotary_pe=rotary_pe
        )

        # Shared attention layers
        self.self_attn = DoubleCrossSelfAttentionModule(
            num_layers=4,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe_0=False,
            rotary_pe_1=rotary_pe,
            use_adaln=True
        )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_self_attn = DoubleCrossSelfAttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe_0=False,
            rotary_pe_1=rotary_pe,
            use_adaln=True
        )

        # 2. Position
        self.position_self_attn = DoubleCrossSelfAttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe_0=False,
            rotary_pe_1=rotary_pe,
            use_adaln=True
        )

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos):
        """
        Arguments:
            traj_feats: (B, trajectory_length, nhand, F)
            trajectory: (B, trajectory_length, nhand, 3+6+X)
            timesteps: (B, 1)
            rgb3d_feats: (B, N, F)
            rgb3d_pos: (B, N, 3)
            rgb2d_feats: (B, N2d, F)
            rgb2d_pos: (B, N2d, 3)
            instr_feats: (B, L, F)
            instr_pos: (B, L, 3)
            proprio_feats: (B, nhist*nhand, F)
            fps_scene_feats: (B, M, F), M < N
            fps_scene_pos: (B, M, 3)

        Returns:
            list of (B, trajectory_length, nhand, 3+6+X)
        """
        _, traj_len, nhand, _ = trajectory.shape

        # Trajectory features
        if nhand > 1:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(len(traj_feats), 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )[-1]
        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]

        # Denoising timesteps' embeddings
        time_embs = self.encode_denoising_timestep(
            timesteps, proprio_feats
        )

        # Positional embeddings
        rel_traj_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
            traj_xyz, traj_feats,
            rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
            timesteps, proprio_feats,
            fps_scene_feats, fps_scene_pos,
            instr_feats, instr_pos
        )

        # Cross attention from gripper to full context
        traj_feats = self.cross_attn(
            seq1=traj_feats,
            seq2=rgb3d_feats,
            seq1_pos=rel_traj_pos,
            seq2_pos=rel_scene_pos,
            ada_sgnl=time_embs
        )[-1]

        # Cross attention to language and then self with scene
        features = self.get_sa_feature_sequence(
            traj_feats, fps_scene_feats,
            rgb3d_feats, rgb2d_feats, instr_feats
        )
        traj_feats = self.self_attn(
            seq=features,
            seq0=instr_feats,
            seq_pos_1=rel_pos,
            ada_sgnl=time_embs
        )[-1]

        # Rotation head
        rotation = self.predict_rot(
            features, instr_feats, rel_pos, time_embs, traj_len
        )

        # Position head
        position, position_features = self.predict_pos(
            features, instr_feats, rel_pos, time_embs, traj_len
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), -1)
                 .unflatten(1, (traj_len, nhand))
        ]

    def predict_pos(self, features, instr_feats, rel_pos, time_embs, traj_len):
        position_features = self.position_self_attn(
            seq=features,
            seq0=instr_feats,
            seq_pos_1=rel_pos,
            ada_sgnl=time_embs
        )[-1]
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, instr_feats, rel_pos, time_embs, traj_len):
        rotation_features = self.rotation_self_attn(
            seq=features,
            seq0=instr_feats,
            seq_pos_1=rel_pos,
            ada_sgnl=time_embs
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation
