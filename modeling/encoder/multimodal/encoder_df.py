import einops
import torch
from torch import nn
from torch.nn import functional as F

from ...utils.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from ...utils.layers import AttentionModule
from .encoder3d import Encoder as BaseEncoder


class Encoder(BaseEncoder):

    def __init__(self,
                 backbone="clip",
                 output_level="res3",
                 upsample=False,
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False):
        super().__init__(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            embedding_dim=embedding_dim,
            nhist=nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder,
        )
        # Cross-view attention layers
        dim_ = 288  # divisible by 6, 8, 9
        self.cv_proj = nn.Linear(2048, dim_)
        self.cv_attention = AttentionModule(
            num_layers=4, d_model=dim_, pre_norm=True,
            rotary_pe=True, is_self=True
        )
        self.cv_unproj = nn.Linear(dim_, 2048)
        self.cv_relative_pe_layer = RotaryPositionEncoding3D(dim_)
        self.pos_embed_2d = SinusoidalPosEmb(embedding_dim)
        self.rgb2d_proj = nn.Linear(2048, embedding_dim)

    def encode_clip(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W), rgb obs of 3D cameras
            - rgb2d: (B, ncam2d, 3, H, W), rgb obs of 2D cameras
            - pcd: (B, ncam3d, 3, H, W)
            - text: [str] of len=B, text instruction

        Returns:
            - rgb3d_feats: (B, Np, F)
            - rgb2d_feats: (B, ncam2d, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        # Encode language
        instruction = self.text_encoder(text)
        instr_feats = self.instruction_encoder(instruction)

        # 3D camera features
        num_cameras = rgb3d.shape[1]
        # Pass each view independently through backbone
        rgb3d = einops.rearrange(rgb3d, "bt ncam c h w -> (bt ncam) c h w")
        if self.upsample:
            rgb3d = F.interpolate(rgb3d, (256, 256), mode='bilinear')
        rgb3d = self.normalize(rgb3d)
        rgb3d_feats = self.backbone(rgb3d)
        # Attention across views
        rgb3d_feats["res5"] = self._cross_view_attn3d(rgb3d_feats["res5"], pcd)
        # Pass visual features through feature pyramid network
        rgb3d_feats = self.feature_pyramid(rgb3d_feats)[self.output_level]
        feat_h, feat_w = rgb3d_feats.shape[-2:]
        # Merge different cameras
        rgb3d_feats = einops.rearrange(
            rgb3d_feats,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        # Attention from vision to language
        rgb3d_feats = self.vl_attention(seq1=rgb3d_feats, seq2=instr_feats)[-1]

        # Point cloud
        num_cameras = pcd.shape[1]
        # Interpolate point cloud to get the corresponding locations
        pcd = F.interpolate(
            einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
            (feat_h, feat_w),
            mode='bilinear'
        )
        # Merge different cameras
        pcd = einops.rearrange(
            pcd,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )

        # 2D camera features
        rgb2d_feats = None
        if rgb2d is not None:
            num_cameras = rgb2d.shape[1]
            # Pass each view independently through backbone
            rgb2d = einops.rearrange(rgb2d, "bt ncam c h w -> (bt ncam) c h w")
            rgb2d = self.normalize(rgb2d)
            rgb2d_feats = self.backbone(rgb2d)["res5"]
            _, _, h, w = rgb2d_feats.shape
            rgb2d_feats = einops.rearrange(
                rgb2d_feats,
                "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            rgb2d_feats = self.rgb2d_proj(rgb2d_feats)
            # Attention from vision to language
            rgb2d_feats = self.vl_attention(seq1=rgb2d_feats, seq2=instr_feats)[-1]
            # Unsqueeze to add embeddings
            rgb2d_feats = einops.rearrange(
                rgb2d_feats,
                "bt (ncam h w) c -> bt ncam c h w", ncam=num_cameras, h=h, w=w
            )
            # Add camera embeddings
            rgb2d_feats = (
                rgb2d_feats
                + self.camera_ids.weight[None, :num_cameras, :, None, None]
            )
            # Add 2D pos embs
            b, nc, _, h, w = rgb2d_feats.shape
            _2d_pos = self.pos_embed_2d(
                torch.arange(0, h * w, device=rgb2d_feats.device)
            ).reshape(h, w, -1)
            _2d_pos = einops.rearrange(_2d_pos, "h w c -> c h w")
            rgb2d_feats = rgb2d_feats + _2d_pos[None, None]
            rgb2d_feats = einops.rearrange(rgb2d_feats, "b n c h w -> b (n h w) c")

        return rgb3d_feats, rgb2d_feats, pcd, instr_feats

    def _cross_view_attn3d(self, feats3d, pcd):
        num_cameras = pcd.shape[1]
        # Interpolate point cloud
        pcd = F.interpolate(
            einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
            (feats3d.size(-2), feats3d.size(-1)),
            mode='bilinear'
        )
        pcd = einops.rearrange(
            pcd,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        # 3D attention
        _, c, h, w = feats3d.shape
        feats3d = einops.rearrange(
            feats3d,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        feats3d = self.cv_proj(feats3d)
        rel_pos = self.cv_relative_pe_layer(pcd)
        feats3d = self.cv_attention(
            seq1=feats3d,
            seq2=feats3d,
            seq1_pos=rel_pos,
            seq2_pos=rel_pos
        )[-1]
        feats3d = self.cv_unproj(feats3d)
        # Return original shape
        return einops.rearrange(
            feats3d,
            "bt (ncam h w) c-> (bt ncam) c h w", c=c, h=h, w=w
        ).contiguous()
