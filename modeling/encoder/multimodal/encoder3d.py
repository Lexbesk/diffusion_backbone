import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation

from ...utils.position_encodings import RotaryPositionEncoding3D
from ...utils.layers import AttentionModule
from ..vision.fpn import EfficientFeaturePyramidNetwork
from .base_encoder import Encoder as BaseEncoder


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
            embedding_dim=embedding_dim,
            nhist=nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

        # Postprocess scene features
        if self._backbone_name == 'clip':
            self.output_level = output_level
            self.upsample = upsample
            self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048],
                embedding_dim, output_level=output_level
            )
            self.rgb2d_proj = nn.Conv2d(2048, embedding_dim, 1)
        else:
            self.inner_block = Conv2dNormActivation(
                768, embedding_dim, kernel_size=1, padding=0,
                norm_layer=None, activation_layer=None
            )

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Proprioception learnable encoding if 3D is used
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False
        )

        # Camera IDs for the 2D cameras
        self.camera_ids = nn.Embedding(2, embedding_dim)

    def encode_proprio(self, proprio, context_feats, context_pos):
        """
        Compute proprioception features and positional embeddings.

        Args:
            - proprio: (B, nhist, 3+)
            - context_feats: (B, npt, C)
            - context_pos: (B, npt, 3)

        Returns:
            - gripper_feats: (B, nhist, F)
        """
        # Learnable embedding for proprioception
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).repeat(
            len(proprio), 1, 1
        )

        # # Project rotation features
        # gripper_rot_feats = self.curr_gripper_rot_proj(gripper_feats[..., 3:9])
        # gripper_feats = gripper_feats + gripper_rot_feats

        # Rotary positional encoding
        proprio_pos = self.relative_pe_layer(proprio[..., :3])
        context_pos = self.relative_pe_layer(context_pos)

        # Attention to scene tokens
        proprio_feats = self.gripper_context_head(
            proprio_feats, context_feats,
            seq1_pos=proprio_pos, seq2_pos=context_pos
        )[-1]

        return proprio_feats

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
            rgb2d_feats = F.adaptive_avg_pool2d(rgb2d_feats, 1)
            rgb2d_feats = self.rgb2d_proj(rgb2d_feats).squeeze(-1).squeeze(-1)
            rgb2d_feats = einops.rearrange(
                rgb2d_feats,
                "(bt ncam) c -> bt ncam c", ncam=num_cameras
            )
            # Attention from vision to language
            rgb2d_feats = self.vl_attention(seq1=rgb2d_feats, seq2=instr_feats)[-1]
            # Add camera embeddings
            rgb2d_feats = rgb2d_feats + self.camera_ids.weight[None, :num_cameras]

        return rgb3d_feats, rgb2d_feats, pcd, instr_feats

    def run_fps(self, features, pos):
        # features (B, Np, F)
        # context_pos (B, Np, 3)
        # outputs of analogous shape, with smaller Np
        if self.fps_subsampling_factor == 1:
            return features, pos

        bs, npts, ch = features.shape
        sampled_inds = density_based_sampler(features, self.fps_subsampling_factor)

        # Sample features
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)  # B Np F
        sampled_features = torch.gather(features, 1, expanded_inds)

        # Sample positions
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, 3)  # B Np 3
        sampled_pos = torch.gather(pos, 1, expanded_inds)
        return sampled_features, sampled_pos


@torch.no_grad()
def density_based_sampler(features, subsample_factor, k=8):
    """
    Args:
        features: Tensor of shape (B, N, C)
        subsample_factor: downsampling factor, e.g., 4 keeps 25% of the points
        k: number of neighbors to compute local density (default: 8)

    Returns:
        sampled_inds: LongTensor (B, N//factor) with sampled point indices
    """
    B, N, C = features.shape
    # (B, N, N) pairwise distances in feature space
    dists = torch.cdist(features, features, p=2)  # L2 distance

    # Get average distance to k nearest neighbors (as inverse density estimate)
    knn_dists, _ = dists.topk(k=k, dim=-1, largest=False)
    density = knn_dists.mean(dim=-1)  # (B, N), higher = more sparse

    # Choose top M points with highest avg distance (i.e. lowest density)
    M = N // subsample_factor
    sampled_inds = density.topk(M, dim=-1, largest=True).indices  # (B, M)

    return sampled_inds
