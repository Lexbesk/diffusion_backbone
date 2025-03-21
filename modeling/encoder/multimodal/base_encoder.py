from torch import nn

from ...utils.layers import AttentionModule
from ..vision import fetch_visual_encoders
from ..text import fetch_text_encoders


class Encoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False):
        super().__init__()
        self.fps_subsampling_factor = fps_subsampling_factor
        self._backbone_name = backbone

        # Instruction encoder
        self.text_encoder, _dim = fetch_text_encoders(backbone)
        if self.text_encoder is not None:  # is None when using a VLM
            for p in self.text_encoder.parameters():
                p.requires_grad = finetune_text_encoder
            self.instruction_encoder = nn.Linear(_dim, embedding_dim)

        # Scene encoder
        self.backbone, self.normalize = fetch_visual_encoders(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = finetune_backbone

        # Attention from vision to language
        self.vl_attention = AttentionModule(
            num_layers=num_vis_instr_attn_layers, d_model=embedding_dim,
            dim_fw=4 * embedding_dim, n_heads=num_attn_heads
        )

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio):
        """
        Encode different modalities, independent of denoising step.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W)
            - rgb2d: (B, ncam2d, 3, H, W)
            - pcd: (B, ncam3d, 3, H, W)
            - instruction: [str], len=B
            - proprio: (B, nhist, 3+6+X)

        Returns:
            - rgb3d_feats: (B, N, F)
            - pcd: (B, N, 3)
            - rgb2d_feats: (B, ncam2d, F)
            - rgb2d_pos: (B, ncam2d, 3)
            - instr_feats: (B, L, F)
            - instr_pos: (B, L, 3)
            - proprio_feats: (B, nhist, F)
            - fps_scene_feats: (B, n, F), n < N
            - fps_scene_pos: (B, n, 3), n < N
        """
        vl_enc_fn = {
            'clip': self.encode_clip,
            # 'siglip2_256': self.encode_siglip,
            # 'siglip2_512': self.encode_siglip
        }[self._backbone_name]
        # Compute scene features/positional embeddings, language embeddings
        rgb3d_feats, rgb2d_feats, pcd, instr_feats = vl_enc_fn(
            rgb3d, rgb2d, pcd, instruction
        )

        # Use the current end-effector position as rgb2d position
        rgb2d_pos = None
        if rgb2d_feats is not None:
            _prop = proprio.reshape(len(proprio), -1, rgb2d_feats.size(1), 9)
            rgb2d_pos = _prop[:, -1, :, :3]

        # Use the current end-effector position as language 'position'
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)

        # Encode proprioception
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd)

        # FPS based on scene features
        fps_scene_feats, fps_scene_pos = self.run_fps(rgb3d_feats, pcd)

        return (
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        )

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
        return None

    def encode_clip(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W), rgb obs of 3D cameras
            - rgb2d: (B, ncam2d, 3, H, W), rgb obs of 2D cameras
            - pcd: (B, ncam3d, 3, H, W) or None
            - text: [str] of len=B, text instruction

        Returns:
            - rgb3d_feats: (B, Np, F)
            - rgb2d_feats: (B, ncam2d, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        return None, None, None, None

    def run_fps(self, features, pos):
        # features (B, Np, F)
        # context_pos (B, Np, 3)
        # outputs of analogous shape, with smaller Np
        return features, pos
