import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import fps
from torchvision.ops import Conv2dNormActivation

from ..utils.position_encodings import RotaryPositionEncoding3D
from ..utils.layers import FFWRelativeCrossAttentionModule, ParallelAttention
from .vision.resnet import load_resnet50, load_resnet18
from .vision.clip import load_clip
from .vision.tiny_vit import load_tiny
from .vision.florence2 import load_florence2
from .vision.fpn import EfficientFeaturePyramidNetwork


class Encoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_ins_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 ayush=False):
        super().__init__()
        self.fps_subsampling_factor = fps_subsampling_factor
        self.ayush = ayush

        # Frozen backbone
        self.use_florence = backbone == "florence2"
        if backbone == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        elif backbone == "tiny":
            self.backbone, self.normalize = load_tiny()
        elif backbone == "florence2":
            self.backbone, self.normalize = load_florence2()
        for p in self.backbone.parameters():
            p.requires_grad = finetune_backbone

        # Coarse RGB features are the 3rd layer of the feature pyramid
        # at 1/8 resolution (32x32)
        self.feature_map_pyramid = ['res3']

        # Semantic visual features at different scales
        output_level = min([int(lvl[3:]) for lvl in self.feature_map_pyramid])
        output_level = f"res{output_level}"
        if self.use_florence:
            self.inner_block = Conv2dNormActivation(
                768, embedding_dim, kernel_size=1, padding=0,
                norm_layer=None, activation_layer=None
            ).half()
        elif backbone != 'tiny':
            self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048],
                embedding_dim, output_level=output_level
            )
        else:
            self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 128, 160, 320],
                embedding_dim, output_level=output_level
            )

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=False
        )

        # Goal gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        # Instruction encoder
        self.instruction_encoder = nn.Linear(
            768 if self.use_florence else 512, embedding_dim,
            dtype=torch.float16 if self.use_florence else None
        )

        # Attention from vision to language
        layer = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=embedding_dim, n_heads=num_attn_heads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )
        self.vl_attention = nn.ModuleList([
            layer
            for _ in range(1)
            for _ in range(1)
        ])

    def forward(self):
        return None

    def encode_curr_gripper(self, curr_gripper, context_feats, context):
        """
        Compute current gripper position features and positional embeddings.

        Args:
            - curr_gripper: (B, nhist, 3+)

        Returns:
            - curr_gripper_feats: (B, nhist, F)
            - curr_gripper_pos: (B, nhist, F, 2)
        """
        return self._encode_gripper(curr_gripper, self.curr_gripper_embed,
                                    context_feats, context)

    def encode_goal_gripper(self, goal_gripper, context_feats, context):
        """
        Compute goal gripper position features and positional embeddings.

        Args:
            - goal_gripper: (B, 3+)

        Returns:
            - goal_gripper_feats: (B, 1, F)
            - goal_gripper_pos: (B, 1, F, 2)
        """
        goal_gripper_feats, goal_gripper_pos = self._encode_gripper(
            goal_gripper[:, None], self.goal_gripper_embed,
            context_feats, context
        )
        return goal_gripper_feats, goal_gripper_pos

    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: (B, npt, 3+)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)

        Returns:
            - gripper_feats: (B, npt, F)
            - gripper_pos: (B, npt, F, 2)
        """
        # Learnable embedding for gripper
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper), 1, 1
        )

        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

        gripper_feats = einops.rearrange(
            gripper_feats, 'b npt c -> npt b c'
        )
        context_feats = einops.rearrange(
            context_feats, 'b npt c -> npt b c'
        )
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats,
            query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = einops.rearrange(
            gripper_feats, 'nhist b c -> b nhist c'
        )

        return gripper_feats, gripper_pos

    def encode_images(self, rgb, pcd):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        rgb_features = self.feature_pyramid(rgb_features)

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        rgb_feats_pyramid = []
        pcd_pyramid = []
        for i in range(len(self.feature_map_pyramid)):
            # Isolate level's visual features
            rgb_features_i = rgb_features[self.feature_map_pyramid[i]]

            # Interpolate xy-depth to get the locations for this level
            feat_h, feat_w = rgb_features_i.shape[-2:]
            pcd_i = F.interpolate(
                pcd,
                (feat_h, feat_w),
                mode='bilinear'
            )

            # Merge different cameras for clouds, separate for rgb features
            pcd_i = einops.rearrange(
                pcd_i,
                "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            rgb_features_i = einops.rearrange(
                rgb_features_i,
                "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )

            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)

        return rgb_feats_pyramid, pcd_pyramid

    def encode_florence(self, rgb, pcd, text):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions
            - text: [str]

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
            - text_features: (B, )
        """
        num_cameras = rgb.shape[1]
        
        # Float16 for florence2
        rgb = rgb.half()
        pcd = pcd.half()

        # Pass all views jointly through backbone
        rgb = self.normalize(rgb)
        _features = self.backbone(rgb, text)
        _features = _features['res5']  # B N=ncam*65+text_len F
        
        # Isolate visual and text features
        _rgb_features = _features[:, :num_cameras * 65]
        rgb_features = torch.stack([
            _rgb_features[:, c * 65 + 1:65 * c + 65]
            for c in range(num_cameras)
        ], 1)  # (B, n_cam, 64, F)
        rgb_features = einops.rearrange(
            rgb_features,
            "B ncam (h w) F -> (B ncam) F h w",
            h=8, w=8
        )
        text_features = _features[:, num_cameras * 65:]
        
        # Get the unprojected visual features
        rgb_features = self.inner_block(rgb_features)
        if not self.ayush:
            rgb_features = F.interpolate(
                rgb_features, size=(32, 32), mode="nearest"
            )
        rgb_feats_pyramid = [einops.rearrange(
            rgb_features,
            "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
        )]
        feat_h, feat_w = rgb_features.shape[-2:]
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")
        pcd = F.interpolate(pcd, (feat_h, feat_w), mode='bilinear')
        pcd_pyramid = [einops.rearrange(
            pcd,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )]

        # Get the projected text features
        text_features, text_pos = self.encode_instruction(text_features)

        return rgb_feats_pyramid, pcd_pyramid, text_features, text_pos

    def encode_instruction(self, instruction):
        """
        Compute language features/pos embeddings on top of CLIP features.

        Args:
            - instruction: (B, max_instruction_length, 512)

        Returns:
            - instr_feats: (B, 53, F)
            - instr_dummy_pos: (B, 53, F, 2)
        """
        instr_feats = self.instruction_encoder(instruction)
        # Dummy positional embeddings, all 0s
        instr_dummy_pos = torch.zeros(
            len(instruction), instr_feats.shape[1], 3,
            device=instr_feats.device
        )
        instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
        return instr_feats, instr_dummy_pos

    def run_fps(self, context_features, context_pos):
        # context_features (Np, B, F)
        # context_pos (B, Np, F, 2)
        # outputs of analogous shape, with smaller Np
        if self.fps_subsampling_factor == 1:
            return context_features, context_pos

        npts, bs, ch = context_features.shape

        batch_indices = torch.arange(
            bs, device=context_features.device).long()
        batch_indices = batch_indices[:, None].repeat(1, npts)
        batch_indices = batch_indices.flatten(0, 1)
        sampled_inds = fps(
            einops.rearrange(
                context_features, "npts b c -> (b npts) c").half(),
            batch_indices,
            1. / self.fps_subsampling_factor,
            random_start=True
        )
        sampled_inds = sampled_inds.reshape(bs, -1)
        sampled_inds = sampled_inds % npts

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c")
        )

        # Sample positional embeddings
        _, _, ch, npos = context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ch, npos)
        )
        sampled_context_pos = torch.gather(
            context_pos, 1, expanded_sampled_inds
        )
        return sampled_context_features, sampled_context_pos

    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats
