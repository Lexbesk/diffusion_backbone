# # grasp_diffuser/modules/pointnet2_backbone.py
# import torch
# import torch.nn as nn
# from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG

# class PointNet2Backbone(nn.Module):
#     """
#     MSG (multi-scale-grouping) PointNet++ that outputs
#     (a) per-point features  (B, N, C)  for local attention
#     (b) a global pooled feature  (B, C)  for conditioning
#     """
#     def __init__(self, out_dim=128):
#         super().__init__()

#         self.sa1 = PointnetSAModuleMSG(
#             npoint=1024,
#             radii=[0.05, 0.1],
#             nsamples=[32, 64],
#             mlps=[[0, 32, 32, 64],
#                   [0, 32, 64, 64]],
#         )
#         self.sa2 = PointnetSAModuleMSG(
#             npoint=256,
#             radii=[0.1, 0.2],
#             nsamples=[32, 64],
#             mlps=[[64+64, 64, 64, 128],
#                   [64+64, 64, 96, 128]],
#         )
#         self.sa3 = PointnetSAModuleMSG(
#             npoint=64,
#             radii=[0.2, 0.4],
#             nsamples=[32, 64],
#             mlps=[[128+128, 128, 128, 256],
#                   [128+128, 128, 196, 256]],
#         )

#         self.fc_global = nn.Sequential(
#             nn.Linear(512, out_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_dim, out_dim),
#         )

#     def forward(self, xyz):
#         # xyz: (B, N, 3) – world frame
#         B, N, _ = xyz.shape
#         l1_xyz, l1_feat = self.sa1(xyz, None)
#         l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
#         l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)

#         # --- global feature --------------------------------------------------
#         global_feat = torch.max(l3_feat, 2)[0]      # (B, 512)
#         global_feat = self.fc_global(global_feat)   # (B, out_dim)

#         # --- per-point feature ----------------------------------------------
#         # interpolate l3 features back to original resolution
#         interp_feat = torch.nn.functional.interpolate(
#             l3_feat.permute(0, 2, 1), size=N, mode='nearest'
#         ).permute(0, 2, 1)                          # (B, N, 512)
#         return interp_feat, global_feat

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


def build_shared_mlp(mlp_spec: List[int], bn: bool = True) -> nn.Sequential:
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], 1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class PointNet2Backbone(nn.Module):
    """
    Local-feature extractor that keeps N points unchanged.

    Args
    ----
    radii      : list[float]   — ball-query radii (one per scale)
    nsamples   : list[int]     — #neighbours per scale
    mlps       : list[list[int]]
                 PointNet MLP spec **per scale** _excluding_ xyz.
                 Example: [[0, 32, 32, 64], [0, 32, 64, 64, 128]]
                 (0 stands for “no extra point features at input”)
    out_dim    : int           — desired P (final per-point dim)
    bn         : bool          — use BatchNorm
    use_xyz    : bool          — concatenate xyz to grouped features
    """

    def __init__(
        self,
        radii: List[float],
        nsamples: List[int],
        mlps: List[List[int]],
        out_dim: int,
        bn: bool = True,
        use_xyz: bool = True,
    ):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)

        # ── Build a multiscale grouper + shared-MLP per scale ────────────────
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for r, k, mlp_spec in zip(radii, nsamples, mlps):
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(r, k, use_xyz=use_xyz)
            )
            mlp_spec = mlp_spec.copy()
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(build_shared_mlp(mlp_spec, bn))

        # ── Fuse all scales and project to P ────────────────────────────────
        in_dim = sum(m[-1] for m in mlps)
        proj = [nn.Conv1d(in_dim, out_dim, 1, bias=not bn)]
        if bn:
            proj.append(nn.BatchNorm1d(out_dim))
        proj.append(nn.ReLU(True))
        self.proj = nn.Sequential(*proj)

    # --------------------------------------------------------------------- #
    # forward
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _prepare_centroids(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Using the input itself as centroids keeps ordering intact.
        """
        # xyz : [B, N, 3]  ->  [B, 3, N] for gather_operation
        return xyz  # already the correct shape for groupers’ `new_xyz`

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : torch.Tensor
            Point cloud (B, N, 3)

        Returns
        -------
        torch.Tensor
            Local features (B, N, P)
        """
        B, N, _ = xyz.shape
        features = None  # no extra per-point input features

        centroids = self._prepare_centroids(xyz)  # [B, N, 3]

        per_scale = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            # (B, C_grp, N, k)
            grouped = grouper(xyz, centroids, features)
            # → shared MLP
            grouped = mlp(grouped)
            # → max-pool over neighbours
            grouped = F.max_pool2d(grouped, kernel_size=[1, grouped.size(3)])
            per_scale.append(grouped.squeeze(-1))  # (B, C_out_s, N)

        fused = torch.cat(per_scale, dim=1)        # (B, ΣC_out_s, N)
        fused = self.proj(fused)                   # (B, P, N)
        return fused.transpose(1, 2).contiguous()  # (B, N, P)
