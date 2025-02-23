import math

import torch
import torch.nn as nn
import einops


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SinusoidalPosEmb2D(nn.Module):

    def __init__(self, dim, temperature=10000.):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        assert self.dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D position embedding'

    def forward(self, h, w, dtype=torch.float32, device="cuda"):
        grid_w = torch.arange(w, dtype=dtype, device=device)
        grid_h = torch.arange(h, dtype=dtype, device=device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w)
        pos_dim = self.dim // 4
        omega = torch.arange(pos_dim, dtype=dtype, device=device)
        omega = omega / pos_dim
        omega = 1. / (self.temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ], dim=1)
        pos_emb = einops.rearrange(pos_emb, '(h w) c -> h w c', h=h, w=w)

        return pos_emb


class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class LearnedAbsolutePositionEncoding3D(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.absolute_pe_layer = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim, kernel_size=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        )

    def forward(self, xyz):
        """
        Arguments:
            xyz: (B, N, 3) tensor of the (x, y, z) coordinates of the points

        Returns:
            absolute_pe: (B, N, embedding_dim) tensor of the absolute position encoding
        """
        return self.absolute_pe_layer(xyz.permute(0, 2, 1)).permute(0, 2, 1)


class LearnedAbsolutePositionEncoding3Dv2(nn.Module):
    def __init__(self, input_dim, embedding_dim, norm="none"):
        super().__init__()
        norm_tb = {
            "none": nn.Identity(),
            "bn": nn.BatchNorm1d(embedding_dim),
        }
        self.absolute_pe_layer = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim, kernel_size=1),
            norm_tb[norm],
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        )

    def forward(self, xyz):
        """
        Arguments:
            xyz: (B, N, 3) tensor of the (x, y, z) coordinates of the points

        Returns:
            absolute_pe: (B, N, embedding_dim) tensor of the absolute position encoding
        """
        return self.absolute_pe_layer(xyz.permute(0, 2, 1)).permute(0, 2, 1)


class PositionEmbeddingLearnedMLP(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, dim=3, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(dim, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding
