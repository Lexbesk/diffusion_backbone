import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- small helpers ----------
def _gn(c: int) -> nn.GroupNorm:
    # choose a divisor of c for stable GN
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)

class ConvGNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.gn   = _gn(cout)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

# ---------- the encoder ----------
class DepthLightCNN(nn.Module):
    """
    Encode (B, T, H, W, 1) depth -> (B, T, d) tokens with a tiny ConvNet.
    Trains from scratch; no RGB pretraining.

    Options:
      d: token dim
      add_validity_channel: concat 1-bit valid mask as a second channel
      robust_norm: per-frame 95th percentile scaling to [0,1]
    """
    def __init__(self, d=512, add_validity_channel=True, robust_norm=True, dropout=0.0):
        super().__init__()
        self.add_validity_channel = add_validity_channel
        self.robust_norm = robust_norm
        in_ch = 1 + int(add_validity_channel)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            _gn(64), nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Stacked stride-2 blocks -> ~16x downsample total
        self.block1 = nn.Sequential(
            ConvGNAct(64, 128, k=3, s=2, p=1),
            ConvGNAct(128, 128, k=3, s=1, p=1),
        )
        self.block2 = nn.Sequential(
            ConvGNAct(128, 256, k=3, s=2, p=1),
            ConvGNAct(256, 256, k=3, s=1, p=1),
        )
        self.block3 = nn.Sequential(
            ConvGNAct(256, 256, k=3, s=2, p=1),
            ConvGNAct(256, 256, k=3, s=1, p=1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B*T, C, 1, 1)
            nn.Flatten(),             # (B*T, C)
            nn.Dropout(dropout),
            nn.Linear(256, d),
        )

        # # Kaiming init
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='gelu')
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He init with ReLU gain works well for GELU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _preprocess(depth: torch.Tensor, robust_norm: bool):
        """
        depth: (B, T, H, W, 1), positive where valid (convert from OpenGL beforehand!)
        Returns:
          x:    (B*T, 1, H, W) in [0,1]
          m:    (B*T, 1, H, W) validity mask {0,1}
        """
        B, T, H, W, _ = depth.shape
        x = depth.squeeze(-1)                     # (B,T,H,W)
        # valid if finite and > 0
        valid = torch.isfinite(x) & (x > 0)
        x = torch.where(valid, x, torch.zeros_like(x))

        x = x.view(B*T, H, W)
        valid = valid.view(B*T, H, W)

        if robust_norm:
            # per-frame 95th percentile scaling -> clamp to [0,1]
            q95 = torch.quantile(x.view(B*T, -1), 0.95, dim=-1, keepdim=True).clamp(min=1e-6)
            x = (x.view(B*T, -1) / q95).view(B*T, H, W)
        else:
            # simple min-max over valid pixels
            eps = 1e-6
            mn = torch.where(valid, x, torch.tensor(float('inf'), device=x.device)).view(B*T, -1).min(dim=-1, keepdim=True)[0]
            mx = torch.where(valid, x, torch.tensor(float('-inf'), device=x.device)).view(B*T, -1).max(dim=-1, keepdim=True)[0]
            x = (x.view(B*T, -1) - mn) / (mx - mn + eps)
            x = x.view(B*T, H, W)
        x = x.clamp(0.0, 1.0)

        return x.unsqueeze(1), valid.float().unsqueeze(1)  # (B*T,1,H,W), (B*T,1,H,W)

    def forward(self, depth_hist: torch.Tensor) -> torch.Tensor:
        """
        Input:  depth_hist (B, T, H, W, 1)
        Output: tokens     (B, T, d)
        """
        B, T, H, W, _ = depth_hist.shape
        x, m = self._preprocess(depth_hist, self.robust_norm)  # (B*T,1,H,W), (B*T,1,H,W)
        if self.add_validity_channel:
            x = torch.cat([x, m], dim=1)  # (B*T,2,H,W)

        z = self.stem(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        tok = self.head(z)                # (B*T, d)
        return tok.view(B, T, -1)         # (B, T, d)


class GoalGraspToken(nn.Module):
    """
    Inputs:
      pcl_cam:  (B, N, 3)  normalized XYZ in CAMERA frame
      grasp31:  (B, 31)    [wrist_xyz(3), wrist_rot6d(6), finger_q(22)]
    Output:
      token:    (B, 1, D)  single conditional token
    """
    def __init__(self, d_token: int, n_heads: int = 4, pdrop: float = 0.1):
        super().__init__()
        self.d = d_token

        # Per-point encoder (PointNet-style). We also feed delta-to-wrist to bias toward contact region.
        self.pt_mlp = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(),
            nn.Linear(128, d_token), nn.GELU(),
            nn.LayerNorm(d_token)
        )

        # Grasp (31-d) -> query
        self.grasp_mlp = nn.Sequential(
            nn.Linear(31, d_token), nn.GELU(),
            nn.Linear(d_token, d_token), nn.GELU(),
            nn.LayerNorm(d_token)
        )

        # Single-query cross-attention
        self.cross_attn = nn.MultiheadAttention(d_token, n_heads, batch_first=True, dropout=pdrop)

        # Fuse [context, query] -> token
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_token, d_token), nn.GELU(),
            nn.Dropout(pdrop),
            nn.LayerNorm(d_token)
        )

    def forward(self, pcl_cam: torch.Tensor, grasp31: torch.Tensor) -> torch.Tensor:
        B, N, _ = pcl_cam.shape

        wrist_xyz = grasp31[:, :3]        # (B,3)
        # Build per-point input: [xyz, xyz - wrist_xyz]
        P = torch.cat([pcl_cam, pcl_cam - wrist_xyz.unsqueeze(1)], dim=-1)  # (B,N,6)
        Pfeat = self.pt_mlp(P)            # (B,N,D)

        q = self.grasp_mlp(grasp31)       # (B,D)
        q = q.unsqueeze(1)                # (B,1,D)

        # Single-query cross-attention: query attends to the object shape
        ctx, _ = self.cross_attn(q, Pfeat, Pfeat)   # (B,1,D)

        # Fuse context with the query to get a single token
        tok = self.fuse(torch.cat([ctx, q], dim=-1))  # (B,1,D)
        return tok
