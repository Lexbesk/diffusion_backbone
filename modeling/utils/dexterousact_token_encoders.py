import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFF(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = F.gelu(h)
        # h = self.dropout(h)
        h = self.fc2(h)
        # h = self.dropout(h)
        return x + h  # residual

class TokenPredictor(nn.Module):
    """
    Robust per-token predictor:
      x: (B, N, D) -> y: (B, N, out_dim)
    """
    def __init__(self, d_model, out_dim, d_hidden=None, num_blocks=1, dropout=0.1):
        super().__init__()
        d_hidden = d_hidden or (4 * d_model)
        self.blocks = nn.ModuleList([
            ResidualFF(d_model, d_hidden, dropout=dropout) for _ in range(num_blocks)
        ])
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

        # Optional: nicer init for heads
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)

class ActionTokenEncoder(nn.Module):
    """
    Encode per-timestep action targets (normalized joint positions) into tokens.

    Inputs:
      act_hist: (B, T, dof)   normalized actions (e.g., in [-1, 1])
      q_hist:   (B, T, dof)   optional, only used if include_err=True

    Output:
      tokens:   (B, T, d)
    """
    def __init__(self, dof=31, d=256, include_err=False, include_delta=False, dropout=0.0):
        super().__init__()
        self.include_err = include_err
        self.include_delta = include_delta

        in_dim = dof
        if include_err:   in_dim += dof
        if include_delta: in_dim += dof

        self.net = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    @staticmethod
    def temporal_diff(x: torch.Tensor) -> torch.Tensor:
        """Return per-step difference with zeros at the first step. Keeps shape (B,T,D)."""
        B, T, D = x.shape
        dx = x[:, 1:] - x[:, :-1]
        dx = F.pad(dx, (0, 0, 1, 0))  # pad a zero at t=0
        return dx

    def forward(self, act_hist: torch.Tensor, q_hist: torch.Tensor = None) -> torch.Tensor:
        feats = [act_hist]
        if self.include_err:
            assert q_hist is not None, "q_hist is required when include_err=True"
            feats.append(act_hist - q_hist)              # tracking error (same normalization domain)
        if self.include_delta:
            feats.append(self.temporal_diff(act_hist))   # temporal delta

        x = torch.cat(feats, dim=-1)                     # (B,T,in_dim)
        return self.net(x)                               # (B,T,d)
    
    
    

class ObjectPoseTokenEncoder(nn.Module):
    """
    Encode per-timestep object pose (xyz + rot6d -> 9D) into tokens.

    Inputs:
      pose: (B, T, 9)  where 9 = normalized xyz (3) + rot6d (6)

    Output:
      tokens: (B, T, d)
    """
    def __init__(self, d=512, include_delta=False, center_first=False, dropout=0.0):
        super().__init__()
        self.include_delta = include_delta
        self.center_first  = center_first

        in_dim = 9 + (9 if include_delta else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    @staticmethod
    def temporal_diff(x: torch.Tensor) -> torch.Tensor:
        """Per-step difference with zeros at t=0. Keeps (B,T,D)."""
        B, T, D = x.shape
        dx = x[:, 1:] - x[:, :-1]
        dx = F.pad(dx, (0, 0, 1, 0))  # pad zero row at the start
        return dx

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        x = pose
        if self.center_first:
            # subtract first-frame translation to reduce scene bias (optional)
            x = x.clone()
            x[..., :3] = x[..., :3] - x[:, :1, :3]

        feats = [x]
        if self.include_delta:
            feats.append(self.temporal_diff(x))

        z = torch.cat(feats, dim=-1)  # (B, T, in_dim)
        return self.net(z)            # (B, T, d)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, layers=2, dropout=0.0):
        super().__init__()
        hs = [in_dim] + [hidden]*(layers-1) + [out_dim]
        mods = []
        for i in range(len(hs)-2):
            mods += [nn.Linear(hs[i], hs[i+1]), nn.GELU()]
        mods += [nn.Linear(hs[-2], hs[-1])]
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

class HistoryStateTokenEncoder(nn.Module):
    """
    Encodes per-timestep history state into one token:
      - joints: q (normalized) + v (z-scored)
      - keypoints: wrist + 5 fingertips (6,3) in robot frame (made wrist-relative)
      - object pose: 9D (xyz + rot6d), already normalized/converted

    Inputs:
      q_hist:        (B, T, 31)
      v_hist:        (B, T, 31)
      ee_fingers:    (B, T, 6, 3)   0th is wrist, others are fingertips
    Output:
      state_tokens:  (B, T, d)
    """
    def __init__(self, dof=31, d=512, kp_hidden=256, dropout=0.0):
        super().__init__()
        # joints (q + v) -> d
        self.joint_mlp = MLP(in_dim=dof + dof, out_dim=d, hidden=d, dropout=dropout)
        # keypoints PointNet: per-point 3 -> kp_hidden, max-pool, then -> d
        self.kp_point = MLP(in_dim=3, out_dim=kp_hidden, hidden=kp_hidden, dropout=dropout)
        self.kp_out   = MLP(in_dim=kp_hidden, out_dim=d, hidden=d, dropout=dropout)
        # fuse 2 tokens -> 1 token
        self.fuse     = MLP(in_dim=2*d, out_dim=d, hidden=d, dropout=dropout)

    def forward(self, q_hist, v_hist, ee_fingers):
        B, T, _ = q_hist.shape

        # 1) joints
        joints_in   = torch.cat([q_hist, v_hist], dim=-1)    # (B,T,62)
        tok_joints  = self.joint_mlp(joints_in)              # (B,T,d)

        # 2) keypoints (wrist-relative)
        wrist       = ee_fingers[:, :, :1, :]                # (B,T,1,3)
        rel         = ee_fingers - wrist                     # (B,T,6,3)
        rel_flat    = rel.reshape(B*T, 6, 3)
        kp_feats    = self.kp_point(rel_flat)                # (B*T,6,kp_hidden)
        kp_pooled   = kp_feats.max(dim=1).values             # (B*T,kp_hidden)
        tok_kp      = self.kp_out(kp_pooled).view(B, T, -1)  # (B,T,d)

        # 3) fuse
        fused       = torch.cat([tok_joints, tok_kp], dim=-1)  # (B,T,2d)
        state_tokens= self.fuse(fused)                         # (B,T,d)
        return state_tokens
    
    
def zeros_xyz(x: torch.Tensor) -> torch.Tensor:
    # x: (B, N, D) -> (B, N, 3)
    return x.new_zeros(x.shape[:-1] + (3,))

# --- Helper: keep track of slices inside concatenations (for later masks, logging, etc.) ---
def build_slices(named_list):
    slices = {}
    start = 0
    for name, t in named_list:
        n = t.shape[1]
        slices[name] = (start, start + n)
        start += n
    return slices

