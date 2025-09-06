import torch
from torch import nn

from .multihead_custom_attention import MultiheadCustomAttention


class AdaLN(nn.Module):
    """Adaptive LayerNorm - signal-modulated linear transformation."""

    def __init__(self, d_model):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)
        )
        # Initialize as 0 (no scale/shift)
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: tensor (B, S, C)
            t: tensor (B, C)

        Returns:
            tensor (B, S, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1)  # (B, C), (B, C)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DummyLayer(nn.Module):
    """Implement adaptive normalization wrappers, pre-/post-norm, pos embed."""

    def __init__(self, pre_norm=False):
        super().__init__()
        self.pre_norm = pre_norm

    def _norm(self, x, layer, normalize=True):
        if normalize and layer is not None:
            return layer(x)
        return x

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def _adaln(self, x, layer, ada_sgnl):
        if layer is not None and ada_sgnl is not None:
            return layer(x, ada_sgnl)
        return x

    def forward(self):
        pass


class FFWLayer(DummyLayer):
    """Feed-forward layer for Transformers."""

    def __init__(self, d_model, dim_fw=None, dropout=0.1, use_adaln=False,
                 pre_norm=False):
        super().__init__(pre_norm=pre_norm)
        # Initialize MLP and normalization
        dim_fw = 4 * d_model if dim_fw is None else dim_fw
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_fw),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_fw, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

        # Initialize those with Xavier
        self._reset_parameters()

        # Initialize adaptive normalization separately
        self.adaln = None
        if use_adaln:
            self.adaln = AdaLN(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, ada_sgnl=None):
        """
        Args:
            x: tensor (B, S, C)
            ada_sgnl: tensor (B, C)

        Returns:
            tensor (B, S, C)
        """
        # Normalize if pre-norm
        x = self._norm(x, self.norm, self.pre_norm)
        # Adaptive normalization if applicable
        x = self._adaln(x, self.adaln, ada_sgnl)
        # Main FFW
        x = x + self.ffn(x)
        # Normalize if post-norm
        x = self._norm(x, self.norm, not self.pre_norm)
        return x


class AttentionLayer(DummyLayer):
    """Attention layer, for self-/cross-attention."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8, pre_norm=False,
                 rotary_pe=False, use_adaln=False, is_self=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__(pre_norm=pre_norm)
        self.rotary_pe = rotary_pe
        self.is_self = is_self  # self-attention, different normalization

        # Normalization and attention layers
        self.adaln = None
        if use_adaln:
            self.adaln = AdaLN(d_model)
        self.attention = MultiheadCustomAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = None
        if pre_norm:
            self.norm_kv = self.norm_q if is_self else nn.LayerNorm(d_model)

    def forward(self, seq1, seq2,
                seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None,
                ada_sgnl=None):
        """
        Args:
            seq1: tensor (B, S1, C)
            seq1_pos: (B, S1, C) if not rotary, else (B, S1, C, 2)
            seq1_sem_pos: (B, S1, C), semantic embedding
            seq2: tensor (B, S2, C)
            seq2_key_padding_mask: tensor (B, S2)
            seq2_pos: (B, S2, C) if not rotary, else (B, S2, C, 2)
            seq2_sem_pos: (B, S2, C), semantic embedding
            ada_sgnl: tensor (B, C)

        Returns:
            tensor (B, S, C)
        """
        # Normalize if pre-norm
        q1 = self._norm(seq1, self.norm_q, self.pre_norm)
        if self.is_self:
            k2 = v2 = self._norm(seq2, self.norm_q, self.pre_norm)
        else:
            k2 = v2 = self._norm(seq2, self.norm_kv, self.pre_norm)
        # Add positional embeddings if not rotary - rotary are handled later
        if not self.rotary_pe:
            q1 = self.with_pos_embed(seq1, seq1_pos)
            k2 = self.with_pos_embed(seq2, seq2_pos)
        # Add semantic embeddings, e.g. ids of each token
        q1 = self.with_pos_embed(q1, seq1_sem_pos)
        k2 = self.with_pos_embed(k2, seq2_sem_pos)
        # Adaptive normalization if applicable
        q1 = self._adaln(q1, self.adaln, ada_sgnl)
        k2 = self._adaln(k2, self.adaln if self.is_self else None, ada_sgnl)
        v2 = self._adaln(v2, self.adaln if self.is_self else None, ada_sgnl)
        # Main attention code
        seq1b = self.attention(
            query=q1.transpose(0, 1),
            key=k2.transpose(0, 1),
            value=v2.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=seq2_key_padding_mask,  # (B, S2)
            rotary_pe=(seq1_pos, seq2_pos) if self.rotary_pe else None
        )[0].transpose(0, 1)
        seq1 = seq1 + self.dropout(seq1b)
        # Normalize if post-norm
        seq1 = self._norm(seq1, self.norm_q, not self.pre_norm)
        return seq1


class AttentionModule(nn.Module):
    """Stacking of attention and feed-forward layers."""

    def __init__(self, num_layers, d_model=256, dim_fw=None,
                 dropout=0.1, n_heads=8, pre_norm=False,
                 rotary_pe=False, use_adaln=False, is_self=False):
        super().__init__()
        self.num_layers = num_layers
        self.is_self = is_self
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(AttentionLayer(
                d_model, dropout, n_heads, pre_norm,
                rotary_pe, use_adaln, is_self
            ))
            self.ffw_layers.append(FFWLayer(
                d_model, dim_fw, dropout, use_adaln, pre_norm=False
            ))

    def forward(self, seq1, seq2,
                seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None,
                ada_sgnl=None):
        """
        Args:
            seq1: tensor (B, S1, C)
            seq2: tensor (B, S2, C)
            seq2_key_padding_mask: tensor (B, S2)
            seq1_pos: (B, S1, C) if not rotary, else (B, S1, C, 2)
            seq2_pos: (B, S2, C) if not rotary, else (B, S2, C, 2)
            seq1_sem_pos: (B, S1, C), semantic embedding
            seq2_sem_pos: (B, S2, C), semantic embedding
            ada_sgnl: tensor (B, C)

        Returns:
            tensor (B, S1, C)
        """
        output = []
        for i in range(self.num_layers):
            if self.is_self:
                seq2 = seq1
            seq1 = self.attn_layers[i](
                seq1, seq2,
                seq2_key_padding_mask,
                seq1_pos, seq2_pos,
                seq1_sem_pos, seq2_sem_pos,
                ada_sgnl
            )
            seq1 = self.ffw_layers[i](seq1, ada_sgnl)
            output.append(seq1)
        return output


class DoubleCrossAttentionModule(nn.Module):
    """Stacking of two attention and one feed-forward layers."""

    def __init__(self, num_layers, d_model=256, dim_fw=None,
                 dropout=0.1, n_heads=8, pre_norm=False,
                 rotary_pe_0=False, rotary_pe_1=False,
                 use_adaln=False):
        super().__init__()
        self.num_layers = num_layers
        self.attn_layers_0 = nn.ModuleList()
        self.attn_layers_1 = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers_0.append(AttentionLayer(
                d_model, dropout, n_heads, pre_norm,
                rotary_pe_0, use_adaln, False
            ))
            self.attn_layers_1.append(AttentionLayer(
                d_model, dropout, n_heads, pre_norm,
                rotary_pe_1, use_adaln, False
            ))
            self.ffw_layers.append(FFWLayer(
                d_model, dim_fw, dropout, use_adaln, pre_norm=False
            ))

    def forward(self, seq, seq0, seq1,
                seq0_key_padding_mask=None, seq1_key_padding_mask=None,
                seq_pos_0=None, seq0_pos=None, seq_pos_1=None, seq1_pos=None,
                seq_sem_pos_0=None, seq0_sem_pos=None,
                seq_sem_pos_1=None, seq1_sem_pos=None,
                ada_sgnl=None):
        """
        Here seq attends to seq0 first, seq1 next, with different pos embed.

        Args:
            seq: tensor (B, S, C)
            seq0: tensor (B, S0, C)
            seq1: tensor (B, S1, C)
            seq0_key_padding_mask: tensor (B, S0)
            seq1_key_padding_mask: tensor (B, S1)
            seq_pos_0: (B, S, C) if not rotary, else (B, S, C, 2), first att
            seq0_pos: (B, S0, C) if not rotary, else (B, S0, C, 2)
            seq_pos_1: (B, S, C) if not rotary, else (B, S, C, 2), second att
            seq1_pos: (B, S1, C) if not rotary, else (B, S1, C, 2)
            seq_sem_pos0: (B, S, C), semantic embedding, first att
            seq0_sem_pos: (B, S0, C), semantic embedding
            seq_sem_pos1: (B, S, C), semantic embedding, second att
            seq1_sem_pos: (B, S1, C), semantic embedding
            ada_sgnl: tensor (B, C)

        Returns:
            tensor (B, S1, C)
        """
        output = []
        for i in range(self.num_layers):
            seq = self.attn_layers_0[i](
                seq, seq0,
                seq0_key_padding_mask,
                seq_pos_0, seq0_pos,
                seq_sem_pos_0, seq0_sem_pos,
                ada_sgnl
            )
            seq = self.attn_layers_1[i](
                seq, seq1,
                seq1_key_padding_mask,
                seq_pos_1, seq1_pos,
                seq_sem_pos_1, seq1_sem_pos,
                ada_sgnl
            )
            seq = self.ffw_layers[i](seq, ada_sgnl)
            output.append(seq)
        return output


class DoubleCrossSelfAttentionModule(nn.Module):
    """Stacking of two attention and one feed-forward layers."""

    def __init__(self, num_layers, d_model=256, dim_fw=None,
                 dropout=0.1, n_heads=8, pre_norm=False,
                 rotary_pe_0=False, rotary_pe_1=False,
                 use_adaln=False):
        super().__init__()
        self.num_layers = num_layers
        self.attn_layers_0 = nn.ModuleList()
        self.attn_layers_1 = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers_0.append(AttentionLayer(
                d_model, dropout, n_heads, pre_norm,
                rotary_pe_0, use_adaln, False
            ))
            self.attn_layers_1.append(AttentionLayer(
                d_model, dropout, n_heads, pre_norm,
                rotary_pe_1, use_adaln, True
            ))
            self.ffw_layers.append(FFWLayer(
                d_model, dim_fw, dropout, use_adaln, pre_norm=False
            ))

    def forward(self, seq, seq0,
                seq0_key_padding_mask=None,
                seq_pos_0=None, seq0_pos=None, seq_pos_1=None,
                seq_sem_pos_0=None, seq0_sem_pos=None,
                seq_sem_pos_1=None,
                ada_sgnl=None):
        """
        Here seq attends to seq0 first, seq1 next, with different pos embed.

        Args:
            seq: tensor (B, S, C)
            seq0: tensor (B, S0, C)
            seq0_key_padding_mask: tensor (B, S0)
            seq_pos_0: (B, S, C) if not rotary, else (B, S, C, 2), first att
            seq0_pos: (B, S0, C) if not rotary, else (B, S0, C, 2)
            seq_pos_1: (B, S, C) if not rotary, else (B, S, C, 2), second att
            seq_sem_pos0: (B, S, C), semantic embedding, first att
            seq0_sem_pos: (B, S0, C), semantic embedding
            seq_sem_pos1: (B, S, C), semantic embedding, second att
            ada_sgnl: tensor (B, C)

        Returns:
            tensor (B, S1, C)
        """
        output = []
        for i in range(self.num_layers):
            seq = self.attn_layers_0[i](
                seq, seq0,
                seq0_key_padding_mask,
                seq_pos_0, seq0_pos,
                seq_sem_pos_0, seq0_sem_pos,
                ada_sgnl
            )
            seq = self.attn_layers_1[i](
                seq, seq,
                None,
                seq_pos_1, seq_pos_1,
                seq_sem_pos_1, seq_sem_pos_1,
                ada_sgnl
            )
            seq = self.ffw_layers[i](seq, ada_sgnl)
            output.append(seq)
        return output


class StackCrossSelfAttentionModule(nn.Module):
    """Stacking of two attention and one feed-forward layers."""

    def __init__(self, num_layers, d_model=256, dim_fw=None,
                 dropout=0.1, n_heads=8, pre_norm=False,
                 rotary_pe_0=False, rotary_pe_1=False,
                 use_adaln=False):
        super().__init__()
        self.num_layers = num_layers
        self.attn_layers_0 = nn.ModuleList()
        self.attn_layers_1 = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers_0.append(AttentionLayer(
                d_model, dropout, n_heads, pre_norm,
                rotary_pe_0, use_adaln, False
            ))
            self.attn_layers_1.append(AttentionLayer(
                d_model, dropout, n_heads, pre_norm,
                rotary_pe_1, use_adaln, True
            ))
            self.ffw_layers.append(FFWLayer(
                d_model, dim_fw, dropout, use_adaln, pre_norm=False
            ))

    def forward(self, seq, seq0, seq1,
                seq0_key_padding_mask=None, seq1_key_padding_mask=None,
                seq_pos_0=None, seq0_pos=None, seq_pos_1=None, seq1_pos=None,
                seq_sem_pos_0=None, seq0_sem_pos=None,
                seq_sem_pos_1=None, seq1_sem_pos=None,
                ada_sgnl=None):
        """
        Here seq attends to seq0 first, seq1 next, with different pos embed.

        Args:
            seq: tensor (B, S, C)
            seq0: tensor (B, S0, C)
            seq1: tensor (B, S1, C)
            seq0_key_padding_mask: tensor (B, S0)
            seq1_key_padding_mask: tensor (B, S1)
            seq_pos_0: (B, S, C) if not rotary, else (B, S, C, 2), first att
            seq0_pos: (B, S0, C) if not rotary, else (B, S0, C, 2)
            seq_pos_1: (B, S, C) if not rotary, else (B, S, C, 2), second att
            seq1_pos: (B, S1, C) if not rotary, else (B, S1, C, 2)
            seq_sem_pos0: (B, S, C), semantic embedding, first att
            seq0_sem_pos: (B, S0, C), semantic embedding
            seq_sem_pos1: (B, S, C), semantic embedding, second att
            seq1_sem_pos: (B, S1, C), semantic embedding
            ada_sgnl: tensor (B, C)

        Returns:
            tensor (B, S1, C)
        """
        output = []
        for i in range(self.num_layers):
            seq = self.attn_layers_0[i](
                seq, seq0,
                seq0_key_padding_mask,
                seq_pos_0, seq0_pos,
                seq_sem_pos_0, seq0_sem_pos,
                ada_sgnl
            )
            len_ = seq.size(1)
            seq_ = self.attn_layers_1[i](
                torch.cat([seq, seq1], 1), torch.cat([seq, seq1], 1),
                None,
                torch.cat([seq_pos_1, seq1_pos], 1),
                torch.cat([seq_pos_1, seq1_pos], 1),
                torch.cat([seq_sem_pos_1, seq1_sem_pos], 1) if seq_sem_pos_1 is not None else None,
                torch.cat([seq_sem_pos_1, seq1_sem_pos], 1) if seq_sem_pos_1 is not None else None,
                ada_sgnl
            )
            seq_ = self.ffw_layers[i](seq_, ada_sgnl)
            output.append(seq_)
            seq = seq_[:, :len_, :]
            seq1 = seq_[:, len_:, :]
        return output



class AdaLayer(nn.Module):
    """LayerNorm(x) -> FiLM with cond: (1+s)*LN(x) + b"""
    def __init__(self, d_model, d_cond):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.to_scale = nn.Linear(d_cond, d_model)
        self.to_shift = nn.Linear(d_cond, d_model)
        # start as identity
        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)

    def forward(self, x, cond):          # x: (B,S,D), cond: (B,Dc)
        s = self.to_scale(cond).unsqueeze(1)   # (B,1,D)
        b = self.to_shift(cond).unsqueeze(1)   # (B,1,D)
        h = self.ln(x)
        return h * (1 + s) + b


# class TorchCrossAttnBlock(nn.Module):
#     def __init__(self, d_model=256, n_heads=8, d_ff=None, dropout=0.1):
#         super().__init__()
#         self.mha = nn.MultiheadAttention(
#             embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
#         )
#         self.do = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         d_ff = d_ff or 4 * d_model
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model), nn.Dropout(dropout),
#         )
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(self, q, kv, attn_mask=None, key_padding_mask=None):
#         """
#         q:  (B, S_Q, D)
#         kv: (B, S_KV, D)
#         attn_mask: (S_Q, S_KV) bool or float (True / -inf == block)
#         key_padding_mask: (B, S_KV) bool (True == pad/ignore)
#         """
#         h, _ = self.mha(q, kv, kv, attn_mask=attn_mask,
#                         key_padding_mask=key_padding_mask, need_weights=False)
#         x = self.norm1(q + self.do(h))
#         h = self.ff(x)
#         x = self.norm2(x + h)
#         return x

class TorchCrossAttnBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ff=None, dropout=0.1, d_cond=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.do = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        d_ff = d_ff or 4 * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # optional: condition with AdaLN before attn and before FFN
        self.d_cond = d_cond
        if d_cond is not None:
            self.ada_q = AdaLayer(d_model, d_cond)  # modulate queries
            self.ada_kv = AdaLayer(d_model, d_cond) # (optional) modulate keys/values
            self.ada_ff = AdaLayer(d_model, d_cond) # modulate stream before FFN
        else:
            self.ada_q = self.ada_kv = self.ada_ff = None

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None, cond=None):
        """
        q:  (B, S_Q, D)
        kv: (B, S_KV, D)
        cond: (B, D_cond)   # diffusion timestep embedding (global per batch)
        """
        if cond is not None and self.ada_q is not None:
            q  = self.ada_q(q,  cond)
            kv = self.ada_kv(kv, cond)

        h, _ = self.mha(q, kv, kv, attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(q + self.do(h))

        if cond is not None and self.ada_ff is not None:
            x = self.ada_ff(x, cond)

        h = self.ff(x)
        x = self.norm2(x + h)
        return x


def build_xattn_mask_modality_temporal(
    nh_act, nh_state, nh_depth, n_goal, n_grasp,
    nf_act, nf_state, nf_obj,
    device, dtype=torch.bool,
    allow_state_to_see_objects=True,   # you said "state blocks only actions"
    constrain_obj=False                # set True if you also want to limit object queries
):
    """
    Q layout:   [Af_q | Sf_q | Of_q]                lengths = (nf_act, nf_state, nf_obj)
    KV layout:  [Ah | Sh | Dh | G | GR | Af | Sf | Of]
                lengths = (nh_act, nh_state, nh_depth, n_goal, n_grasp, nf_act, nf_state, nf_obj)

    Returns a (S_Q, S_KV) bool mask (True=BLOCK).
    """
    # Build KV offsets
    kv_groups = [
        ("Ah", nh_act), ("Sh", nh_state), ("Dh", nh_depth),
        ("G", n_goal), ("GR", n_grasp),
        ("Af", nf_act), ("Sf", nf_state), ("Of", nf_obj),
    ]
    kv_off, s = {}, 0
    for name, n in kv_groups:
        kv_off[name] = (s, s + n); s += n
    S_KV = s

    # Build Q offsets
    q_groups = [("Af_q", nf_act), ("Sf_q", nf_state), ("Of_q", nf_obj)]
    q_off, s = {}, 0
    for name, n in q_groups:
        q_off[name] = (s, s + n); s += n
    S_Q = s

    mask = torch.zeros((S_Q, S_KV), dtype=dtype, device=device)  # False=allow

    def block(q_name, kv_name):
        qs, qe = q_off[q_name]; ks, ke = kv_off[kv_name]
        if qe > qs and ke > ks:
            mask[qs:qe, ks:ke] = True

    def causal_upper(q_name, kv_name):
        # Block keys with step > query step (upper triangle) inside future blocks.
        qs, qe = q_off[q_name]; ks, ke = kv_off[kv_name]
        Fq, Fk = qe - qs, ke - ks
        F = min(Fq, Fk)
        if F <= 0: return
        tri = torch.triu(torch.ones((F, F), dtype=dtype, device=device), diagonal=1)  # True=BLOCK
        mask[qs:qs+F, ks:ks+F] = tri

    Ah, Sh, Dh, G, GR, Af, Sf, Of = "Ah","Sh","Dh","G","GR","Af","Sf","Of"

    # ===== Rules =====
    # (1) State-future queries: block ALL future actions; allow everything else (including all states).
    block("Sf_q", Af)
    if not allow_state_to_see_objects:
        block("Sf_q", Of)
    # No causal on Sf_q→Sf: states can see all future states.

    # (2) Action-future queries: causal to Af and Sf; block Of entirely.
    causal_upper("Af_q", Af)
    causal_upper("Af_q", Sf)
    block("Af_q", Of)

    # (3) Object-future queries (optional)
    if constrain_obj:
        causal_upper("Of_q", Of)
        block("Of_q", Af)
        block("Of_q", Sf)

    return mask, q_off, kv_off


import matplotlib.pyplot as plt

def plot_attn_mask(mask, q_off, kv_off,
                   q_labels=("Af_q","Sf_q","Of_q"),
                   kv_labels=("Ah","Sh","Dh","G","GR","Af","Sf","Of"),
                   title="Cross-Attention Mask (black = BLOCKED)",
                   savepath=None, figsize=(7,5)):
    """
    mask: (S_Q, S_KV) torch.bool or float mask (True or -inf = block)
    q_off, kv_off: dicts like {"Af_q": (start,end), ...} from your builder
    """
    # Normalize to boolean "blocked"
    if mask.dtype == torch.bool:
        blocked = mask
    else:
        # float/additive mask: assume -inf or very negative means blocked
        blocked = ~torch.isfinite(mask) | (mask < -1e6)

    arr = blocked.detach().cpu().numpy().astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap="gray_r", interpolation="nearest", aspect="auto")
    # gray_r makes 1.0 (blocked) = black, 0.0 (allowed) = white

    # Draw boundaries
    # Horizontal (queries)
    for name in q_labels:
        s,e = q_off[name]
        ax.hlines(e-0.5, -0.5, arr.shape[1]-0.5, lw=0.6, color="k")
    # Vertical (KV)
    for name in kv_labels:
        s,e = kv_off[name]
        ax.vlines(e-0.5, -0.5, arr.shape[0]-0.5, lw=0.6, color="k")

    # Ticks at group centers
    q_ticks = [ (q_off[n][0] + q_off[n][1] - 1) / 2 for n in q_labels ]
    kv_ticks = [ (kv_off[n][0] + kv_off[n][1] - 1) / 2 for n in kv_labels ]
    ax.set_yticks(q_ticks); ax.set_yticklabels(q_labels, fontsize=9)
    ax.set_xticks(kv_ticks); ax.set_xticklabels(kv_labels, rotation=90, fontsize=9)

    ax.set_ylabel("Queries (S_Q)")
    ax.set_xlabel("Keys/Values (S_KV)")
    ax.set_title(title)
    ax.grid(False)

    # Legend text
    ax.text(1.01, 0.02, "white = allowed\nblack = blocked",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=9)

    plt.tight_layout()
    plt.savefig(savepath, dpi=220, bbox_inches="tight")