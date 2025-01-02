import torch
from torch import nn
from torch.nn import functional as F

import diffuser_actor.utils.libs.pointops2.functions.pointops as pointops


class CrossViewPAnet(nn.Module):

    def __init__(self, latent_dim, num_layers=6, nheads=8,
                 nsample=16, dropout=0.0, dim_feedforward=None):
        super().__init__()
        self.cross_view_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=latent_dim,
                nhead=nheads,
                dropout=dropout,
                normalize_before=True,
                activation='relu',
            ) for _ in range(num_layers)
            ])
        if dim_feedforward is None:
            dim_feedforward = 4 * latent_dim
        self.ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=latent_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=True,
                activation='relu',
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        self.nsample = nsample
        self.num_layers = num_layers
        self.pe_layer = self.init_pe(latent_dim)

    def init_pe(self, latent_dim):
        # if self.cfg.USE_MLP_POSITIONAL_ENCODING:
        pe_layer = PositionEmbeddingLearnedMLP(
                dim=3, num_pos_feats=latent_dim
        )
        # else:
        #     pe_layer = PositionEmbeddingLearned(
        #             dim=3, num_pos_feats=latent_dim
        #     )
        return pe_layer

    def encode_pe(self, xyz=None):
        return self.pe_layer(xyz)
        
    def forward(self, feature, xyz):
        """
        Args:
            feature: tensor (B, V, C, H, W)
            xyz: tensor (B, V, H, W, 3)

        Returns:
            output: tensor (B, V, C, H, W)
        """
        bs, v, f, h, w = feature.shape

        # B, V, F, H, W -> B, V*H*W, F
        feature = feature.permute(0, 1, 3, 4, 2).flatten(1, 3)  # B, VHW, F
        xyz = xyz.flatten(1, 3)  # B, VHW, 3

        # queryandgroup expects N, F and N, 3 with additional batch offset
        xyz = xyz.flatten(0, 1).contiguous()
        feature = feature.flatten(0, 1).contiguous()
        batch_offset = v * h * w * (
            torch.arange(bs, dtype=torch.int32, device=xyz.device) + 1
        )

        knn_points_feats, idx = pointops.queryandgroup(
            self.nsample, xyz, xyz, feature, None,
            batch_offset, batch_offset, use_xyz=True, return_indx=True
        )  # (B*n, nsample, 3+c)

        knn_points = knn_points_feats[..., :3]  # B*N, nsample, 3
        
        # encode_pe expects B, N, 3
        query_pe = self.encode_pe(torch.zeros_like(xyz[:, None])).permute(1, 0, 2)
        knn_pe = self.encode_pe(knn_points).permute(1, 0, 2)

        output = feature[:, None]  # B*N, 1, c
        bn, _, c = output.shape
        for i in range(self.num_layers):
            # get knn features from updated output
            # from pdb import set_trace; set_trace()
            key = output.flatten(0, 1)[idx.view(-1).long(), :].reshape(bn, self.nsample, c).permute(1, 0, 2)
            output = self.cross_view_attention_layers[i](
                tgt=output.permute(1, 0, 2),
                memory=key,
                query_pos=query_pe,
                pos=knn_pe
            )
            output = self.ffn_layers[i](output).permute(1, 0, 2) 
            output = self.layer_norms[i](output)
                
        output = output.reshape(bs, v, h, w, c).permute(0, 1, 4, 2, 3)
        return output


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, dim=3, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(dim, num_pos_feats, kernel_size=1),
            nn.GroupNorm(1, num_pos_feats),
            nn.ReLU(),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        shape_len = len(xyz.shape)
        if shape_len == 5:
            B, V, H, W, _ = xyz.shape
            xyz = xyz.flatten(1, 3).permute(0, 2, 1)
        elif shape_len == 3:
            xyz = xyz.permute(0, 2, 1)
        else:
            raise ValueError("xyz should be 3 or 5 dimensional")
        position_embedding = self.position_embedding_head(xyz)
        if shape_len == 5:
            return position_embedding.permute(0, 2, 1).reshape(B, V, H, W, -1)
        else:
            return position_embedding.permute(0, 2, 1)
        
        
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
        shape_len = len(xyz.shape)
        if shape_len == 5:
            B, V, H, W, _ = xyz.shape
            xyz = xyz.flatten(1, 3)
        elif shape_len == 3:
            xyz = xyz
        else:
            raise ValueError("xyz should be 3 or 5 dimensional")
        position_embedding = self.position_embedding_head(xyz)
        if shape_len == 5:
            return position_embedding.reshape(B, V, H, W, -1)
        else:
            return position_embedding


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False,
                ):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask=None,
                     memory_key_padding_mask=None,
                     pos=None,
                     query_pos=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None,
                    query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):

        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False, 
                ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
