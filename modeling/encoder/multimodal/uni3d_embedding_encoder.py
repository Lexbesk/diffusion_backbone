"""
See https://github.com/baaivision/Uni3D for source code
"""
import os
import torch
import torch.nn as nn
import timm
import numpy as np
# from pointnet2_ops import pointnet2_utils
from pointnet2_ops import pointnet2_utils as pn2
import open_clip
from huggingface_hub import hf_hub_download
import sys
sys.path.append('')
from uni3d.utils.tokenizer import SimpleTokenizer

import logging

from typing import Sequence
from abc import ABC, abstractmethod
import torch
from PIL.Image import Image

class FeatureExtractor(ABC):
    @abstractmethod
    def encode_image(self, img_list: Sequence[Image]) -> torch.Tensor:
        """
        Encode the input images and return the corresponding embeddings.

        Args:
            img_list: A list of PIL.Image.Image objects.

        Returns:
            The embeddings of the input images. The shape should be (len(img_list), embedding_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text_list: Sequence[str]) -> torch.Tensor:
        """
        Encode the input text data and return the corresponding embeddings.

        Args:
            text_list: A list of strings.

        Returns:
            The embeddings of the input text data. The shape should be (len(text_list), embedding_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def encode_3D(self, pc_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode the input 3D point cloud and return the corresponding embeddings.

        Args:
            pc_tensor: A tensor of shape (B, N, 3 + 3).
        
        Returns:
            The embeddings of the input 3D point cloud. The shape should be (B, embedding_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, queries: Sequence[str]) -> torch.Tensor:
        """Encode the queries and return the corresponding embeddings.

        Args:
            queries: A list of strings.

        Returns:
            The embeddings of the input text data. The shape should be (len(input_text), embedding_dim).
        """
        raise NotImplementedError
    
def three_interpolate(feat, idx, weight):
    """
    feat:   (B, M, C)  features of the coarse points
    idx:    (B, N, 3)  indices of the 3 nearest coarse points for every fine point
    weight: (B, N, 3)  inverse-distance weights (row-normalised)
    returns (B, N, C)  interpolated features
    """
    B, M, C = feat.shape
    _, N, _ = idx.shape
    feat = feat.gather(1, idx.view(B, N * 3, 1).expand(-1, -1, C))         # (B, N*3, C)
    feat = feat.view(B, N, 3, C)                                           # (B, N, 3, C)
    weight = weight.unsqueeze(-1)                                          # (B, N, 3, 1)
    feat = (feat * weight).sum(2)                                          # (B, N, C)
    return feat

class FPBlock(nn.Module):
    """One PointNet++ Feature-Propagation layer (3-NN interp + MLP)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))

    def forward(self, xyz_target, xyz_src, feat_target, feat_src):
        """
        xyz_target : (B, N, 3)  fine points (want features here)
        xyz_src    : (B, M, 3)  coarse points (have features here)
        feat_target: (B, N, C1)  skip connection (may be None for top FP)
        feat_src   : (B, M, C2)
        returns (B, N, out_channels)
        """
        dists, idx = pn2.three_nn(xyz_target, xyz_src)         # (B,N,3)
        idx  = idx.long()
        norm = 1.0 / (dists + 1e-8)
        norm = norm / norm.sum(dim=2, keepdim=True)           # row-normalise
        interp = three_interpolate(feat_src, idx, norm)       # (B,N,C2)

        if feat_target is not None:
            feat = torch.cat([interp, feat_target], dim=-1)   # (B,N,C1+C2)
        else:
            feat = interp
        return self.mlp(feat)                                 # (B,N,out)
    

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pn2.furthest_point_sample(data, number) 
    fps_data = pn2.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

# https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py 
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info("patch dropout prob is {}".format(prob))

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features, idx

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, trans_dim=1408):
        # use the giant branch of uni3d
        super().__init__()
        from easydict import EasyDict
        self.trans_dim = trans_dim
        self.embed_dim = 1024
        self.group_size = 64
        self.num_group = 512
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim = 512
        self.encoder = Encoder(encoder_channel = self.encoder_dim)
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(0.) if 0. > 0. else nn.Identity()
        self.visual = point_transformer
        
        self.fp1 = FPBlock(in_channels=1539,  out_channels=256)   # L12  → L8 # 2819 for giant
        self.fp2 = FPBlock(in_channels=trans_dim + 256 + 3, out_channels=256)  # L8 → L4
        self.fp3 = FPBlock(in_channels=515, out_channels=256)  # L4 → points

    def forward_with_points(self, pts, colors):
        """
        pts    (B, N, 3), colors (B, N, 3)
        returns {"cls": (B,1024), "points": (B,N,256)}
        """
        # ❶ patch grouping (grouper already returns idx now)
        neighborhood, center, features, idx = self.group_divider(pts, colors)

        # ❷ tokenise + ViT (same as your current forward)
        group_tok = self.encoder2trans(self.encoder(features))          # (B,G,1408)

        cls_tok   = self.cls_token.expand(group_tok.size(0), -1, -1)
        pos       = self.pos_embed(center)
        x = torch.cat([cls_tok, group_tok], 1) + \
            torch.cat([self.cls_pos.expand_as(cls_tok), pos], 1)

        x = self.patch_dropout(x)
        x = self.visual.pos_drop(x)

        hidden4, hidden8 = None, None
        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)
            if i == 3: hidden4 = x
            if i == 7: hidden8 = x
        t12 = x[:, 1:]          # (B,G,1408)
        t8  = hidden8[:, 1:]
        t4  = hidden4[:, 1:]

        cls_embed = self.trans2embed(x[:, 0])          # (B,1024)

        # ❸ PointNet++ FP hierarchy (three layers)
        # anchors = FPS centres already in `center`  (B,G,3)
        # idx     = (B,G,M)  original-point indices for each anchor
        B, N, _ = pts.shape
        device  = pts.device

        # L12 → L8   (same anchor xyz ⇒ no interp)
        f12 = t12                                  # (B,G,1408)
        f8  = self.fp1(center, center, None,
                    torch.cat([t8, f12, center], -1))          # +xyz residual

        # L8 → L4
        f4  = self.fp2(center, center, None,
                    torch.cat([t4, f8, center], -1))

        # L4 → original points
        # build 3-NN from every point to its anchor (all Ns are within that anchor)
        idx_flat   = idx.reshape(B, -1)[:, :self.num_group]        # (B,G)
        anchor_xyz = center                                        # (B,G,3)
        xyz_tgt    = pts                                           # (B,N,3)

        feat_points = self.fp3(xyz_tgt, anchor_xyz, None,
                            torch.cat([f4, f8, anchor_xyz], -1))  # (B,N,256)

        return {"cls": cls_embed, "points": feat_points}

    def forward(self, pts, colors):
        # divide the point cloud in the same form. This is important
        _, center, features = self.group_divider(pts, colors)

        # encoder the input cloud patches
        group_input_tokens = self.encoder(features)  #  B G N
        group_input_tokens = self.encoder2trans(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        # x = x.half()
        
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)

        x = self.visual.pos_drop(x)
        
        hidden_4, hidden_8 = None, None

        # ModuleList not support forward
        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)
            if i == 3:   hidden_4 = x                 # after block 4 (0-based)
            if i == 7:   hidden_8 = x 
        x = self.visual.norm(x[:, 0, :])
        x = self.visual.fc_norm(x)

        x = self.trans2embed(x)
        return x

class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        xyz = pc[:,:,:3].contiguous()
        color = pc[:,:,3:].contiguous()
        # pc_feat = self.point_encoder(xyz, color)
        return self.point_encoder.forward_with_points(xyz, color)
        # return pc_feat

    def forward(self, pc, text, image):
        text_embed_all = text
        image_embed = image   
        pc_embed = self.encode_pc(pc)
        return {'text_embed': text_embed_all,
                'pc_embed': pc_embed,
                'image_embed': image_embed,
                'logit_scale': self.logit_scale.exp()}

def get_metric_names(model):
    return ['loss', 'uni3d_loss', 'pc_image_acc', 'pc_text_acc']

def create_uni3d(size_tag="b"):  
    # create transformer blocks for point cloud via timm
    EVA_VARIANTS = {
        "b": ("eva02_base_patch14_448",  768),
        "l": ("eva02_large_patch14_448", 1024),
        "g": ("eva_giant_patch14_560", 1408),
    }
    
    uni3d_paths = {
        'b': '/data/user_data/austinz/Robots/manipulation/analogical_manipulation/checkpoints/uni3d/models--BAAI--Uni3D/snapshots/3d8233b76aa350d72f6213ecd2123c2026b42355/modelzoo/uni3d-b/model.pt',
        
    }
    eva_name, trans_dim = EVA_VARIANTS[size_tag]
    uni3d_path = uni3d_paths[size_tag]
    # point_transformer = timm.create_model("eva_giant_patch14_560")
    point_transformer = timm.create_model(
        eva_name,
        pretrained=True,
        # checkpoint_path='/data/user_data/austinz/Robots/manipulation/analogical_manipulation/checkpoints/clip_folder/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_pytorch_model.bin', 
    )

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer, trans_dim=trans_dim)

    # uni3d model
    model = Uni3D(point_encoder=point_encoder,)

    checkpoint = torch.load(uni3d_path, map_location='cpu')
    logging.info('loaded checkpoint {}'.format(uni3d_path))
    sd = checkpoint['module']
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    # model.load_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded Uni3D weights  |  missing: {len(missing)}  unexpected: {len(unexpected)}")
    print(f"  • Missing keys    ({len(missing)}):")
    for k in missing:
        print(f"    - {k}")
    print(f"  • Unexpected keys ({len(unexpected)}):")
    for k in unexpected:
        print(f"    - {k}")
    return model

class Uni3dEmbeddingEncoder(FeatureExtractor):
    def __init__(self, cache_dir, **kwargs) -> None:
        bpe_path = "utils/bpe_simple_vocab_16e6.txt.gz"
        # uni3d_path = os.path.join(cache_dir, "Uni3D", "modelzoo", "uni3d-g", "model.pt") # concat the subfolder as hf_hub_download will put it here
        clip_path = os.path.join(cache_dir, "Uni3D", "open_clip_pytorch_model.bin")

        # if not os.path.exists(uni3d_path):
        #     hf_hub_download("BAAI/Uni3D", "model.pt", subfolder="modelzoo/uni3d-g", cache_dir=cache_dir, 
        #                     local_dir=cache_dir + os.sep + "Uni3D")
        if not os.path.exists(clip_path):
            hf_hub_download("timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k", "open_clip_pytorch_model.bin", 
                            cache_dir=cache_dir, local_dir=cache_dir + os.sep + "Uni3D")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = SimpleTokenizer(bpe_path)
        # self.model = create_uni3d(uni3d_path)
        # self.model.eval()
        # self.model.to(self.device)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name="EVA02-E-14-plus", pretrained=clip_path)
        self.clip_model.to(self.device)
        
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    @torch.no_grad()
    def encode_3D(self, data):
        pass
    #     pc = data.to(device=self.device, non_blocking=True)
    #     pc_features = self.model.encode_pc(pc)
    #     pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
    #     return pc_features.float()

    @torch.no_grad()
    def encode_text(self, input_text):
        texts = self.tokenizer(input_text).to(device=self.device, non_blocking=True)
        if len(texts.shape) < 2:
            texts = texts[None, ...]
        class_embeddings = self.clip_model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        return class_embeddings.float()

    @torch.no_grad()
    def encode_image(self, img_tensor_list):
        image = img_tensor_list.to(device=self.device, non_blocking=True)
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    def encode_query(self, query_list):
        return self.encode_text(query_list)
    
    def get_img_transform(self):
        return self.preprocess