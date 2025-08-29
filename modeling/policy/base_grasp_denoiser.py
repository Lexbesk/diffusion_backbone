import torch
from torch import nn
from torch.nn import functional as F
import einops

from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb
from ..utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)
# from ..utils.vis_utils import render_grasps
from ..encoder.multimodal.pointnet import PointNet2Backbone
from ..encoder.multimodal.uni3d_embedding_encoder import create_uni3d
from ..utils.position_encodings import RotaryPositionEncoding3D
import mujoco
from mujoco import mj_id2name, mjtObj
import numpy as np
from easydict import EasyDict
from huggingface_hub import hf_hub_download
import time
from collections import defaultdict

from utils.forward_kinematics.pk_utils import build_chain_from_mjcf_path
from utils.forward_kinematics.pytorchfk import get_joint_positions

from pytorch3d.ops import sample_farthest_points


def prepare_pc(xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz : (B, N, 3) world-space point cloud
    Returns pc : (B, N, 6) xyz  +  dummy colour
    """
    B,N,_ = xyz.shape
    colour = torch.zeros(B, N, 3, device=xyz.device, dtype=xyz.dtype)
    return torch.cat([xyz, colour], dim=-1)

class SceneEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, N, P)
        return self.net(feats)


class _ResBlock(nn.Module):
    """Pre-activation residual MLP block with FiLM-style scene conditioning."""
    def __init__(self, width: int, scene_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(scene_dim, width * 2)          # γ ⊕ β
        self.dropout = nn.Dropout(dropout)

        # Kaiming-zero so the whole block is an (almost) identity at init
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, scene_feat):
        γ, β = self.film(scene_feat).chunk(2, dim=-1)
        h_ = self.norm(h)
        h_ = F.gelu(self.fc1(h_))
        h_ = self.dropout(h_)
        h_ = self.fc2(h_)
        h_ = h_ * (1 + γ) + β          # FiLM modulation
        return h + h_                  # residual
        

class AuxPoseHead(nn.Module):
    """
    Predict (Δpre-grasp, Δsqueeze) given a sampled grasp pose and a scene feature.
    Input : grasp (B, 29), scene_feat (B, 128)
    Output: (B, 2, 29)
    """
    def __init__(
        self,
        pose_dim:   int   = 29,
        scene_dim:  int   = 128,
        width:      int   = 512,
        depth:      int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(pose_dim + scene_dim, width),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [_ResBlock(width, scene_dim, dropout) for _ in range(depth)]
        )

        # *Separate* heads help the network specialise
        self.pre_out = nn.Linear(width, pose_dim)
        self.sq_out  = nn.Linear(width, pose_dim)

        # biases = 0 → model starts by predicting zero deltas
        nn.init.zeros_(self.pre_out.weight)
        nn.init.zeros_(self.pre_out.bias)
        nn.init.zeros_(self.sq_out.weight)
        nn.init.zeros_(self.sq_out.bias)

    def forward(self, grasp, scene_feat):
        x = torch.cat([grasp, scene_feat], dim=-1)    # (B, 29+128)
        h = self.in_proj(x)                           # (B, W)
        for blk in self.blocks:
            h = blk(h, scene_feat)                   # FiLM every block

        pre_delta = self.pre_out(h)
        sq_delta  = self.sq_out(h)
        return torch.stack([pre_delta, sq_delta], dim=1)   # (B, 2, 29)

# class AuxPoseHead(nn.Module):
#     """
#     Predict (Δpre, Δsqueeze) from the sampled grasp and a
#     pooled scene feature.  All tensors are in *normalised*
#     coordinates.
#     """
#     def __init__(self, pose_dim: int = 29, scene_dim: int = 128,
#                  hidden: int = 256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(pose_dim + scene_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, 2 * pose_dim)          # stack [Δpre, Δsq]
#         )

#     def forward(self, grasp, scene_feat):
#         """
#         grasp      (B, 29) ─ normalised xyz+rot6d+22φ
#         scene_feat (B, 128) ─ global mean-pooled PointNet2 feature
#         returns    (B, 2, 29)
#         """
#         x = torch.cat([grasp, scene_feat], dim=-1)
#         delta = self.net(x)                          # (B, 58)
#         return delta.view(grasp.size(0), 2, -1)      # (B, 2, 29)


def fps_subsample(pcd: torch.Tensor,          # (B, N, 3)
                  pcd_feats: torch.Tensor,    # (B, N, F)
                  k: int):                   # points per cloud after FPS
    """
    Sub-samples a batched point cloud with farthest-point sampling (FPS)
    and gathers the matching feature tokens.

    Returns
    -------
    sub_xyz   : (B, k, 3)    coordinates after FPS
    sub_feats : (B, k, F)    features aligned with `sub_xyz`
    """
    # 1. FPS on the coordinates (pure CUDA kernel inside PyTorch3D)
    sub_xyz, idx = sample_farthest_points(pcd, K=k, random_start_point=False)           # idx shape = (B, k)

    # 2. Gather the features with the same indices
    B, k = idx.shape
    F     = pcd_feats.size(-1)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, F)        # (B, k, F)
    sub_feats    = torch.gather(pcd_feats, 1, idx_expanded)   # (B, k, F)

    return sub_xyz, sub_feats

class GraspTokenEmbedder(nn.Module):
    """
    Build a sequence of (1 + J) tokens from
        • wrist xyz (3) + 6-D rot  (9)
        • J scalar joint angles   (J,)

    Output:
        tokens      (B, 1+J, d)
        xyz_anchor  (B, 1+J, 3)  (needed for 3-D attention bias)
    """
    def __init__(self, embed_dim: int, n_joints: int = 22):
        super().__init__()

        # 1. Wrist token  (xyz + 6-D rot → d)
        self.wrist_mlp = nn.Linear(9, embed_dim)

        # 2. Joint-angle token  (scalar → d)
        self.angle_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Joint-id embedding so MCP/ PIP / DIP don’t get mixed up
        self.jid_embed = nn.Embedding(n_joints, embed_dim)

        self.n_joints = n_joints

    def forward(self, xyz: torch.Tensor, rot6d: torch.Tensor,
                angles: torch.Tensor):
        """
        xyz         (B, 3)
        rot6d       (B, 6)
        angles      (B, J)           22 joint angles
        """
        B, J = angles.shape
        assert J == self.n_joints

        # ---- wrist token --------------------------------------------------
        # print(xyz.dtype, rot6d.dtype, angles.dtype)
        wrist_feat = torch.cat([xyz, rot6d], dim=-1)            # (B, 9)
        # print(wrist_feat.dtype)# torch float 64
        wrist_tok  = self.wrist_mlp(wrist_feat)                 # (B, d)

        # ---- joint tokens -------------------------------------------------
        # expand angles to (B*J, 1) so MLP is vectorised
        ang_tok = self.angle_mlp(angles.unsqueeze(-1))          # (B, J, d)
        ang_tok = ang_tok + self.jid_embed.weight               # add id-embed

        # ---- stack --------------------------------------------------------
        tokens = torch.cat([wrist_tok.unsqueeze(1), ang_tok], dim=1)    # (B, 1+J, d)
        return tokens
    
import torch.nn as nn
import math


class RotationPredictor(nn.Module):
    """
    Projects an `embedding_dim` token into a 6-DoF rotation
    using a deep residual MLP.
    """

    def __init__(
        self,
        embedding_dim: int,
        rot_dim: int = 6,      # 6-D continuous rep (Zhou-et-al, 2019)
        n_blocks: int = 3,     # depth – >2 rarely helps here
        w: int = 4,            # hidden width multiplier
        p_drop: float = 0.1    # dropout after GELU
    ):
        super().__init__()

        self.blocks = nn.ModuleList([])
        hidden = w * embedding_dim          # e.g. 4× wider

        for _ in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(p_drop),
                    nn.Linear(hidden, embedding_dim),
                )
            )

        # final projection
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, rot_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # good defaults for residual MLPs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=1 / math.sqrt(m.in_features))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)   # pre-norm residual
        return self.head(x)

class PositionPredictor(nn.Module):
    """
    Projects an `embedding_dim` token into a 3-D position vector
    using the same residual MLP design as RotationPredictor.
    """

    def __init__(
        self,
        embedding_dim: int,
        pos_dim: int = 3,      # (x, y, z)
        n_blocks: int = 3,
        w: int = 4,
        p_drop: float = 0.1
    ):
        super().__init__()

        self.blocks = nn.ModuleList([])
        hidden = w * embedding_dim

        for _ in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(p_drop),
                    nn.Linear(hidden, embedding_dim),
                )
            )

        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, pos_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=1 / math.sqrt(m.in_features))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


def freeze_all(mod: nn.Module):
    """Helper: turn *every* Param.requires_grad off."""
    for p in mod.parameters():
        p.requires_grad = False

def unfreeze(mod: nn.Module):
    """Helper: turn Param.requires_grad back on for a sub-module."""
    for p in mod.parameters():
        p.requires_grad = True


class GraspDenoiser(nn.Module):

    def __init__(self,
                 # Encoder and decoder arguments
                 embedding_dim=128,
                 num_attn_heads=8,
                 nhist=3,
                 nhand=1,
                 pcd_feat_dim=256,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_wxyz',
                 # Denoising arguments
                 denoise_timesteps=1000,
                 denoise_model="rectified_flow",
                 # Training arguments
                 lv2_batch_size=1,
                 # hand arguments
                 xml_path='/data/user_data/austinz/Robots/DexGraspBench/assets/hand/shadow/right_hand.xml',
                 visualize_denoising_steps=False,
                 accurate_joint_pos=False,
                 guidance_weight=None,
                 ):
        super().__init__()
        # Arguments to be accessed by the main class
        self._rotation_format = rotation_format
        self._relative = relative
        self._lv2_batch_size = lv2_batch_size
        dtype = torch.float32
        self.dtype = dtype
        self.timers = defaultdict(float)
        self._profiling = False
        self.visualize_denoising_steps = visualize_denoising_steps
        self.visualization_data = {} # "partial_points": [B, N, 3], "grasps": [B, M, 29]}
        self.accurate_joint_pos = accurate_joint_pos
        
        self.guidance_weight = guidance_weight

        # Vision-language encoder, runs only once
        self.encoder = SceneEncoder(pcd_feat_dim, embedding_dim)
        # self.pcd_encoder = PointNet2Backbone(
        #         radii=[0.1, 0.2],
        #         nsamples=[16, 32],
        #         mlps=[[0, 32, 32, 64], [0, 64, 64, 128]],
        #         out_dim=128,
        #             )
        
        # ckpt_path = hf_hub_download(
        #     repo_id="BAAI/Uni3D",
        #     filename="model.pt",
        #     subfolder="modelzoo/uni3d-g",      # <- same sub-folder the paper uses
        #     cache_dir="./checkpoints/uni3d"
        # )
        
        self.uni3d_model = create_uni3d(size_tag='b').eval().to('cuda')
        
        # (a) freeze all weights first
        freeze_all(self.uni3d_model)

        # (b) un-freeze just the three FP blocks
        unfreeze(self.uni3d_model.point_encoder.fp1)
        unfreeze(self.uni3d_model.point_encoder.fp2)
        unfreeze(self.uni3d_model.point_encoder.fp3)
        
        # trainable = []
        # for name, param in self.uni3d_model.named_parameters():
        #     if name.startswith(("blocks.10", "blocks.11", "point_encoder")):
        #         param.requires_grad_(True)
        #         trainable.append(param)
        # self.uni3d_model.point_encoder.train()
        
        
        self.pcd_proj = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
                
        
        self.xml_path = xml_path
        model = mujoco.MjModel.from_xml_path(xml_path)
        joint_names = [mj_id2name(model, mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        # joint_names = list(model.joint_names)              # tuple → Python list
        joint_types = model.jnt_type                       # (njoint,) 0=free,3=hinge
        joint_range = model.jnt_range.copy()               # (njoint, 2)  [min, max] rad

        # Filter to the 22 finger DOF (hinge joints, not the 6-DoF free wrist)
        HINGE = mujoco.mjtJoint.mjJNT_HINGE
        finger_idx = np.where(joint_types == HINGE)[0]

        finger_names = [joint_names[i] for i in finger_idx]
        self.jmin    = joint_range[finger_idx, 0]          # radians
        self.jmax    = joint_range[finger_idx, 1]
        self.jmin    = torch.tensor(self.jmin).to('cuda').to(dtype)    # (22,)
        self.jmax    = torch.tensor(self.jmax).to('cuda').to(dtype)    # (22,)
        
        self.chain = build_chain_from_mjcf_path(xml_path)
        self.chain = self.chain.to(dtype=dtype, device='cuda')
        lower_limits, upper_limits = self.chain.get_joint_limits()
        # print("lower limits", lower_limits)
        # print("upper limits", upper_limits)
        # print(self.jmin, self.jmax, 'jmin, jmax')
        lower_limits = torch.tensor(lower_limits, dtype=dtype, device='cuda')
        upper_limits = torch.tensor(upper_limits, dtype=dtype, device='cuda')
        
        

        # print("22 finger joints:", finger_names)
        # print("lower limits   :", self.jmin)
        # print("upper limits   :", self.jmax)

        # Action decoder, runs at every denoising timestep
        self.grasp_encoder = GraspTokenEmbedder(
            embed_dim=embedding_dim, n_joints=22
        )
        # print("embedding dim", embedding_dim)
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format == 'euler' else 6,
            angle_dim=22,
        )

        # Noise/denoise schedulers and hyperparameters
        self.position_scheduler, self.rotation_scheduler, self.angle_scheduler = fetch_schedulers(
            denoise_model, denoise_timesteps
        )
        self.n_steps = denoise_timesteps

        # Normalization for the 3D space, will be loaded in the main process
        if rotation_format == 'euler':  # normalize pos+rot
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0, 0, 0, 0, 0], [1., 1, 1, 1, 1, 1]]),
                requires_grad=False
            )
        else:
            distance = 0.3
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[-distance, -distance, -distance], [distance, distance, distance]]),
                requires_grad=False
            )
                    
        self.pose_dim   = 31                                 # 3 + 6 + 22
        self.aux_head   = AuxPoseHead(self.pose_dim,
                                    scene_dim=embedding_dim)  # 128 by default
        self.aux_w_pre  = 0.01                              # λ weights – tune
        self.aux_w_sq   = 0.001
        

    def encode_inputs(self, pcd, uni3d=True):
        _t0 = self._start_timer()
        if not uni3d:
            pcd_feats = self.pcd_encoder(pcd)
            # print(pcd_feats.shape, 'pcd feats')
            fixed_inputs = self.encoder(
                pcd_feats
            )
            return fixed_inputs # if use this, please change the pcd_feat_dim to 128
        
        else:
            """
            pcd : (B, N, 3)  world-space cloud
            return : (B, N, embed_dim)
            """
            xyz_rgb  = prepare_pc(pcd)                      # add dummy RGB
            with torch.cuda.amp.autocast():                 # Uni3D is fp16-safe
                feats256 = self.uni3d_model.encode_pc(xyz_rgb)['points']   # (B, N, 256)
            feats128 = self.pcd_proj(feats256)              # (B, N, 128) ↦ match rest of pipeline
            self._stop_timer(_t0, "encode_pc")
            return feats128
    
    def _predict_aux(self, grasp_norm, scene_feat):
        _t0 = self._start_timer()
        """
        Inputs/outputs are *normalised*.
        """
        # print(grasp_norm.shape, scene_feat.shape, 'grasp norm, scene feat')
        delta = self.aux_head(grasp_norm, scene_feat)     # (B, 2, 29)
        pre_norm     = grasp_norm + delta[:, 0]
        squeeze_norm = grasp_norm + delta[:, 1]
        self._stop_timer(_t0, "aux_head")
        return pre_norm, squeeze_norm
    
    def fk_layer(self, grasp, accurate_pos=True):
        if accurate_pos:
            # Implement Forward Kinematics (FK) to compute joint positions!
            xyz = get_joint_positions(self.chain, grasp, pose_normalized=False) # (B, 22, 3)
            # print(xyz.shape, 'xyz shape in fk_layer')
            if len(xyz.shape) == 2:
                xyz = xyz.unsqueeze(0)
            return xyz
        else:
            xyz = grasp[..., :3]
            joint_xyz = xyz.unsqueeze(1).expand(-1,22,3)
        return joint_xyz

    def policy_forward_pass(self, grasp, timestep, fixed_inputs, grasp_type_id, train):
        _t0 = self._start_timer()
        # Parse inputs
        pcd_feats, pcd, focus_idx = fixed_inputs

        # FK for joint xyz (optional)
        actual_grasp = grasp.detach()  # (B, 3+6+J)
        actual_grasp = self.unnormalize_pos(actual_grasp)
        actual_grasp = self.unconvert_rot(actual_grasp)  # (B, 3+4+J)
        # print(actual_grasp.dtype)
        actual_grasp = self.unnormalize_angles(actual_grasp)
        # print(actual_grasp.dtype, 'after unnorm')
        # print(actual_grasp.shape)
        assert actual_grasp.shape[-1] == 29
        # print(actual_grasp.shape, 'actual grasp shape')
        actual_joint_xyz = self.fk_layer(actual_grasp, accurate_pos=self.accurate_joint_pos)           # (B, 22, 3)
        # print(actual_joint_xyz.shape, 'actual joint xyz shape')
        grasp_feats = self.grasp_encoder(grasp[..., :3], grasp[..., 3:9], grasp[..., 9:]) # (B, 23, d) 
        # But use positions from unnormalized absolute trajectory
        grasp_xyzs = torch.cat([actual_grasp[..., :3].unsqueeze(1), actual_joint_xyz], dim=1)     # (B, 1+J, 3)
        out = self.prediction_head(
            grasp_feats,
            grasp_xyzs,
            timestep,
            pcd_feats,
            pcd,
            focus_idx=focus_idx,
            grasp_type_id=grasp_type_id,
            train=train
        )
        
        self._stop_timer(_t0, "transformer_head")
        return out

    def conditional_sample(self, grasp, device, fixed_inputs, grasp_type_id):
        # Set schedulers
        self.position_scheduler.set_timesteps(self.n_steps, device=device)

        # Iterative denoising
        timesteps = self.position_scheduler.timesteps
        # print(timesteps, 'timesteps')
        for t_ind, t in enumerate(timesteps):
            t_batch = t * torch.ones(len(grasp), device=device, dtype=torch.long)
            if self.guidance_weight is not None:  # e.g., 1.5
                pred_uncond = self.policy_forward_pass(grasp, t_batch, fixed_inputs,
                                                    grasp_type_id=None, train=False)[-1]
                pred_cond   = self.policy_forward_pass(grasp, t_batch, fixed_inputs,
                                                    grasp_type_id=grasp_type_id, train=False)[-1]
                pred = pred_uncond + self.guidance_weight * (pred_cond - pred_uncond)
            else:
                pred = self.policy_forward_pass(grasp, t_batch, fixed_inputs,
                                                grasp_type_id=grasp_type_id, train=False)[-1]
            grasp = self.position_scheduler.step(pred, t_ind, grasp).prev_sample
            
            # out = self.policy_forward_pass(
            #     grasp,
            #     t * torch.ones(len(grasp)).to(device).long(),
            #     fixed_inputs, grasp_type_id
            # )
            # out = out[-1]  # keep only last layer's output
            # grasp = self.position_scheduler.step(
            #     out,
            #     t_ind, grasp).prev_sample
            
            # print(grasp.shape, 'grasp shape') # [B, 31]
            # Back to quaternion
            if self.visualize_denoising_steps:
                grasp_unnorm = self.unconvert_rot(grasp)
                grasp_unnorm = self.unnormalize_pos(grasp_unnorm)
                grasp_unnorm = self.unnormalize_angles(grasp_unnorm)
                # print(grasp_unnorm.shape, 'grasp unnorm shape') # [B, 29]
                if t_ind == 0:
                    self.visualization_data["partial_points"] = fixed_inputs[1].cpu().numpy()
                    self.visualization_data["grasps"] = np.expand_dims(grasp_unnorm.cpu().numpy(), axis=1)  # (B, 1, 29)
                    self.visualization_data["joint_positions"] = np.expand_dims(self.fk_layer(grasp_unnorm, accurate_pos=True).cpu().numpy(), axis=1) # (B, 1, 22, 3)
                else:
                    self.visualization_data["grasps"] = np.concatenate(
                        [self.visualization_data["grasps"], np.expand_dims(grasp_unnorm.cpu().numpy(), axis=1)],
                        axis=1
                    ) # (B, T+1, 29)
                    self.visualization_data["joint_positions"] = np.concatenate(
                        [self.visualization_data["joint_positions"], np.expand_dims(self.fk_layer(grasp_unnorm, accurate_pos=True).cpu().numpy(), axis=1)],
                        axis=1)


        return grasp

    def compute_grasp(self, pcd, focus_idx, grasp_type_id=None):
        # Encode observations, states, instructions
        fixed_inputs = (self.encode_inputs(pcd), pcd, focus_idx)

        # Sample from learned model starting from noise
        out_dim = 6 if self._rotation_format == 'euler' else 9
        grasp = torch.randn(
            size=(pcd.shape[0], out_dim + 22),
            device=pcd.device
        )
        grasp = self.conditional_sample(
            grasp,
            device=pcd.device,
            fixed_inputs=fixed_inputs,
            grasp_type_id=grasp_type_id
        )
        
        
        scene_feat  = fixed_inputs[0].mean(dim=1)
        pre_norm, sq_norm = self._predict_aux(grasp, scene_feat)

        # Back to quaternion
        grasp = self.unconvert_rot(grasp)
        # unnormalize position
        grasp = self.unnormalize_pos(grasp)
        # unnormalize angles
        grasp = self.unnormalize_angles(grasp)
        
        pregrasp = self.unnormalize_pos(pre_norm)
        pregrasp = self.unconvert_rot(pregrasp)
        pregrasp = self.unnormalize_angles(pregrasp)
        
        squeeze = self.unnormalize_pos(sq_norm)
        squeeze = self.unconvert_rot(squeeze)
        squeeze = self.unnormalize_angles(squeeze)
        
        return grasp, pregrasp, squeeze

    def compute_loss(self, gt_grasp, gt_pregrasp, gt_squeeze, pcd, focus_idx, grasp_type_id=None):
        # Encode observations, states, instructions
        fixed_inputs = (self.encode_inputs(pcd), pcd, focus_idx)
        # print(gt_grasp)

        # Normalize all pos (order matters)
        gt_grasp = self.normalize_pos(gt_grasp)
        gt_grasp = self.convert_rot(gt_grasp)
        gt_grasp = self.normalize_angles(gt_grasp)
        gt_pregrasp = self.normalize_pos(gt_pregrasp)
        gt_pregrasp = self.convert_rot(gt_pregrasp)
        gt_pregrasp = self.normalize_angles(gt_pregrasp)
        gt_squeeze = self.normalize_pos(gt_squeeze)
        gt_squeeze = self.convert_rot(gt_squeeze)
        gt_squeeze = self.normalize_angles(gt_squeeze)
        # print(gt_grasp[0])
        # print(gt_grasp[1])
        # print(gt_grasp[2])
        # print(gt_grasp[3])
        # print(gt_grasp[4])
        # print(gt_grasp[5])
        # assert torch.max(gt_grasp[..., :3]) <= 1.0, "Pos normalization failed!"
        # assert torch.min(gt_grasp[..., :3]) >= -1.0, "Pos normalization failed!"
        # assert torch.max(gt_grasp[..., 3:9]) <= 1.0, "Rot normalization failed!"
        # assert torch.min(gt_grasp[..., 3:9]) >= -1.0, "Rot normalization failed!"
        # assert torch.max(gt_grasp[..., 9:]) <= 1.3, "Angle normalization failed!"
        # assert torch.min(gt_grasp[..., 9:]) >= -1.3, "Angle normalization failed!"
        
        

        # Loop lv2_batch_size times and sample different noises with same input
        # Trick to effectively increase the batch size without re-encoding
        total_loss = 0
        for _ in range(self._lv2_batch_size):
            # Sample noise
            noise = torch.randn(gt_grasp.shape, device=gt_grasp.device)
            # Sample a random timestep
            # breakpoint()
            timesteps = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )

            noisy_grasp = self.position_scheduler.add_noise(
                gt_grasp, noise,
                timesteps
            )

            # Predict the noise residual
            pred = self.policy_forward_pass(
                noisy_grasp,
                timesteps, fixed_inputs, grasp_type_id, train=True
            )

            # Compute loss
            for layer_pred in pred:
                pos = layer_pred[..., :3]
                rot = layer_pred[..., 3:9]
                angles = layer_pred[..., 9:]
                denoise_target = self.position_scheduler.prepare_target(
                    noise, gt_grasp 
                ) # default gt_grasp itself (or noise)
                # print(layer_pred.shape, denoise_target.shape, 'pred, target shape')
                pos_loss = 30 * F.l1_loss(pos, denoise_target[..., :3], reduction='mean')
                rot_loss = 10 * F.l1_loss(rot, denoise_target[..., 3:9], reduction='mean')
                angle_loss = 10 * F.l1_loss(angles, denoise_target[..., 9:], reduction='mean')
                loss = pos_loss + rot_loss + angle_loss
                
                scene_feat = fixed_inputs[0].mean(dim=1)
                pre_pred, sq_pred = self._predict_aux(gt_grasp, scene_feat)
                pre_loss = self._pose_l1(pre_pred, gt_pregrasp)
                sq_loss  = self._pose_l1(sq_pred,  gt_squeeze)

                loss_aux = self.aux_w_pre * pre_loss + self.aux_w_sq * sq_loss
                
                loss = loss + loss_aux
                
                # print(denoise_target)
                # print(loss.item(), 'loss at this step')
                # print('pos loss:', pos_loss.item(), 'rot loss:', rot_loss.item(), 'angle loss:', angle_loss.item(), 'aux loss:', loss_aux.item())
                # print('avg norm:', avg_norm.item(), 'max norm:', max_norm.item())
                total_loss = total_loss + loss
        return total_loss / self._lv2_batch_size


    def normalize_pos(self, signal):
        n = min(self.workspace_normalizer.size(-1), signal.size(-1))
        _min = self.workspace_normalizer[0][:n].float()
        _max = self.workspace_normalizer[1][:n].float()
        return torch.cat((
            (signal[..., :n] - _min) / (_max - _min) * 2.0 - 1.0,
            signal[..., n:]
        ), -1)

    def unnormalize_pos(self, signal):
        n = min(self.workspace_normalizer.size(-1), signal.size(-1))
        _min = self.workspace_normalizer[0][:n].float()
        _max = self.workspace_normalizer[1][:n].float()
        return torch.cat((
            (signal[..., :n] + 1.0) / 2.0 * (_max - _min) + _min,
            signal[..., n:]
        ), -1)
        
    def normalize_angles(self, signal):
        # Normalize angles to [-1, 1]
        assert signal.size(-1) > 9
        if signal.size(-1) > 9:
            angles = signal[..., 9:]
            angles = (angles - self.jmin) / (self.jmax - self.jmin) * 2.0 - 1.0
            angles = torch.clamp(angles, min=-1.0, max=1.0) 
            return torch.cat((signal[..., :9], angles), -1)
        return signal

    def unnormalize_angles(self, signal):
        # Unnormalize angles from [-1, 1] to [jmin, jmax]
        if signal.size(-1) == 29:
            angles = signal[..., 7:]
            angles = (angles + 1.0) / 2.0 * (self.jmax - self.jmin) + self.jmin
            return torch.cat((signal[..., :7], angles), -1)
        return signal

    def convert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        rot = normalise_quat(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        # The following code expects wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        # Convert to rotation matrix
        rot = quaternion_to_matrix(rot)
        # Convert to 6D
        if len(rot.shape) == 4:
            B, L, D1, D2 = rot.shape
            rot = rot.reshape(B * L, D1, D2)
            rot = get_ortho6d_from_rotation_matrix(rot)
            rot = rot.reshape(B, L, 6)
        else:
            rot = get_ortho6d_from_rotation_matrix(rot)
        # Concatenate pos, rot, other state info
        signal = torch.cat([signal[..., :3], rot], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        res = signal[..., 9:] if signal.size(-1) > 9 else None
        if len(signal.shape) == 3:
            B, L, _ = signal.shape
            rot = signal[..., 3:9].reshape(B * L, 6)
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
            quat = quat.reshape(B, L, 4)
        else:
            rot = signal[..., 3:9]
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
        # The above code handled wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (1, 2, 3, 0)]
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal
    
    def _pose_l1(self, pred, target):
        pos_l  = 30 * F.l1_loss(pred[...,:3],  target[...,:3])
        rot_l  = 10 * F.l1_loss(pred[...,3:9], target[...,3:9])
        ang_l  = 10 * F.l1_loss(pred[...,9:],  target[...,9:])
        return pos_l + rot_l + ang_l
    
    def _start_timer(self):
        """Return (event/time) object marking the start of a region."""
        if not self._profiling:
            return None
        if torch.cuda.is_available():
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()
    
    def _stop_timer(self, start, key):
        """Accumulate elapsed time into self.timers[key]."""
        if (not self._profiling) or (start is None):
            return
        if torch.cuda.is_available():
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            self.timers[key] += start.elapsed_time(end)      # ms
        else:
            self.timers[key] += (time.perf_counter() - start) * 1e3  # ms
            
    def report_timers(self, reset=False):
        """Return a nicely formatted string and optionally reset counters."""
        rep = "⊛ Profiling (cumulative ms): " + ", ".join(
            f"{k} = {v:8.2f}" for k, v in self.timers.items()
        )
        if reset:
            self.timers.clear()
        return rep


    def forward(
        self,
        gt_grasp,
        gt_pregrasp, 
        gt_squeeze,
        pcd,
        focus_idx=None,
        grasp_type_id=None,
        run_inference=False
        ):
        """
        Arguments:
            gt_grasp: (B, 3+4+J)
            pcd: (B, N, 3) in world coordinates
            focus_idx: int, index of the corresponding point

        Note:
            The input rotation is expressed as quaternion (4D), then the model converts it to 6D internally.

        Returns:
            - loss: scalar, if run_inference is False
            - grasp: (B, 3+4+J), at inference
        """
        # Inference, don't use gt_grasp
        if run_inference:
            return self.compute_grasp(
                pcd, focus_idx, grasp_type_id=grasp_type_id
            )

        # Training, use gt_grasp to compute loss
        return self.compute_loss(
            gt_grasp, gt_pregrasp, gt_squeeze, pcd, focus_idx, grasp_type_id=grasp_type_id
        )


class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=128,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 num_shared_attn_layers_head=20,
                 nhist=3,
                 rotary_pe=True,
                 rot_dim=6,
                 angle_dim=22):
        super().__init__()
        
        embedding_dim_fw = embedding_dim * 4  # Feed-forward dimension

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Fuse (time, class) → embedding_dim
        self.cond_fuse = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        num_grasp_types = 34
        self.type_embed = nn.Embedding(num_grasp_types, embedding_dim)

        # For classifier-free guidance (optional):
        self.null_type = nn.Parameter(torch.zeros(embedding_dim))  # learned null vector (better than pure zeros)
        self.class_dropout_prob = 0.1  # p to drop conditioning during training

        # Estimate attends to context (no subsampling)
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim_fw,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=False
        )

        # Shared attention layers
        self.self_attn = nn.ModuleList([
            AttentionModule(
                num_layers=1,
                d_model=embedding_dim,
                dim_fw=embedding_dim_fw,
                dropout=0.1,
                n_heads=num_attn_heads,
                pre_norm=False,
                rotary_pe=rotary_pe,
                use_adaln=True,
                is_self=False
            )
            for _ in range(num_shared_attn_layers)
        ])

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        # self.rotation_self_attn = AttentionModule(
        #     num_layers=2,
        #     d_model=embedding_dim,
        #     dim_fw=embedding_dim_fw,
        #     dropout=0.1,
        #     n_heads=num_attn_heads,
        #     pre_norm=False,
        #     rotary_pe=rotary_pe,
        #     use_adaln=True,
        #     is_self=True
        # )
        self.rotation_self_attn = nn.ModuleList([
            AttentionModule(
                num_layers=1,
                d_model=embedding_dim,
                dim_fw=embedding_dim_fw,
                dropout=0.1,
                n_heads=num_attn_heads,
                pre_norm=False,
                rotary_pe=rotary_pe,
                use_adaln=True,
                is_self=False
            )
            for _ in range(num_shared_attn_layers_head)
        ])
        # self.rotation_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, rot_dim)
        # )
        self.rotation_predictor = RotationPredictor(embedding_dim)

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        # self.position_self_attn = AttentionModule(
        #     num_layers=2,
        #     d_model=embedding_dim,
        #     dim_fw=embedding_dim_fw,
        #     dropout=0.1,
        #     n_heads=num_attn_heads,
        #     pre_norm=False,
        #     rotary_pe=rotary_pe,
        #     use_adaln=True,
        #     is_self=True
        # )
        # self.position_self_attn = nn.ModuleList([
        #     AttentionModule(
        #         num_layers=1,
        #         d_model=embedding_dim,
        #         dim_fw=embedding_dim_fw,
        #         dropout=0.1,
        #         n_heads=num_attn_heads,
        #         pre_norm=False,
        #         rotary_pe=rotary_pe,
        #         use_adaln=True,
        #         is_self=False
        #     )
        #     for _ in range(num_shared_attn_layers_head)
        # ])
        # self.position_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, 3)
        # )
        self.position_predictor = PositionPredictor(embedding_dim)

        # 3. Joints
        self.angle_proj = nn.Linear(embedding_dim, embedding_dim)
        # self.angle_self_attn = AttentionModule(
        #     num_layers=2,
        #     d_model=embedding_dim,
        #     dim_fw=embedding_dim_fw,
        #     dropout=0.1,
        #     n_heads=num_attn_heads,
        #     pre_norm=False,
        #     rotary_pe=rotary_pe,
        #     use_adaln=True,
        #     is_self=True
        # )
        # self.angle_self_attn = nn.ModuleList([
        #     AttentionModule(
        #         num_layers=1,
        #         d_model=embedding_dim,
        #         dim_fw=embedding_dim_fw,
        #         dropout=0.1,
        #         n_heads=num_attn_heads,
        #         pre_norm=False,
        #         rotary_pe=rotary_pe,
        #         use_adaln=True,
        #         is_self=False
        #     )
        #     for _ in range(num_shared_attn_layers_head)
        # ])
        self.angle_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
        # focus on the point
        self.focus_embed = nn.Parameter(torch.zeros(embedding_dim))    # ΔE_focus
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
    
    def build_condition(self, timesteps, grasp_type_id, train: bool):
        time_embs = self.time_emb(timesteps)              # (B, d)
        if grasp_type_id is None:
            # Unconditional path (e.g., inference w/o class)
            class_emb = self.null_type.unsqueeze(0).expand(time_embs.size(0), -1)
        else:
            class_emb = self.type_embed(grasp_type_id)    # (B, d)
            # print(grasp_type_id, 'grasp type id shape')
            if train and self.class_dropout_prob > 0:
                drop_mask = (torch.rand_like(grasp_type_id.float()) < self.class_dropout_prob)
                # Replace dropped examples with null_type
                if drop_mask.any():
                    class_emb[drop_mask] = self.null_type
                    # print('using grasp type id, but some are dropped')
                    # print(class_emb[drop_mask].shape)
        fused = self.cond_fuse(torch.cat([time_embs, class_emb], dim=-1))
        return fused, class_emb  # shape (B, d)


    def forward(self, grasp_feats, grasp_xyzs, timesteps, pcd_feats, pcd, focus_idx=None, grasp_type_id=None, train=True):
        """
        Arguments:
            traj_feats: (B, trajectory_length, nhand, F)
            trajectory: (B, trajectory_length, nhand, 3+6+X)
            timesteps: (B, 1)
            rgb3d_feats: (B, N, F)
            rgb3d_pos: (B, N, 3)
            rgb2d_feats: (B, N2d, F)
            rgb2d_pos: (B, N2d, 3)
            instr_feats: (B, L, F)
            instr_pos: (B, L, 3)
            proprio_feats: (B, nhist*nhand, F)
            fps_scene_feats: (B, M, F), M < N
            fps_scene_pos: (B, M, 3)
            
        Arguments:
            grasp_feats: (B, 1+J, F)
            grasp_xyzs: (B, 1+J, 3)
            timesteps: (B, 1)
            pcd_feats: (B, N, F)
            pcd: (B, N, 3)
            
        Returns:
            list of (B, trajectory_length, nhand, 3+6+X)
        """
        B, J1, F = grasp_feats.shape           # J1 = 1 + J
        N = pcd_feats.shape[1]
        S = J1 + N                             # total sequence length
        
        pcd_feats = pcd_feats.clone()          # avoid in-place on input
        # if focus_idx == 0: # if it is -1, then no anchor is visible
        #     print('yes')
        #     pcd_feats[:, focus_idx] += self.focus_embed    # (B, d)
        mask = focus_idx.bool()
        pcd_feats[mask, 0] += self.focus_embed
        
        pcd, pcd_feats = fps_subsample(pcd, pcd_feats, k=1024)
        # print(pcd.shape, pcd_feats.shape, 'pcd shape, pcd feats shape')

        # Denoising timesteps' embeddings
        # time_embs = self.encode_denoising_timestep(timesteps)
        time_embs, class_emb = self.build_condition(
            timesteps=timesteps,
            grasp_type_id=grasp_type_id,
            train=train
        )  # (B, F)
        if grasp_type_id is not None:
            class_shift = class_emb.unsqueeze(1)  # (B, 1, d)
            grasp_feats = grasp_feats + class_shift

        # Positional embeddings
        rel_grasp_pos, rel_pcd_pos, rel_pos = self.get_positional_embeddings(grasp_xyzs, pcd)

        # Cross attention from gripper to full context
        grasp_feats = self.cross_attn(
            seq1=grasp_feats,
            seq2=pcd_feats,
            seq1_pos=rel_grasp_pos,
            seq2_pos=rel_pcd_pos,
            ada_sgnl=time_embs
        )[-1]
        

        # # Self attention among gripper and sampled context
        # features = self.get_sa_feature_sequence(grasp_feats, pcd_feats)
        # features = self.self_attn(
        #     seq1=features,
        #     seq2=features,
        #     seq1_pos=rel_pos,
        #     seq2_pos=rel_pos,
        #     ada_sgnl=time_embs
        # )[-1]
        
        for layer in self.self_attn:
            grasp_feats = layer(
                seq1=grasp_feats,
                seq2=torch.cat((pcd_feats, grasp_feats), 1),
                seq1_pos=rel_grasp_pos,
                seq2_pos=torch.cat((rel_pcd_pos, rel_grasp_pos), 1),
                ada_sgnl=time_embs
            )[-1]
            
        wrist_features = grasp_feats[:, 0:1, :]
        rel_pos_pos = rel_grasp_pos[:, 0:1, :]  # (B, 1, 3)
        for layer in self.rotation_self_attn:
            wrist_features = layer(
                seq1=wrist_features,
                seq2=torch.cat((pcd_feats, wrist_features), 1),
                seq1_pos=rel_pos_pos,
                seq2_pos=torch.cat((rel_pcd_pos, rel_pos_pos), 1),
                ada_sgnl=time_embs
            )[-1]
        
        # features = self.self_attn(
        #     seq1=features,
        #     seq2=features,
        #     seq1_pos=None,
        #     seq2_pos=None,
        #     ada_sgnl=time_embs
        # )[-1]
        
        # print(features.shape, 'all features, B 1+J+1024 F')

        # Rotation head
        rotation = self.predict_rot(
            wrist_features, pcd_feats, rel_pcd_pos, rel_grasp_pos, time_embs, grasp_feats.shape[1]
        )

        # Position head
        position = self.predict_pos(
            wrist_features, pcd_feats, rel_pcd_pos, rel_grasp_pos, time_embs, grasp_feats.shape[1]
        )

        # Joint angles head
        angles = self.predict_ang(
            grasp_feats, pcd_feats, rel_pcd_pos, rel_grasp_pos, time_embs, grasp_feats.shape[1]
        )

        return [
            torch.cat((position, rotation, angles), -1)
        ]

    def encode_denoising_timestep(self, timestep):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        return time_feats

    def get_positional_embeddings(
        self,
        grasp_xyzs, pcd
    ):
        rel_grasp_pos = self.relative_pe_layer(grasp_xyzs)
        rel_pcd_pos = self.relative_pe_layer(pcd)
        # rel_fps_pos = self.relative_pe_layer(fps_scene_pos)
        rel_pos = torch.cat([rel_grasp_pos, rel_pcd_pos], 1)
        return rel_grasp_pos, rel_pcd_pos, rel_pos

    def get_sa_feature_sequence(
        self,
        grasp_feats, pcd_feats
    ):
        return torch.cat([grasp_feats, pcd_feats], 1)

    def predict_pos(self, wrist_features, pcd_feats, rel_pcd_pos, rel_grasp_pos, time_embs, traj_len):
        # position_features = grasp_feats[:, 0:1, :]
        # rel_pos_pos = rel_grasp_pos[:, 0:1, :]  # (B, 1, 3)
        # for layer in self.rotation_self_attn:
        #     position_features = layer(
        #         seq1=position_features,
        #         seq2=torch.cat((pcd_feats, position_features), 1),
        #         seq1_pos=rel_pos_pos,
        #         seq2_pos=torch.cat((rel_pcd_pos, rel_pos_pos), 1),
        #         ada_sgnl=time_embs
        #     )[-1]
        # print(position_features.shape, 'position features shape')
        wrist_features = wrist_features[:, 0, :]
        wrist_features = self.position_proj(wrist_features)  # (B, N, C)
        # print(position_features.shape, 'position features after proj')
        position = self.position_predictor(wrist_features)
        # print(position.shape, 'position shape')
        return position

    def predict_rot(self, wrist_features, pcd_feats, rel_pcd_pos, rel_grasp_pos, time_embs, traj_len):
        # rotation_features = grasp_feats[:, 0:1, :]
        # rel_rot_pos = rel_grasp_pos[:, 0:1, :]  # (B, 1, 3)
        # for layer in self.rotation_self_attn:
        #     rotation_features = layer(
        #         seq1=rotation_features,
        #         seq2=torch.cat((pcd_feats, rotation_features), 1),
        #         seq1_pos=rel_rot_pos,
        #         seq2_pos=torch.cat((rel_pcd_pos, rel_rot_pos), 1),
        #         ada_sgnl=time_embs
        #     )[-1]
        # print(rotation_features.shape, 'rotation features shape')
        wrist_features = wrist_features[:, 0, :]
        wrist_features = self.rotation_proj(wrist_features)  # (B, N, C)
        # print(rotation_features.shape, 'rotation features after proj')
        rotation = self.rotation_predictor(wrist_features)
        # print(rotation.shape, 'rotation shape')
        return rotation
    
    def predict_ang(self, grasp_feats, pcd_feats, rel_pcd_pos, rel_grasp_pos, time_embs, traj_len):
        # angle_features = self.angle_self_attn(
        #     seq1=features,
        #     seq2=features,
        #     seq1_pos=pos,
        #     seq2_pos=pos,
        #     ada_sgnl=time_embs
        # )[-1]                          # (B, S, C)
        angle_features = grasp_feats[:, 1:traj_len, :]
        rel_angle_pos = rel_grasp_pos[:, 1:traj_len, :]
        # for layer in self.angle_self_attn:
        #     angle_features = layer(
        #         seq1=angle_features,
        #         seq2=torch.cat((pcd_feats, angle_features), 1),
        #         seq1_pos=rel_angle_pos,
        #         seq2_pos=torch.cat((rel_pcd_pos, rel_angle_pos), 1),
        #         ada_sgnl=time_embs
        #     )[-1]
        # angle_features = angle_features[:, 1:traj_len, :]
        assert angle_features.shape[1] == 22
        angle_features = self.angle_proj(angle_features)  # (B, C)
        angles = self.angle_predictor(angle_features)     # (B, J, 1)
        return angles.squeeze(-1)



from collections import OrderedDict
if __name__ == "__main__":
    # Test the GraspDenoiser class
    def trainable_params(m: torch.nn.Module) -> int:
        """Number of parameters that will be updated by the optimiser."""
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    model = GraspDenoiser()
    modules = OrderedDict(
        SceneEncoder        = model.encoder,          # (B, N, P) → (B, N, d)
        PointNet2Backbone   = model.pcd_encoder,      # point-cloud feature extractor
        GraspTokenEmbedder  = model.grasp_encoder,    # wrist + joint tokens
        TransformerHead     = model.prediction_head,  # the large decoder stack
        AuxPoseHead         = model.aux_head,         # auxiliary Δpre / Δsq head
        Uni3D              = model.uni3d_model,           # uncomment if you load it
    )
    totals = {name: trainable_params(mod) for name, mod in modules.items()}
    totals["TOTAL (model)"] = trainable_params(model)
    
    width   = max(len(k) for k in totals) + 2
    header  = f'{"Module":{width}} #Params    (M)'
    print('=' * (width + 20))
    print(header)
    print('=' * (width + 20))
    for k, v in totals.items():
        print(f'{k:{width}} {v:>12,}  ({v/1e6:5.2f} M)')
    print('=' * (width + 20))
    # print(model)
    # Add more tests as needed