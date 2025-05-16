import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseTrainTester


class CALVINTrainTester(BaseTrainTester):

    @torch.no_grad()
    def get_workspace_normalizer(self):
        return nn.Parameter(torch.tensor([
            [-0.0522, -0.0433, -0.0457, -0.1683, -0.1089, -0.2247],
            [ 0.0606,  0.0364,  0.0621,  0.1001,  0.1260,  0.1412]
        ]), requires_grad=False)
        # return super().get_workspace_normalizer(ndims=6)

    def _run_depth2cloud(self, sample):
        pcd, pcd_w = self.depth2cloud(
            sample['depth'].cuda(non_blocking=True).float()[:, 0],  # one camera
            sample['wrist_depth'].cuda(non_blocking=True).float()[:, 0],
            sample['extrinsics_wrist'].cuda(non_blocking=True).float()
        )
        return pcd[:, None], pcd_w[:, None]  # B 1 3 H W

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        # Actions
        if self.args.keypose_only:
            sample["action"] = sample["action"][:, [-1]]

        # Observations
        pcds, pcd_w = self._run_depth2cloud(sample)
        if augment:
            b, nc, _, h, w = sample['rgb'].shape
            obs = torch.cat((
                sample['rgb'].cuda(non_blocking=True).half() / 255,
                pcds.half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)
            obs = self.aug(obs)
            rgbs = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcds = obs[:, 3:].reshape(b, nc, 3, h, w).float()
        else:
            rgbs = sample['rgb'].cuda(non_blocking=True).float() / 255
        # Upsample to 224x224
        rgbs = F.interpolate(rgbs[:, 0], (224, 224), mode='bilinear', antialias=True)[:, None]
        # Handle wrist camera
        pcd_w = pcd_w.float()
        h, w = rgbs.shape[-2:]
        # pcd_w = F.interpolate(pcd_w[:, 0], (h, w), mode='bilinear')[:, None]
        # pcds = torch.cat((pcds, pcd_w), 1)
        rgb_w = sample["rgb2d"].cuda(non_blocking=True).float() / 255
        rgb_w = F.interpolate(rgb_w[:, 0], (h, w), mode='bilinear', antialias=True)[:, None]
        rgbs = torch.cat((rgbs, rgb_w), 1)

        # Check for history requirements
        proprio = sample["proprioception"].cuda(non_blocking=True)
        nhist_ = proprio.size(1)  # proprio is B nhist nhand 7+X
        assert nhist_ >= self.args.num_history, "not enough proprio timesteps"
        proprio = proprio[:, :max(self.args.num_history, 1)]

        return (
            sample["action"].cuda(non_blocking=True),
            torch.zeros(sample["action"].shape[:-1], dtype=bool, device='cuda'),
            rgbs,
            None,
            pcds,
            sample["instr"],
            proprio
        )

    def _model_forward(self, model, sample, training=True):
        action, action_mask, rgbs, rgb2d, pcds, instr, prop = self.prepare_batch(
            sample, augment=training
        )
        # from time import time
        # torch.cuda.synchronize()
        # start = time()
        # instr = self.tokenizer(instr).cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                action, action_mask, rgbs, rgb2d, pcds, instr, prop,
                run_inference=not training
            )
        # torch.cuda.synchronize()
        # print("Time taken for forward pass: ", time() - start)
        return out  # loss if training, else action
