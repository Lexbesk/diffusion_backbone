import torch
from torch.nn import functional as F

from .base import BaseTrainTester


class CALVINTrainTester(BaseTrainTester):

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
        # Handle wrist camera
        pcd_w = pcd_w.float()
        h, w = pcds.shape[-2:]
        pcd_w = F.interpolate(pcd_w[:, 0], (h, w), mode='bilinear')[:, None]
        pcds = torch.cat((pcds, pcd_w), 1)
        rgb_w = sample["rgb2d"].cuda(non_blocking=True).float() / 255
        rgb_w = F.interpolate(rgb_w[:, 0], (h, w), mode='bilinear')[:, None]
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
