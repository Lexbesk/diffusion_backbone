import torch

from .base import BaseTrainTester


class RLBenchTrainTester(BaseTrainTester):

    def _run_depth2cloud(self, sample):
        return self.depth2cloud(
            sample['depth'].cuda(non_blocking=True).to(torch.bfloat16),
            sample['extrinsics'].cuda(non_blocking=True).to(torch.bfloat16),
            sample['intrinsics'].cuda(non_blocking=True).to(torch.bfloat16)
        )
