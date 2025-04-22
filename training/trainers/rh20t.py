from .base import BaseTrainTester


class RH20TTrainTester(BaseTrainTester):\

    def _run_depth2cloud(self, sample):
        return self.depth2cloud(
            sample['depth'].cuda(non_blocking=True).float(),
            sample['extrinsics'].cuda(non_blocking=True).float(),
            sample['intrinsics'].cuda(non_blocking=True).float()
        )
