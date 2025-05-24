import torch

from .base import BaseTrainTester


class RLBenchTrainTester(BaseTrainTester):

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        proprio = self.preprocessor.process_actions(sample["proprioception"])
        rgbs, pcds = self.preprocessor.process_obs(
            sample['rgb'], sample["rgb2d"],
            sample['depth'], sample['wrist_depth'], sample['extrinsics_wrist'],
            augment=augment
        )
        return (
            sample["action"],
            torch.zeros(sample["action"].shape[:-1], dtype=bool, device='cuda'),
            rgbs,
            None,
            pcds,
            sample["instr"],
            proprio
        )
