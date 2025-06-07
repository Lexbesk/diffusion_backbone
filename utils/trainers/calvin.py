import torch
from torch import nn

from .base import BaseTrainTester


class CALVINTrainTester(BaseTrainTester):

    @torch.no_grad()
    def get_workspace_normalizer(self):
        return nn.Parameter(torch.tensor([
            [-0.02, -0.02, -0.02, -0.05, -0.05, -0.05],
            [ 0.02,  0.02,  0.02,  0.05,  0.05,  0.05]
        ]), requires_grad=False)

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        proprio = self.preprocessor.process_proprio(sample["proprioception"])
        rgbs, pcds = self.preprocessor.process_obs(
            sample["rgb"], sample["rgb2d"],
            sample["depth"], sample["wrist_depth"], sample["extrinsics_wrist"],
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
