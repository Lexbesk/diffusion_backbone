import torch
from torch import Tensor


class DexterousActDataPreprocessor:
    def pass_through(self, batch):
        # Convert everything to tensors and move to GPU
        out = {k: to_tensor(v).to('cuda') for k, v in batch.items()}
        return out