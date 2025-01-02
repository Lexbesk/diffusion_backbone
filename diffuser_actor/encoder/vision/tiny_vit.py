import timm
import torch
from torch import nn


class ViTTransform:

    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class TinyViT(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tiny_vit_5m_224.dist_in22k_ft_in1k',
            pretrained=True,
            features_only=True
        )
        self.names = ['res2', 'res3', 'res4', 'res5']

    def forward(self, x):
        x = self.backbone(x)
        return {name: val for name, val in zip(self.names, x)}


def load_tiny():
    return TinyViT(), ViTTransform()
