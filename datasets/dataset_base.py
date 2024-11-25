from pathlib import Path

import torch
from torch.utils.data import Dataset

from .utils import Resize, ColorAugmentation


class BaseDataset(Dataset):
    """Base dataset."""

    def __init__(
        self,
        root,
        training=True,
        image_rescale=(1.0, 1.0),
        relative_action=False,
        color_aug=False,
    ):
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action
        self._training = training

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)
        else:
            self._resize = None

        if self._training and color_aug:
            self._color_aug = ColorAugmentation()
        else:
            self._color_aug = None