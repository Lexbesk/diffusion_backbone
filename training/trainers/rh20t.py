from kornia import augmentation as K

from .base import BaseTrainTester


class RH20TTrainTester(BaseTrainTester):

    def __init__(self, args, dataset_cls, model_cls, depth2cloud, im_size=256):
        """Initialize."""
        super().__init__(
            args=args,
            dataset_cls=dataset_cls,
            model_cls=model_cls,
            depth2cloud=depth2cloud
        )

        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(
                degrees=0,
                scale=(0.75, 1.25),
                padding_mode="reflection",
                p=1.0
            ),
            K.RandomResizedCrop(
                size=(im_size, im_size),
                scale=(0.7, 1.0)
            )
        ).cuda()

    def _run_depth2cloud(self, sample):
        return self.depth2cloud(
            sample['depth'].cuda(non_blocking=True).float(),
            sample['extrinsics'].cuda(non_blocking=True).float(),
            sample['intrinsics'].cuda(non_blocking=True).float()
        )