from .calvin import CALVINDataset
from .rlbench import (
    Peract2Dataset,
    Peract2SingleCamDataset,
    Peract2Dataset3cam2Dwrist,
    PeractDataset,
    PeractTwoCamDataset,
    SinglePeract2Dataset,
    Peract2AllDataset,
    DatPeractDataset,
    DatPeractTwoCamDataset
)
# from .base_dex import DexDataset
from .dexonomy import DexZarrDataset as DexDataset
from omegaconf import DictConfig, ListConfig
from copy import deepcopy
import torch
from torch.utils.data import DataLoader


def fetch_dataset_class(dataset_name):
    """Fetch the dataset class based on the dataset name."""
    dataset_classes = {
        "Peract2_3dfront_3dwrist": Peract2Dataset,
        "Peract2_3dfront": Peract2SingleCamDataset,
        "Peract2_3dfront_2dwrist": Peract2Dataset3cam2Dwrist,
        "Peract": PeractDataset,
        "PeractTwoCam": PeractTwoCamDataset,
        "PeractDat": DatPeractDataset,
        "PeractDatTwoCam": DatPeractTwoCamDataset,
        'Calvin': CALVINDataset,
        'Calvin10th': CALVINDataset,
        "Peract2TCSingle": SinglePeract2Dataset,
        "Peract2All": Peract2AllDataset,
        "Dexonomy": DexDataset,
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_classes[dataset_name]

def create_dataset(config, mode):
    if isinstance(config.data.object_path, ListConfig):
        dataset_lst = []
        for p in config.data.object_path:
            new_data_config = deepcopy(config.data)
            new_data_config.object_path = p
            dataset_lst.append(DexDataset(new_data_config, mode))
        dataset = torch.utils.data.ConcatDataset(dataset_lst)
    else:
        dataset = DexDataset(config.data, mode)
    return dataset


def create_train_dataloader(config: DictConfig):
    train_dataset = create_dataset(config, mode="train")
    val_dataset = create_dataset(config, mode="eval")

    train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.data.batch_size,
                    drop_last=True,
                    num_workers=config.data.num_workers,
                    shuffle=False)
    val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.data.batch_size,
                    drop_last=True,
                    num_workers=config.data.num_workers,
                    shuffle=False)
    return train_loader, val_loader


def create_test_dataloader(config: DictConfig, mode="test"):
    test_dataset = create_dataset(config, mode=mode)
    test_loader = DataLoader(
                                test_dataset,
                                batch_size=config.data.batch_size,
                                drop_last=False,
                                num_workers=config.data.num_workers,
                                shuffle=False
                            )
    return test_loader