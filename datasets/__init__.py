from .calvin import CALVINDataset
from .rlbench import (
    Peract2SingleCamDataset,
    Peract2Dataset3cam,
    Mobaloha3cam
    Peract2Dataset3cam2Dwrist,
    PeractDataset,
    PeractTwoCamDataset

)
from .rh20t import RH20TDataset


def fetch_dataset_class(dataset_name):
    """Fetch the dataset class based on the dataset name."""
    dataset_classes = {
        "Peract2": Peract2SingleCamDataset,
        "Peract2TC": Peract2Dataset3cam,
        "Peract2_3dfront_2dwrist": Peract2Dataset3cam2Dwrist,
        "Peract": PeractDataset,
        "PeractTwoCam": PeractTwoCamDataset,
        'Calvin': CALVINDataset,
        'RH20T': RH20TDataset
        "Mobaloha": Mobaloha3cam,
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_classes[dataset_name]
