from .rlbench import (
    Peract2SingleCamDataset,
    Peract2Dataset3cam
)


def fetch_dataset_class(dataset_name):
    """Fetch the dataset class based on the dataset name."""
    dataset_classes = {
        "Peract2": Peract2SingleCamDataset,
        "Peract2TC": Peract2Dataset3cam
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_classes[dataset_name]
