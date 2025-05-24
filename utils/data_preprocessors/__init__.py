from functools import partial

from .calvin import CALVINDataPreprocessor
from .rlbench import RLBenchDataPreprocessor


def fetch_data_preprocessor(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=256)
    if 'rlbench' in dataset_name or 'peract' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=128)
    if 'calvin' in dataset_name:
        return partial(CALVINDataPreprocessor, orig_imsize=200)
    return None
