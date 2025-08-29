from .calvin import CALVINTrainTester
from .rlbench import RLBenchTrainTester
from .dexonomy import DexonomyTrainTester
from .dexterousact import DexterousActTrainTester


def fetch_train_tester(dataset_name):
    dataset_name = dataset_name.lower()
    if 'mobaloha' in dataset_name:
        return RLBenchTrainTester
    if 'peract2' in dataset_name:
        return RLBenchTrainTester
    if 'rlbench' in dataset_name or 'peract' in dataset_name:
        return RLBenchTrainTester
    if 'calvin' in dataset_name:
        return CALVINTrainTester
    if 'dexonomy' in dataset_name or "graspxl" in dataset_name:
        return DexonomyTrainTester
    if 'dexterousact' in dataset_name:
        return DexterousActTrainTester
    return None
