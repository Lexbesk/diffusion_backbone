from .calvin import CALVINTrainTester
from .rlbench import RLBenchTrainTester


def fetch_train_tester(dataset_name):
    dataset_name = dataset_name.lower()
    if 'rlbench' in dataset_name or 'peract' in dataset_name:
        return RLBenchTrainTester
    if 'calvin' in dataset_name:
        return CALVINTrainTester
    return None
