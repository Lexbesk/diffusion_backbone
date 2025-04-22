from functools import partial

from .calvin import CALVINTrainTester
from .rlbench import RLBenchTrainTester
from .rh20t import RH20TTrainTester

def fetch_train_tester(dataset_name):
    dataset_name = dataset_name.lower()
    if 'mobaloha' in dataset_name:
        return partial(RLBenchTrainTester, im_size=256)
    if 'peract2' in dataset_name:
        return partial(RLBenchTrainTester, im_size=256)
    if 'rlbench' in dataset_name or 'peract' in dataset_name:
        return partial(RLBenchTrainTester, im_size=128)
    if 'calvin' in dataset_name:
        return partial(CALVINTrainTester, im_size=160)
    if 'rh20t' in dataset_name:
        return partial(RH20TTrainTester, im_size=256)
    return None
