from .calvin import CALVINDepth2Cloud
from .rlbench import RLBenchDepth2Cloud
from .rh20t import RH20TDepth2Cloud

def fetch_depth2cloud(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return RLBenchDepth2Cloud((256, 256))
    if 'rlbench' in dataset_name or 'peract' in dataset_name:
        return RLBenchDepth2Cloud((128, 128))
    if 'calvin' in dataset_name:
        return CALVINDepth2Cloud()
    if 'rh20t' in dataset_name:
        return RH20TDepth2Cloud((256, 256))
    return None
