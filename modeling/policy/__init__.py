from .denoise_actor_2d import DenoiseActor as DenoiseActor2D
from .denoise_actor_3d import DenoiseActor as DenoiseActor3D
# from .denoise_actor_3d_sa import DenoiseActor as DenoiseActor3DSA
from .denoise_actor_3d_df import DenoiseActor as DenoiseActor3DDF
# from .denoise_actor_3d_ca import DenoiseActor as DenoiseActor3DCA
from .denoise_actor_3d_2dwrist import DenoiseActor as DenoiseActor2Dwrist
from .denoise_actor_3d_le import DenoiseActor as DenoiseActor3DLE
# from .denoise_actor_3d_2ndstage import DenoiseActor as DenoiseActor3D2ndStage
from .flower.model import FLOWERVLA
from .flower.model_peract2 import FLOWERVLA as FLOWERVLARLB
from .flower.model_peract2_3d import FLOWERVLA3D
from .flower_actor import DenoiseActor as FlowerActor
from .base_grasp_denoiser import GraspDenoiser
from .grasp_trajectory_denoiser import DexterousActor


def fetch_model_class(model_type):
    if model_type == 'denoise3d':  # standard 3DFA
        return DenoiseActor3D
    if model_type == 'denoise2d':  # standard 2DFA
        return DenoiseActor2D
    if model_type == 'denoise3dle':  # current CALVIN version
        return DenoiseActor3DLE
    if model_type == 'denoise3d_2dwrist':  # 3D front + 2D wrist
        return DenoiseActor2Dwrist
    if model_type == 'flower':
        return FLOWERVLA
    if model_type == 'flower_rlbench':
        return FLOWERVLARLB
    if model_type == 'flower_rlbench_3d':
        return FLOWERVLA3D
    # if model_type == 'denoise3dsa':
    #     return DenoiseActor3DSA
    if model_type == 'denoise3ddf':
        return DenoiseActor3DDF
    # if model_type == 'denoise3dca':
    #     return DenoiseActor3DCA
    # if model_type == 'denoise3d_2ndstage':
    #     return DenoiseActor3D2ndStage
    if model_type == 'flower_actor':
        return FlowerActor
    if model_type == 'grasp_denoiser':
        return GraspDenoiser
    if model_type == 'dexterousactor':
        return DexterousActor
    return None
