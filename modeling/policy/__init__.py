from .denoise_actor_2d import DenoiseActor as DenoiseActor2D
from .denoise_actor_3d import DenoiseActor as DenoiseActor3D
from .denoise_actor_3d_sa import DenoiseActor as DenoiseActor3DSA
from .denoise_actor_3d_df import DenoiseActor as DenoiseActor3DDF
from .denoise_actor_3d_ca import DenoiseActor as DenoiseActor3DCA


def fetch_model_class(model_type):
    if model_type == 'denoise3d':
        return DenoiseActor3D
    if model_type == 'denoise2d':
        return DenoiseActor2D
    if model_type == 'denoise3dsa':
        return DenoiseActor3DSA
    if model_type == 'denoise3ddf':
        return DenoiseActor3DDF
    if model_type == 'denoise3dca':
        return DenoiseActor3DCA
    return None
