import numpy as np
import torch
from utils.visualize_keypose_frames import visualize_actions_and_point_clouds
from datasets.utils import read_zarr_with_cache
from modeling.depth2cloud.rlbench import GNFactorDepth2Cloud
from matplotlib import pyplot as plt


fname = '/lustre/fsw/portfolios/nvr/users/ngkanatsios/Peract2_zarr/train.zarr'
annos = read_zarr_with_cache(fname)
rgb = annos['rgb'][:8]  # (8, 5, 3, 256, 256)
rgb = rgb.transpose(0, 1, 3, 4, 2)  # (8, 5, 256, 256, 3)
rgb = rgb.reshape(len(rgb), 5*256, 256, 3)
rgb = np.concatenate(rgb, 1)
print(rgb.shape)
plt.imshow(rgb)
plt.savefig('2.jpg')

d2c = GNFactorDepth2Cloud((256, 256))
depth = annos['depth'][:4, -1:]
pcd = d2c(torch.from_numpy(depth).cuda()).cpu()  # 4 1 3 256 256
pcd = pcd.view(len(pcd), 3, 256, 256).permute(0, 2, 3, 1)  # 4 256 256 3
pcd = np.concatenate(pcd.float().numpy(), 1)
plt.imshow(pcd)
plt.savefig('2.jpg')