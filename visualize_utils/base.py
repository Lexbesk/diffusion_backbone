"""All datasets can use this class."""

from kornia import augmentation as K
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
import torch
import zarr
from zarr.storage import DirectoryStore
from zarr import LRUStoreCache

from utils.pytorch3d_transforms import quaternion_to_matrix
from utils.visualize_keypose_frames import (
    get_three_points_from_curr_action,
    compute_rectangle_polygons
)
from visualize_utils.meshcat_utils import (
    create_visualizer,
    visualize_pointcloud,
    visualize_triad
)


def visualize_actions_and_point_clouds(visible_pcd, visible_rgb,
                                       gripper_pose_trajs=[],
                                       dataset_name='rlbench',
                                       legends=[], markers=[],
                                       rotation_param="quat_from_query",
                                       save=True, savename='diff_traj.png'):
    """Visualize by plotting the point clouds and gripper pose.

    Args:
        visible_pcd: A tensor of shape (ncam, 3, H, W)
        visible_rgb: A tensor of shape (ncam, 3, H, W)
        gripper_pose_trajs: A list of tensors of shape (8,)
    """
    gripper_pose_trajs = [t.data.cpu() for t in gripper_pose_trajs]
    cur_vis_pcd = visible_pcd.permute(0, 2, 3, 1).flatten(0, -2).data.cpu().numpy()
    cur_vis_rgb = visible_rgb.permute(0, 2, 3, 1).flatten(0, -2).data.cpu().numpy()

    # Subsample points and restrict view
    rand_inds = torch.randperm(cur_vis_pcd.shape[0]).data.cpu().numpy()[:50000]
    if dataset_name == 'rlbench':
        mask = (
            (cur_vis_pcd[rand_inds, 2] >= 0.5)
            & (cur_vis_pcd[rand_inds, 0] >= -0.2)
        )
        rand_inds = rand_inds[mask]

    # Figure
    fig = plt.figure()
    canvas = fig.canvas
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # Scatter point cloud with color
    ax.scatter(cur_vis_pcd[rand_inds, 0],
               cur_vis_pcd[rand_inds, 1],
               cur_vis_pcd[rand_inds, 2],
               c=cur_vis_rgb[rand_inds], s=1,
               zorder=1)

    # Overlay grippers
    cont_range_inds = np.linspace(0, 1, len(gripper_pose_trajs)).astype(float)
    cm = plt.get_cmap('brg')
    colors = cm(cont_range_inds)
    legends = (
        legends if len(legends) == len(gripper_pose_trajs)
        else [""] * len(gripper_pose_trajs)
    )
    markers = (
        markers if len(markers) == len(gripper_pose_trajs)
        else ["*"] * len(gripper_pose_trajs)
    )
    for gripper_pose, color, legend, marker in (
        zip(gripper_pose_trajs, colors, legends, markers)
    ):
        gripper_pcd = get_three_points_from_curr_action(
            gripper_pose, rotation_param=rotation_param, for_vis=True
        )
        ax.plot(gripper_pcd[0, [1, 0, 2], 0],
                gripper_pcd[0, [1, 0, 2], 1],
                gripper_pcd[0, [1, 0, 2], 2],
                c=color,
                markersize=1, marker=marker,
                linestyle='--', linewidth=1,
                label=legend,
                zorder=2)
        polygons = compute_rectangle_polygons(gripper_pcd[0])
        for poly_ind, polygon in enumerate(polygons):
            polygon = Poly3DCollection(polygon, facecolors=color)
            alpha = 0.5 if poly_ind == 0 else 1.3
            polygon.set_edgecolor([min(c * alpha, 1.0) for c in color])
            polygon.set_zorder(3)
            ax.add_collection3d(polygon)

    # SHow point cloud from different angles
    fig.tight_layout()
    ax.legend(loc="lower center", ncol=len(gripper_pose_trajs))
    images = []
    for elev, azim in zip([10, 90], [0, 0]):
        ax.view_init(elev=elev, azim=azim, roll=0)
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 4)[..., 1:]
        image = image[60:, 110:-110]
        images.append(image)
    images = np.concatenate(images, 1)
    if save:
        Image.fromarray(images, mode='RGB').save(savename)
    plt.close()


def read_zarr_with_cache(fname, mem_gb=16):
    # Configure the underlying store
    store = DirectoryStore(fname)

    # Wrap the store with a cache
    cached_store = LRUStoreCache(store, max_size=mem_gb * 2**30)  # GB cache

    # Open Zarr file with caching
    return zarr.open_group(cached_store, mode="r")


class Visualizer:

    def __init__(self, zarr_path, depth2cloud, use_meshcat=False, im_size=256):
        self.annos = read_zarr_with_cache(zarr_path, 0.1)
        self.d2c = depth2cloud
        if use_meshcat:
            self.mesh_vis = create_visualizer(clear=False)
        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(
                degrees=0,
                translate=0.01,
                scale=(0.75, 1.25),
                padding_mode="reflection",
                p=1.0
            ),
            K.RandomRotation((-5, 5), p=0.3),
            K.RandomResizedCrop(
                size=(im_size, im_size),
                scale=(0.95, 1.05),
                p=0.1
            )
        ).cuda()

    def plot_images_depths(self, t, cams=None, img='rgb', depth='depth',
                           save_path=''):
        if cams is None:
            imgs = self.annos[img][t]  # Nc 3 h w
            depths = self.annos[depth][t]  # Nc h w
        else:
            imgs = self.annos[img][t][cams]  # Nc 3 h w
            depths = self.annos[depth][t][cams]  # Nc h w
        # Concatenate imgs across width
        imgs = np.concatenate(imgs, -1)  # 3 h Nc*w
        imgs = imgs.astype(float) / 255.0
        # Concatenate depths across width and repeat
        depths = np.concatenate(depths, -1)
        depths = np.stack([depths] * 3)  # 3 h Nc*w
        # Top images, bottom depths
        print(imgs.shape, depths.shape)
        _cat = np.concatenate([imgs, depths], 1).transpose(1, 2, 0)
        # Plot
        plt.imshow(_cat)
        plt.savefig(save_path + f'img_depth_{t}.jpg')
        plt.close()

    def plot_images(self, t, cams=None, img='rgb', save_path=''):
        if cams is None:
            imgs = self.annos[img][t]  # Nc 3 h w
        else:
            imgs = self.annos[img][t][cams]  # Nc 3 h w
        # Concatenate imgs across width
        imgs = np.concatenate(imgs, -1)  # 3 h Nc*w
        imgs = imgs.astype(float) / 255.0
        imgs = imgs.transpose(1, 2, 0)
        # Plot
        plt.imshow(imgs)
        plt.savefig(save_path + f'img_{t}.jpg')
        plt.close()

    def plot_aug_images(self, t, cams=None, img='rgb', save_path=''):
        if cams is None:
            imgs = self.annos[img][t]  # Nc 3 h w
        else:
            imgs = self.annos[img][t][cams]  # Nc 3 h w
        # Augment
        aug = self.aug(torch.from_numpy(imgs).cuda().float() / 255).cpu().numpy()
        # Concatenate imgs across width
        imgs = np.concatenate(imgs, -1)  # 3 h Nc*w
        imgs = imgs.astype(float) / 255.0
        imgs = imgs.transpose(1, 2, 0)
        # Concatenate aug imgs across width
        aug = np.concatenate(aug, -1)  # 3 h Nc*w
        aug = aug.transpose(1, 2, 0)
        # Plot
        imgs = np.concatenate((imgs, aug))
        plt.imshow(imgs)
        plt.savefig(save_path + f'img_aug_{t}.jpg')
        plt.close()

    def plot_point_cloud_grippers(self, t, cams=None, img='rgb', depth='depth',
                                  curr=True, action=True, save_path='',
                                  dataset_name='rlbench'):
        if cams is None:
            imgs = self.annos[img][t]  # Nc 3 h w
            depths = self.annos[depth][t]  # Nc h w
            extrinsics = self.annos['extrinsics'][t]
            intrinsics = self.annos['intrinsics'][t]
        else:
            imgs = self.annos[img][t][cams]  # Nc 3 h w
            depths = self.annos[depth][t][cams]  # Nc h w
            extrinsics = self.annos['extrinsics'][t][cams]
            intrinsics = self.annos['intrinsics'][t][cams]
        # Get point cloud
        pc = self.d2c(
            torch.from_numpy(depths)[None].float(),
            torch.from_numpy(extrinsics)[None].float(),
            torch.from_numpy(intrinsics)[None].float()
        )[0]  # Nc 3 h w
        # Grippers
        grippers = []
        if curr:
            prop = torch.from_numpy(self.annos['proprioception'][t][-1])  # (2, 8) or (8,)
            if len(prop.shape) > 1:
                grippers.extend([p for p in prop])
            else:
                grippers.append(prop)
        if action:
            traj = torch.from_numpy(self.annos['action'][t])  # (N, 2, 8) or (N, 8)
            if len(traj.shape) > 2:
                for i in range(traj.shape[1]):
                    grippers.extend([t_ for t_ in traj[:, i]])
            else:
                grippers.extend([t_ for t_ in traj])
        # Visualize
        visualize_actions_and_point_clouds(
            pc,
            torch.from_numpy(imgs),
            grippers,
            dataset_name=dataset_name,
            savename=save_path + f'pcd_{t}.jpg'
        )

    def meshcat_pcd(self, t, cams=None, img='rgb', depth='depth'):
        imgs = self.annos[img][t]  # Nc 3 h w
        depths = self.annos[depth][t]  # Nc h w
        # Get point cloud
        pc = self.d2c(
            torch.from_numpy(depths)[None].float(),
            torch.from_numpy(self.annos['extrinsics'][t])[None].float(),
            torch.from_numpy(self.annos['intrinsics'][t])[None].float()
        )[0].numpy()  # Nc 3 h w
        # Visualize point cloud
        ncam, _, H, W = pc.shape
        pc2 = pc.reshape(ncam, 3, H // 8, 8, W // 8, 8).mean(-1).mean(-2)
        imgs = imgs.reshape(ncam, 3, H // 8, 8, W // 8, 8).mean(-1).mean(-2)
        visualize_pointcloud(
            self.mesh_vis,
            f"pc", 
            pc2.transpose(0, 2, 3, 1).reshape(-1, 3),
            imgs.transpose(0, 2, 3, 1).reshape(-1, 3),
            size=0.01
        )
        # Visualize trajectories
        traj = torch.from_numpy(self.annos['action'][t])  # (N, 2, 8) or (N, 1, 8)
        if len(traj.shape) > 2:
            if  traj.size(1) == 2:
                traj = torch.cat((traj[:, 0], traj[:, 1]))
            else:
                traj = traj[:, 0]
        mat_ = torch.stack([torch.eye(4)] * len(traj))
        # import ipdb; ipdb.set_trace()
        mat_[:, :3, :3] = quaternion_to_matrix(traj[:, 3:7])
        mat_[:, :3, 3] = traj[:, :3]
        for t_ in range(len(traj)):
            visualize_triad(
                self.mesh_vis,
                f"end_effector{t_}",
                T=mat_[t_].numpy(),
                radius=0.01
            )

    def batch_meshcat(self, cams=None, img='rgb', depth='depth'):
        for t in range(5):
            self.mesh_vis.delete()
            self.meshcat_pcd(t, cams, img, depth)
            if input("Enter a key to continue [q to exit]") == "q":
                break

    def show_language(self, t):
        return None


def verify_data_pipeline():
    vis = Visualizer()  # give actual argument!
    for t in range(10):
        vis.plot_images_depths(t)
        vis.plot_aug_images(t)
        vis.plot_point_cloud_grippers(t)
        vis.show_language(t)
    vis.plot_images_depths()
