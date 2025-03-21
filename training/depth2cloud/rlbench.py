import numpy as np
import torch


class RLBenchDepth2Cloud:

    def __init__(self, shape):
        self.uniforms = torch.from_numpy(
            self._create_uniform_pixel_coords_image(shape)
        ).permute(2, 0, 1)  # (3, H, W)

    @staticmethod
    def _create_uniform_pixel_coords_image(resolution):
        pixel_x_coords = np.reshape(
            np.tile(np.arange(resolution[1]), [resolution[0]]),
            (resolution[0], resolution[1], 1)).astype(np.uint8)
        pixel_y_coords = np.reshape(
            np.tile(np.arange(resolution[0]), [resolution[1]]),
            (resolution[1], resolution[0], 1)).astype(np.uint8)
        pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
        uniform_pixel_coords = np.concatenate(
            (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
        return uniform_pixel_coords

    @staticmethod
    def _get_cam_proj_mat_inv_b(extrinsics, intrinsics):
        # Construct camera projection
        C = extrinsics[:, :3, [3]]  # (B, 3, 1)
        R = extrinsics[:, :3, :3]  # (B, 3, 3)
        R_inv = R.permute(0, 2, 1)  # inverse of rot matrix is transpose
        R_inv_C = torch.matmul(R_inv, C)  # (B, 3, 1)
        extrinsics = torch.cat((R_inv, -R_inv_C), -1)  # (B, 3, 4)
        cam_proj_mat = torch.matmul(intrinsics, extrinsics)  # (B, 3, 4)
        cam_proj_mat_homo = torch.cat([
            cam_proj_mat,
            torch.tensor(
                [0, 0, 0, 1],
                dtype=cam_proj_mat.dtype,
                device=cam_proj_mat.device
            )[None, None].repeat(len(cam_proj_mat), 1, 1)
        ], 1)  # (B, 4, 4)
        cam_proj_mat_inv = torch.linalg.inv(cam_proj_mat_homo)[:, :3]
        return cam_proj_mat_inv  # (B, 3, 4)

    def unproject(self, depth, extrinsics, intrinsics):
        # depth is (B, H, W), extrinsics (B, 4, 4), intrinsics (B, 3, 3)
        # output is (B, 3, H, W)
        device = depth.device
        pc = self.uniforms[None].to(device) * depth[:, None]  # (B, 3, H, W)
        b, _, h, w = pc.shape
        pc = torch.cat([pc, torch.ones_like(pc[:, :1])], 1)  # (B, 4, H, W)
        pc = pc.view(b, 4, -1)  # (4, B*H*W)
        cam_proj_mat_inv = self._get_cam_proj_mat_inv_b(extrinsics, intrinsics)
        pc = torch.matmul(cam_proj_mat_inv, pc)  # (B, 3, H*W)
        pc = pc.reshape(b, 3, h, w)  # (B, 3, H, W)
        return pc

    def __call__(self, depth, extrinsics, intrinsics):
        # depth is (B Nc H W), extrinsics (B Nc 4 4), intrinsics (B Nc 3 3)
        # output is (B Nc 3 H W)
        # Nc is the number of cameras
        b, nc, h, w = depth.shape
        pc = self.unproject(
            depth.flatten(0, 1),
            extrinsics.flatten(0, 1),
            intrinsics.flatten(0, 1)
        )
        return pc.reshape(b, nc, 3, h, w)
