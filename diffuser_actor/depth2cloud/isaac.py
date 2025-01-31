import torch


class IsaacDepth2Cloud:

    def __init__(self):
        pass

    @staticmethod
    def backproject_gym_depth(depth_map, proj_matrix):
        fx = 2.0 / proj_matrix[..., 0, 0]  # (B, nc)
        fy = 2.0 / proj_matrix[..., 1, 1]  # (B, nc)
        x, y = torch.meshgrid(
            torch.arange(depth_map.shape[-1]),
            torch.arange(depth_map.shape[-2])
        )
        input_x = x.T.to(depth_map.device).half()
        input_y = y.T.to(depth_map.device).half()
        z = depth_map
        
        input_x -= depth_map.shape[-1] // 2
        input_y -= depth_map.shape[-2] // 2
        input_x /= depth_map.shape[-1]
        input_y /= depth_map.shape[-2]

        output_x = z * fx[..., None, None] * input_x[None, None]
        output_y = z * fy[..., None, None] * input_y[None, None]

        return torch.stack((output_x, output_y, z), -1)

    def __call__(self, gym_depth, proj_matrix, extrinsics, xyz_image=None):
        """
        gym_depth: (B, 2, H, W),
        proj_matrix: (B, 2, 4, 4),
        extrinsics: (B, 2, 4, 4),
        xyz_image: (B, 2, H, W, 3)
        """
        # From gym_depth to camera frame
        pcd = self.backproject_gym_depth(gym_depth, proj_matrix)  # B nc H W 3
        if xyz_image is not None:
            assert torch.allclose(pcd, xyz_image)
        # From camera frame to world frame
        _, _, h, w, _ = pcd.shape
        pcd = pcd.reshape(len(pcd), 2, -1, 3)  # (B, nc, H*W, 3)
        pcd = torch.cat((pcd, torch.ones_like(pcd)[..., :1]), -1)
        rz = torch.tensor([  # rotation matrix 90 degrees around z-axis
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).to(pcd.device)[None, None].half()
        ry = torch.tensor([  # rotation matrix 90 degrees around y-axis
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ]).to(pcd.device)[None, None].half()
        result = torch.matmul(
            extrinsics.half() @ ry @ rz,
            pcd.permute(0, 1, 3, 2).half()
        )
        result = result.permute(0, 1, 3, 2)[..., :3]
        result = result.reshape(len(result), 2, h, w, 3)
        return result.permute(0, 1, 4, 2, 3)
