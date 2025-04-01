import torch


class RH20TDepth2Cloud:

    def __init__(self):
        pass

    def unproject(self, depth, extrinsics, intrinsics):
        # depth is (B, H, W), extrinsics (B, 4, 4), intrinsics (B, 3, 3)
        # output is (B, 3, H, W)
        device = depth.device
        B, H, W = depth.shape
        depth = depth / 1000.

        # Create meshgrid of pixel coordinates
        u = torch.linspace(0, W - 1, W, device=device).repeat(H, 1)  # (H, W)
        v = torch.linspace(0, H - 1, H, device=device).view(H, 1).repeat(1, W)  # (H, W)

        # Stack and add batch dimension
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=0).unsqueeze(0)  # Shape (1, 3, H, W)
        uv1 = uv1.repeat(B, 1, 1, 1)  # (B, 3, H, W)

        # Invert intrinsics to get normalized camera rays
        K_inv = torch.inverse(intrinsics)  # (B, 3, 3)

        # Compute camera coordinates
        cam_coords = torch.einsum('bij, bjhw -> bihw', K_inv.float(), uv1.float())  # (B, 3, H, W)
        xyz_camera = cam_coords * depth.unsqueeze(1)  # Scale by depth, Shape (B, 3, H, W)

        # Convert to homogeneous coordinates (add a row of ones)
        ones = torch.ones((B, 1, H, W), device=depth.device)  # (B, 1, H, W)
        xyz_homogeneous = torch.cat([xyz_camera, ones], dim=1)  # (B, 4, H, W)
        inv_extrinsics = torch.inverse(extrinsics)

        # Apply extrinsics (transform from camera to world)
        return torch.einsum(
            'bij, bjhw -> bihw',
            inv_extrinsics[:, :3, :],
            xyz_homogeneous
        )  # (B, 3, H, W)

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
