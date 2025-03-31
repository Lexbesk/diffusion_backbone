import numpy as np
import torch


class CALVINDepth2Cloud:

    def __init__(self):
        self.T_world_cam = torch.from_numpy(np.linalg.inv(np.asarray([
            0.7232445478439331, -0.030260592699050903, 0.6899287700653076, 0.0,
            0.5141233801841736, 0.6906105875968933, -0.5086592435836792, 0.0,
            -0.46107974648475647, 0.7225935459136963, 0.5150379538536072, 0.0,
            0.21526622772216797, -0.26317155361175537, -4.399168968200684, 1.0
        ]).reshape((4, 4)).T)).float().cuda()
        self.width, self.height, self.fov = 200, 200, 10
        self.foc = self.height / (2 * np.tan(np.deg2rad(self.fov) / 2))

    def __call__(self, depth):
        """
        Args
            depth: (B, H, W), if H,W != 200, we assume center crop
        Output
            point cloud: (B, 3, H, W)
        """
        b, h, w = depth.shape
        v, u = torch.meshgrid(torch.arange(200), torch.arange(200))
        h_offset = (200 - h) // 2
        w_offset = (200 - w) // 2
        v = v[h_offset:h+h_offset, w_offset:w+w_offset].to(depth.device)
        u = u[h_offset:h+h_offset, w_offset:w+w_offset].to(depth.device)

        # Camera XYZ
        x = (u - self.width // 2)[None] * depth / self.foc
        y = -(v - self.height // 2)[None] * depth / self.foc
        z = -depth
        ones = torch.ones_like(z)

        # Unproject to world coordinates
        cam_pos = torch.stack([x, y, z, ones], 1).reshape(b, 4, -1)  # B 4 HW
        T_world_cam = self.T_world_cam[None].repeat(b, 1, 1)  # B 4 4
        world_pos = torch.matmul(T_world_cam, cam_pos)  # B 4 HW

        # Reshape
        return world_pos[:, :3].reshape(b, 3, h, w)
