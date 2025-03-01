import numpy as np
import torch


left_shoulder_camera_extrinsics = np.array([
    [ 1.73648179e-01,  8.92538846e-01,  4.16198105e-01, -1.74999714e-01],
    [ 9.84807789e-01, -1.57378674e-01, -7.33871460e-02, 2.00000003e-01],
    [-1.78813934e-07,  4.22618657e-01, -9.06307697e-01, 1.97999895e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])
left_shoulder_camera_intrinsics = np.array([
    [-351.6771208,    0.       ,  128.       ],
    [   0.       , -351.6771208,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])
right_shoulder_camera_extrinsics = np.array([
    [-1.73648357e-01,  8.92538846e-01,  4.16198105e-01, -1.74997091e-01],
    [ 9.84807789e-01,  1.57378793e-01,  7.33869076e-02, -2.00000003e-01],
    [-1.19209290e-07,  4.22618628e-01, -9.06307697e-01, 1.97999227e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])
right_shoulder_camera_intrinsics = np.array([
    [-351.6771208,    0.       ,  128.       ],
    [   0.       , -351.6771208,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])
wrist_camera_extrinsics = np.array([
    [ 8.34465027e-07,  9.87690389e-01,  1.56421274e-01, 3.04353595e-01],
    [ 9.99999940e-01, -7.15255737e-07,  1.86264515e-07, -6.17044233e-03],
    [ 3.05473804e-07,  1.56421274e-01, -9.87690210e-01, 1.57466102e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])
wrist_camera_intrinsics = np.array([
    [-221.70249591,    0.        ,  128.        ],
    [   0.        , -221.70249591,  128.        ],
    [   0.        ,    0.        ,    1.        ]
])
front_camera_extrinsics = np.array([
    [ 1.19209290e-07, -4.22617942e-01, -9.06307936e-01, 1.34999919e+00],
    [-1.00000000e+00, -5.96046448e-07,  1.49011612e-07, 3.71546562e-08],
    [-5.66244125e-07,  9.06307936e-01, -4.22617912e-01, 1.57999933e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])
front_camera_intrinsics = np.array([
    [-351.6771208,    0.       ,  128.       ],
    [   0.       , -351.6771208,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])
cameras = {
    'left': {
        'extrinsics': left_shoulder_camera_extrinsics,
        'intrinsics': left_shoulder_camera_intrinsics
    },
    'right': {
        'extrinsics': right_shoulder_camera_extrinsics,
        'intrinsics': right_shoulder_camera_intrinsics
    },
    'wrist': {
        'extrinsics': wrist_camera_extrinsics,
        'intrinsics': wrist_camera_intrinsics
    },
    'front': {
        'extrinsics': front_camera_extrinsics,
        'intrinsics': front_camera_intrinsics
    }
}


class Depth2Cloud:

    def __init__(self, shape, extrinsics, intrinsics):
        self.uniforms = torch.from_numpy(
            self._create_uniform_pixel_coords_image(shape)
        ).cuda().half().permute(2, 0, 1)  # (3, H, W)
        self.cam_proj_mat_inv = torch.from_numpy(
            self._get_cam_proj_mat_inv(extrinsics, intrinsics)
        ).cuda().half()  # (3, 4)

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
    def _get_cam_proj_mat_inv(extrinsics, intrinsics):
        # Construct camera projection
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        cam_proj_mat = np.matmul(intrinsics, extrinsics)
        cam_proj_mat_homo = np.concatenate(
            [cam_proj_mat, [np.array([0, 0, 0, 1])]])
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[:3]
        return cam_proj_mat_inv

    def __call__(self, depth):
        # depth is (B, H, W)
        # output is (B, 3, H, W)
        pc = self.uniforms[None] * depth[:, None]  # (B, 3, H, W)
        b, _, h, w = pc.shape
        pc = torch.cat([pc, torch.ones_like(pc[:, :1])], 1)  # (B, 4, H, W)
        pc = pc.transpose(0, 1).view(4, -1)  # (4, B*H*W)
        pc = torch.matmul(self.cam_proj_mat_inv, pc)  # (3, B*H*W)
        pc = pc.reshape(3, b, h, w).transpose(0, 1)  # (B, 3, H, W)
        return pc


class PeractDepth2Cloud:

    def __init__(self, shape):
        self.d2cs = [
            Depth2Cloud(
                shape,
                cameras[cam]['extrinsics'],
                cameras[cam]['intrinsics']
            )
            for cam in ['left', 'right', 'wrist', 'front']
        ]

    def __call__(self, depth):
        # depth is (B, ncam, H, W)
        assert depth.shape[1] == len(self.d2cs)
        return torch.stack([
            self.d2cs[i](depth[:, i]) for i in range(depth.shape[1])
        ], 1)  # (B, ncam, 3, H, W)


class PeractTwoCam2Cloud:

    def __init__(self, shape):
        self.d2cs = [
            Depth2Cloud(
                shape,
                cameras[cam]['extrinsics'],
                cameras[cam]['intrinsics']
            )
            for cam in ['wrist', 'front']
        ]

    def __call__(self, depth):
        # depth is (B, ncam, H, W)
        assert depth.shape[1] == len(self.d2cs)
        return torch.stack([
            self.d2cs[i](depth[:, i]) for i in range(depth.shape[1])
        ], 1)  # (B, ncam, 3, H, W)


class GNFactorDepth2Cloud:

    def __init__(self, shape):
        self.d2cs = [
            Depth2Cloud(
                shape,
                cameras[cam]['extrinsics'],
                cameras[cam]['intrinsics']
            )
            for cam in ['front']
        ]

    def __call__(self, depth):
        # depth is (B, ncam, H, W)
        assert depth.shape[1] == len(self.d2cs)
        return torch.stack([
            self.d2cs[i](depth[:, i]) for i in range(depth.shape[1])
        ], 1)  # (B, ncam, 3, H, W)


class Peract2Depth2Cloud:

    def __init__(self, shape):
        self.d2cs = [
            Depth2Cloud(
                shape,
                cameras[cam]['extrinsics'],
                cameras[cam]['intrinsics']
            )
            for cam in ['wrist']
        ]

    def __call__(self, depth):
        # depth is (B, ncam, H, W)
        assert depth.shape[1] == len(self.d2cs)
        return torch.stack([
            self.d2cs[i](depth[:, i]) for i in range(depth.shape[1])
        ], 1)  # (B, ncam, 3, H, W)
