import os
from pathlib import Path
import contextlib
from collections import Counter

import numpy as np
from numpy import pi
import torch
import utils.pytorch3d_transforms as pytorch3d_transforms
import pybullet
import hydra
from omegaconf import OmegaConf


############################################################
# Functions to prepare inputs/outputs of model
############################################################
def convert_rotation(rot):
    """Convert Euler angles to Quarternion
    """
    rot = torch.as_tensor(rot)
    mat = pytorch3d_transforms.euler_angles_to_matrix(rot, "XYZ")
    quat = pytorch3d_transforms.matrix_to_quaternion(mat)
    return quat.numpy()


def deproject(cam, depth_img, homogeneous=False, sanity_check=False):
    """
    Deprojects a pixel point to 3D coordinates
    Args
        point: tuple (u, v); pixel coordinates of point to deproject
        depth_img: np.array; depth image used as reference to generate 3D coordinates
        homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                     else returns the world coordinates (x, y, z) position
    Output
        (x, y, z): (3, npts) np.array; world coordinates of the deprojected point
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    if hasattr(cam, "viewMatrix"):
        # For camera in pybullet
        T_world_cam = np.linalg.inv(np.array(cam.viewMatrix).reshape((4, 4)).T)
    else:
        # Some CALVIN decided to use another convention
        T_world_cam = np.linalg.inv(np.array(cam.view_matrix).reshape((4, 4)).T)
    z = depth_img[v, u]
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    x = (u - cam.width // 2) * z / foc
    y = -(v - cam.height // 2) * z / foc
    z = -z
    ones = np.ones_like(z)

    cam_pos = np.stack([x, y, z, ones], axis=0)
    world_pos = T_world_cam @ cam_pos

    # Sanity check by using camera.deproject function.  Check 2000 points.
    if sanity_check:
        sample_inds = np.random.permutation(u.shape[0])[:2000]
        for ind in sample_inds:
            cam_world_pos = cam.deproject((u[ind], v[ind]), depth_img, homogeneous=True)
            assert np.abs(cam_world_pos-world_pos[:, ind]).max() <= 1e-3

    if not homogeneous:
        world_pos = world_pos[:3]

    return world_pos


def prepare_visual_states(obs, env):

    """Prepare point cloud given RGB-D observations.  In-place add point clouds
    to the observation dictionary.

    Args:
        obs: a dictionary of observations
            - rgb_obs: a dictionary of RGB images
            - depth_obs: a dictionary of depth images
            - robot_obs: a dictionary of proprioceptive states
        env: a PlayTableSimEnv instance which contains camera information
    """
    obs["pcd_obs"] = {}
    # Compute point cloud for front camera
    depth_static = obs["depth_obs"]["depth_static"]
    static_pcd = deproject(
        env.cameras[0], depth_static,
        homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    static_pcd = np.reshape(
        static_pcd, (depth_static.shape[0], depth_static.shape[1], 3)
    )
    obs["pcd_obs"]["pcd_static"] = static_pcd
    # Compute point cloud for wrist camera
    depth_wrist = obs["depth_obs"]["depth_gripper"]
    wrist_pcd = deproject(
        env.cameras[1], depth_wrist,
        homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    wrist_pcd = np.reshape(
        wrist_pcd, (depth_wrist.shape[0], depth_wrist.shape[1], 3)
    )
    obs["pcd_obs"]["pcd_gripper"] = static_pcd

    # Map RGB to [0, 1]
    obs["rgb_obs"]["rgb_static"] = obs["rgb_obs"]["rgb_static"] / 255.
    obs["rgb_obs"]["rgb_gripper"] = obs["rgb_obs"]["rgb_gripper"] / 255.

    return obs


def prepare_proprio_states(obs):
    """Prepare robot proprioceptive states. In-place add proprioceptive states
    to the observation dictionary.

    Args:
        obs: a dictionary of observations
            - rgb_obs: a dictionary of RGB images
            - depth_obs: a dictionary of depth images
            - robot_obs: a dictionary of proprioceptive states
    """
    # Map gripper openess to [0, 1]
    proprio = np.concatenate([
        obs['robot_obs'][:3],
        convert_rotation(obs['robot_obs'][3:6]),
        (obs['robot_obs'][[-1]] + 1) / 2
    ], axis=-1)

    if 'proprio' not in obs:
        obs['proprio'] = np.stack([proprio] * 3, axis=0)
    else:
        obs['proprio'] = np.concatenate([obs['proprio'][1:], proprio[None]])

    return obs


def convert_quaternion_to_euler(quat):
    """Convert Euler angles to Quarternion."""
    quat = torch.as_tensor(quat)
    mat = pytorch3d_transforms.quaternion_to_matrix(quat)
    rot = pytorch3d_transforms.matrix_to_euler_angles(mat, "XYZ")
    rot = rot.data.cpu().numpy()

    return rot


def convert_action(trajectory):
    """Convert [position, rotation, openess] to the same format as Calvin

    Args:
        trajectory: a torch.Tensor or np.ndarray of shape [bs, traj_len, 8]
            - position: absolute [x, y, z] in the world coordinates
            - rotation: absolute quarternion in the world coordinates
            - openess: [0, 1]

    Returns:
        trajectory: a torch.Tensor or np.ndarray of shape [bs, traj_len, 8]
            - position: absolute [x, y, z] in the world coordinates
            - rotation: absolute 'XYZ' Euler angles in the world coordinates
            - openess: [-1, 1]
    """
    assert trajectory.shape[-1] == 8
    position, rotation, openess = (
        trajectory[..., :3], trajectory[..., 3:7], trajectory[..., -1:]
    )
    position = position.data.cpu().numpy()
    _rot = convert_quaternion_to_euler(rotation)
    # pytorch3d.transforms does not deal with Gumbel lock, the conversion
    # of some rotation matrix results in nan values.  We usepybullet's
    # implementation in this case.
    if (_rot != _rot).any():
        # Pybullet has different convention of Quaternion.
        _rot_shape = list(rotation.shape)[:-1] + [3]
        _rot = rotation.reshape(-1, 4).data.cpu().numpy()
        rotation = np.array([
            pybullet.getEulerFromQuaternion([r[-1], r[0], r[1], r[2]])
            for r in _rot
        ]).reshape(_rot_shape)
    else:
        rotation = _rot
    openess = (2 * (openess >= 0.5).long() - 1).data.cpu().numpy()

    trajectory = np.concatenate([position, rotation, openess], axis=-1)
    return trajectory


######################################################
#     Functions in calvin_agent.evaluation.utils     #
######################################################
def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def get_env(config_file, show_gui=True):
    render_conf = OmegaConf.load(config_file)

    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(
        render_conf.env,
        show_gui=show_gui,
        use_vr=False,
        use_scene_info=True
    )
    return env


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([
        -2.40851662e-01, 9.24044687e-02, 4.60990009e-01
    ])
    block_slider_right = np.array([
        7.03416330e-02, 9.24044687e-02, 4.60990009e-01
    ])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    import pyhash
    hasher = pyhash.fnv1_32()
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")
            os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


######################################################
#      Functions to cache the evaluation results     #
######################################################
def collect_results(log_dir):
    """Load the number of completed tasks of each instruction chain from a file.
    """
    if os.path.isfile(str(Path(log_dir) / "result.txt")):
        with open(str(Path(log_dir) / "result.txt")) as f:
            lines = f.read().split("\n")[:-1]
    else:
        lines = []

    results, seq_inds = [], []
    for line in lines:
        seq, res = line.split(" ")
        results.append(int(res))
        seq_inds.append(int(seq))

    return results, seq_inds


def write_results(log_dir, seq_ind, result):
    """Write the number of completed tasks of each instruction chain to a file.
    """
    with open(log_dir / f"result.txt", "a") as write_file:
        write_file.write(f"{seq_ind} {result}\n")
