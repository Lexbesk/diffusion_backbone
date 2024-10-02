import torch

import utils.pytorch3d_transforms as pytorch3d_transforms


def to_relative_action(actions, robot_obs):
    assert actions.shape[-1] == 16 and robot_obs.shape[-1] == 16

    actions = actions.unflatten(-1, (2, 8))
    robot_obs = robot_obs.unflatten(-1, (2, 8))

    rel_pos = actions[..., :3] - robot_obs[..., :3]

    # pytorch3d takes wxyz quaternion, the input is xyzw
    rel_orn = pytorch3d_transforms.quaternion_multiply(
        actions[..., [6, 3, 4, 5]],
        pytorch3d_transforms.quaternion_invert(robot_obs[..., [6,3,4,5]])
    )[..., [1, 2, 3, 0]]

    gripper = actions[..., -1:]
    rel_actions = torch.concat([rel_pos, rel_orn, gripper], dim=-1)
    rel_actions = rel_actions.flatten(-2)

    return rel_actions

def to_absolute_action(actions, robot_obs):
    assert actions.shape[-1] == 8 and robot_obs.shape[-1] == 8

    # actions = actions.unflatten(-1, (2, 8))
    # robot_obs = robot_obs.unflatten(-1, (2, 8))

    abs_pos = actions[..., :3] + robot_obs[..., :3]

    # pytorch3d takes wxyz quaternion, the input is xyzw
    abs_orn = pytorch3d_transforms.quaternion_multiply(
        actions[..., [6, 3, 4, 5]],
        robot_obs[..., [6,3,4,5]]
    )[..., [1, 2, 3, 0]]

    gripper = actions[..., -1:]
    abs_actions = torch.concat([abs_pos, abs_orn, gripper], dim=-1)
    abs_actions = abs_actions.flatten(-2)

    return abs_actions