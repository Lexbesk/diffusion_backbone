import io

import cv2
from matplotlib import pyplot as plt
import numpy as np
from torch.nn import functional as F


def compute_metrics(pred, gt):
    # pred/gt are (B, L, 7), mask (B, L)
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
    # symmetric quaternion eval
    quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
    quat_l1_ = (pred[..., 3:7] + gt[..., 3:7]).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
    # gripper openess
    openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).bool()
    tr = 'traj_'

    # Trajectory metrics
    ret_1, ret_2 = {
        tr + 'action_mse': F.mse_loss(pred, gt),
        tr + 'pos_l2': pos_l2.mean(),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
        tr + 'rot_l1': quat_l1.mean(),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
        tr + 'gripper': openess.flatten().float().mean()
    }, {
        tr + 'pos_l2': pos_l2.mean(-1),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
        tr + 'rot_l1': quat_l1.mean(-1),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1)
    }

    return ret_1, ret_2


def fig_to_numpy(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


def generate_visualizations(pred, gt, box_size=0.3):
    batch_idx = 0
    pred = pred[batch_idx].detach().cpu().numpy()
    gt = gt[batch_idx].detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    if pred.ndim == 2 and gt.ndim == 2:
        ax.scatter3D(
            pred[:, 0], pred[:, 1], pred[:, 2],
            color='red', label='pred'
        )
        ax.scatter3D(
            gt[:, 0], gt[:, 1], gt[:, 2],
            color='blue', label='gt'
        )
        center = gt.mean(0)
    elif pred.ndim == 3 and gt.ndim == 3:
        ax.scatter3D(
            pred[:, 0, 0], pred[:, 0, 1], pred[:, 0, 2],
            color='red', label='pred-left'
        )
        if(pred.shape[1]>1):
            ax.scatter3D(
                pred[:, 1, 0], pred[:, 1, 1], pred[:, 1, 2],
                color='magenta', label='pred-right'
            )
        ax.scatter3D(
            gt[:, 0, 0], gt[:, 0, 1], gt[:, 0, 2],
            color='blue', label='gt-left'
        )
        if(pred.shape[1]>1):
            ax.scatter3D(
                gt[:, 1, 0], gt[:, 1, 1], gt[:, 1, 2],
                color='cyan', label='gt-right'
            )
        center = np.reshape(gt, (-1, gt.shape[-1])).mean(0)
    else:
        raise ValueError("Invalid dimensions")

    ax.set_xlim(center[0] - box_size, center[0] + box_size)
    ax.set_ylim(center[1] - box_size, center[1] + box_size)
    ax.set_zlim(center[2] - box_size, center[2] + box_size)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.legend()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    img = fig_to_numpy(fig, dpi=120)
    plt.close()
    return img.transpose(2, 0, 1)
