import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .rot_util import np_get_delta_qpos
from .base import BaseEval
import mujoco

class fcMocapEval(BaseEval):
    def _simulate_under_extforce_details(self, pre_obj_qpos):
        external_force_direction = np.array(
            [
                [-1.0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        ) 
        
        all_points_to_mark = self.grasp_data['partial_points'][:]
        # all_points_to_mark = np.concatenate(
        #     [all_points_to_mark, self.grasp_data["palm_centre"][None], self.grasp_data["thumb_tip"][None],
        #      self.grasp_data["middle_tip"][None]], axis=0)
        
        scale = 0.005
        if self.configs.task.simulation_test:
            visualize = False
        else:
            visualize = True
        
        # assert all_points_to_mark.shape == (4, 3)
        
        for i in range(len(external_force_direction)):
            if visualize:
                self.mj_ho.reset_pose_qpos(
                    self.grasp_data["grasp_qpos"],
                    self.grasp_data["obj_pose"],
                )
            else:
                self.mj_ho.reset_pose_qpos(
                    self.grasp_data["pregrasp_qpos"],
                    self.grasp_data["obj_pose"],
                )

            # 2. Move hand to grasp pose
            if not visualize:
                self.mj_ho.control_hand_with_interp(
                    self.grasp_data["pregrasp_qpos"],
                    self.grasp_data["grasp_qpos"],
                    all_points_to_mark, scale=scale
                )

                # 3. Move hand to squeeze pose.
                # NOTE step 2 and 3 are seperate because pre -> grasp -> squeeze are stage-wise linear.
                # If step 2 and 3 are merged to one linear interpolation, the performance will drop a lot.
                self.mj_ho.control_hand_with_interp(
                    self.grasp_data["grasp_qpos"],
                    self.grasp_data["squeeze_qpos"],
                    all_points_to_mark, scale=scale
                )

                # 4. Add external force on the object
                self.mj_ho.set_ext_force_on_obj(
                    10 * external_force_direction[i] * self.configs.task.obj_mass
                )

            # 5. Wait for 2 seconds
            for _ in range(10):
                self.mj_ho.control_hand_step(step_inner=50, partial_points=all_points_to_mark, scale=scale)

                # Early stop
                latter_obj_qpos = self.mj_ho.get_obj_pose()
                delta_pos, delta_angle = np_get_delta_qpos(
                    pre_obj_qpos, latter_obj_qpos
                )
                succ_flag = (
                    delta_pos < self.configs.task.simulation_metrics.trans_thre
                ) & (delta_angle < self.configs.task.simulation_metrics.angle_thre)
                if not succ_flag:
                    break
            if not succ_flag:
                break

        return
