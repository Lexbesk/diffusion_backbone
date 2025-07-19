import os
import pdb

import trimesh
import numpy as np
import mujoco
import mujoco.viewer
import transforms3d.quaternions as tq

from .rot_util import interplote_pose, interplote_qpos
import imageio.v2 as imageio

import numpy as np

def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix from a MuJoCo (w,x,y,z) quaternion."""
    quat = np.ascontiguousarray(quat_wxyz, dtype=np.float64).reshape(4)   # (4,)
    mat_flat = np.empty(9, dtype=np.float64)                              # (9,)
    mujoco.mju_quat2Mat(mat_flat, quat)                                   # fills in-place
    return mat_flat.reshape(3, 3)

def _lookat_to_xmat(cam_pos: np.ndarray,
                    target: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    Build the 3 × 3 row-major rotation matrix mjModel expects in `cam_xmat`
    so that the camera at `cam_pos` looks at `target` with +Z as up.

    Returns
    -------
    (3, 3) np.ndarray   # rows = camera x-, y-, z-axes
    """
    fwd = (target - cam_pos)
    fwd /= np.linalg.norm(fwd) + 1e-9

    up_tmp = np.array([0., 0., 1.])
    right = np.cross(fwd, up_tmp)
    if np.linalg.norm(right) < 1e-6:              # coplanar with +Z – pick +Y
        up_tmp = np.array([0., 1., 0.])
        right = np.cross(fwd, up_tmp)
    right /= np.linalg.norm(right) + 1e-9

    up = np.cross(right, fwd)
    return np.stack([right, up, -fwd], axis=0) 

def lookat_to_xyaxes(pos, lookat, up=(0, 0, 1)):
    """
    Convert (pos, lookat) into a MuJoCo-compatible xyaxes list.

    Parameters
    ----------
    pos : (3,) array-like
        World-space camera position.
    lookat : (3,) array-like
        World-space point the camera should look at.
    up : (3,) array-like, optional
        Preferred world-up direction.  Defaults to +Z.

    Returns
    -------
    list[float]  # length 6
        First three numbers – camera X axis (points right in the image)
        Next three numbers  – camera Y axis (points up   in the image)

    Notes
    -----
    * The camera optical axis is –Z in MuJoCo; X points right, Y points up.:contentReference[oaicite:0]{index=0}
    * The two returned axes are **unit-length** and mutually orthogonal.
    * If `pos` and `lookat` are coincident, a ValueError is raised.
    """
    pos   = np.asarray(pos,   dtype=float)
    look  = np.asarray(lookat, dtype=float)
    up    = np.asarray(up,    dtype=float)

    forward = look - pos
    norm_f  = np.linalg.norm(forward)
    if norm_f < 1e-9:
        raise ValueError("`pos` and `lookat` cannot be the same point")
    forward /= norm_f

    z_axis = -forward                     # camera –Z  (view direction)
    x_axis = np.cross(up, z_axis)         # world-up ⨯ –Z → camera X
    if np.linalg.norm(x_axis) < 1e-9:     # forward colinear with up → pick new up
        x_axis = np.cross([0, 1, 0], z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)     # right-handed frame
    y_axis /= np.linalg.norm(y_axis)

    return [*x_axis, *y_axis]


def add_axes(scene, pos, rot, length=0.1, radius=0.005):
    """Draw RGB XYZ axes at (pos, rot) in world frame."""
    colors = (
        ([1, 0, 0, 1], np.array([1, 0, 0])),   # X (red)
        ([0, 1, 0, 1], np.array([0, 1, 0])),   # Y (green)
        ([0, 0, 1, 1], np.array([0, 0, 1])),   # Z (blue)
    )
    for rgba, axis_dir in colors:
        geom = scene.geoms[scene.ngeom]               # grab a free slot
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),            # size filled by connector
            np.zeros(3, dtype=np.float64),            # pos   "
            np.zeros(9, dtype=np.float64),            # mat   "
            np.asarray(rgba, dtype=np.float32),       # colour
        )
        # Set arrow endpoints
        a = pos
        b = pos + length * (rot @ axis_dir)           # world-space tip
        mujoco.mjv_connector(                         # fills size/pos/mat
            geom, mujoco.mjtGeom.mjGEOM_ARROW, radius,
            *a, *b
        )
        scene.ngeom += 1

def _mat2quat(M: np.ndarray) -> np.ndarray:
    mat_col = M.T.ravel().astype(np.float64, copy=False)   # 9-vector, col-major
    q = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, mat_col)
    return q

class MjHO:

    hand_prefix: str = "child-"

    def __init__(
        self,
        obj_path,
        obj_scale,
        obj_density,
        hand_xml_path,
        hand_mocap,
        exclude_table_contact,
        friction_coef,
        has_floor_z0,
        debug_render=False,
        debug_viewer=False,
        configs=None,
    ):
        self.hand_mocap = hand_mocap
        self.configs = configs
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # self.spec.option.timestep = 0.004
        # self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY
        self.spec.option.timestep = 0.004
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # Disable gravity (as before) *and* contact handling.
        if self.configs.task.simulation_test:
            self.spec.option.disableflags = (
                mujoco.mjtDisableBit.mjDSBL_GRAVITY
                # | mujoco.mjtDisableBit.mjDSBL_CONTACT        # <-- no collisions
            )
        else:
            self.spec.option.disableflags = (
                mujoco.mjtDisableBit.mjDSBL_GRAVITY
                | mujoco.mjtDisableBit.mjDSBL_CONTACT        # <-- no collisions
            )
        
        if debug_render or debug_viewer:
            self.spec.add_texture(
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=[0.3, 0.5, 0.7],
                rgb2=[0.3, 0.5, 0.7],
                width=512,
                height=512,
            )
            self.spec.worldbody.add_light(
                name="spotlight",
                pos=[0, -1, 2],
                castshadow=False,
            )
            cam_pos    = [0.3, 0.3, 0.3]
            cam_lookat = [0.00, 0.00, 0.00]
            xyaxes = lookat_to_xyaxes(cam_pos, cam_lookat)
            # print(xyaxes)
            # self.spec.worldbody.add_camera(
            #     name="closeup", pos=[0.75, 1.0, 1.0], xyaxes=[-1, 0, 0, 0, -1, 1]
            # )
            self.spec.worldbody.add_camera(
                name="closeup", pos=cam_pos, xyaxes=xyaxes
            )

        self._add_hand(hand_xml_path, hand_mocap)
        self._add_object(obj_path, obj_scale, obj_density, has_floor_z0)
        self._set_friction(friction_coef)
        self.spec.add_key()
        if exclude_table_contact is not None:
            for body_name in exclude_table_contact:
                self.spec.add_exclude(
                    bodyname1="world", bodyname2=f"{self.hand_prefix}{body_name}"
                )

        # Get ready for simulation
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # For ctrl
        qpos2ctrl_matrix = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            qpos2ctrl_matrix,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        self._qpos2ctrl_matrix = qpos2ctrl_matrix[..., :-6]

        self.debug_viewer = None
        self.debug_render = None
        if debug_viewer:
            self.debug_viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.debug_viewer.sync()
            pdb.set_trace()

        if debug_render:
            # pdb.set_trace()
            self.debug_render = mujoco.Renderer(self.model, 480, 640)
            self.debug_options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.debug_options)
            self.debug_options.frame        = mujoco.mjtFrame.mjFRAME_WORLD      # body frames (pick any of mjFRAME_BODY/GEOM/SITE/CAMERA/LIGHT/WORLD)
            self.model.vis.scale.framelength = 5    # metres, default = 1
            self.model.vis.scale.framewidth  = 0.05
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            # self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_FRAME] = True

            self.debug_images = []
            
            self.cam_step     = 0.005            # metres to move to the right each frame
            self.cam_id       = mujoco.mj_name2id(
                                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, "closeup")
            self._cam_origin  = self.model.cam_pos[self.cam_id].copy()
            self._cam_frames  = 0
        return
    
    def _move_camera(self):
        """Shift the 'closeup' camera +X each frame and keep it looking at (0,0,0)."""
        self._cam_frames += 1

        new_pos = self._cam_origin + np.array([-self._cam_frames * self.cam_step,
                                            0.0, 0.0])
        self.model.cam_pos[self.cam_id] = new_pos

        R = _lookat_to_xmat(new_pos)            # 3×3 row-major rotation


        # write orientation – prefer cam_mat if it’s compiled in, else fall back to cam_quat
        if hasattr(self.model, "cam_mat"):
            self.model.cam_mat[self.cam_id] = R.ravel()
        self.model.cam_quat[self.cam_id] = _mat2quat(R)

    def _add_hand(self, xml_path, mocap_base):
        # Read hand xml
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            # This solimp and solref comes from the Shadow Hand xml
            # They can generate larger force with smaller penetration
            # The body will be more "rigid" and less "soft"
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self.hand_prefix, ""
        )
        # Add freejoint and mocap of hand root
        if mocap_base:
            child_world.add_freejoint(name="hand_freejoint")
            self.spec.worldbody.add_body(name="mocap_body", mocap=True)
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=[0.9, 0.95, 0.001, 0.5, 2],
                data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            )
        return

    def _add_object(self, obj_path, obj_scale, obj_density, has_floor_z0):
        if has_floor_z0:
            floor_geom = self.spec.worldbody.add_geom(
                name="object_collision_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, 0],
                size=[0, 0, 1.0],
            )

        obj_body = self.spec.worldbody.add_body(name="object")
        obj_body.add_freejoint(name="obj_freejoint")
        parts_folder = os.path.join(obj_path, "urdf/meshes")
        for file in os.listdir(parts_folder):
            file_path = os.path.join(parts_folder, file)
            mesh_name = file.replace(".obj", "")
            mesh_id = mesh_name.replace("convex_piece_", "")

            self.spec.add_mesh(
                name=mesh_name,
                file=file_path,
                scale=[obj_scale, obj_scale, obj_scale],
            )
            obj_body.add_geom(
                name=f"object_visual_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=0,
                contype=0,
                conaffinity=0,
            )
            obj_body.add_geom(
                name=f"object_collision_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=obj_density,
            )

        return

    def _set_friction(self, test_friction):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = test_friction
            g.condim = 4
        return

    def _qpos2ctrl(self, hand_qpos):
        if self.hand_mocap:
            return self._qpos2ctrl_matrix[:, 6:] @ hand_qpos[7:]
        else:
            return self._qpos2ctrl_matrix @ hand_qpos

    def get_obj_pose(self):
        return self.data.qpos[-7:]

    def get_contact_info(self, hand_qpos, obj_pose, obj_margin=0):
        # Set margin and gap to detect contact
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = obj_margin

        # Set pose and qpos for hand and object
        self.reset_pose_qpos(hand_qpos, obj_pose)

        object_id = self.model.nbody - 1
        hand_id = self.model.nbody - 2
        world_id = -1 if self.hand_mocap else 0

        # Processing all contact information
        ho_contact = []
        hh_contact = []
        for contact in self.data.contact:
            body1_id = self.model.geom(contact.geom1).bodyid
            body2_id = self.model.geom(contact.geom2).bodyid
            body1_name = self.model.body(self.model.geom(contact.geom1).bodyid).name
            body2_name = self.model.body(self.model.geom(contact.geom2).bodyid).name
            # hand and object
            if (
                body1_id > world_id and body1_id < hand_id and body2_id == object_id
            ) or (body2_id > world_id and body2_id < hand_id and body1_id == object_id):
                # keep body1=hand and body2=object
                if body2_id == object_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name.removeprefix(self.hand_prefix)
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name.removeprefix(self.hand_prefix)
                    obj_body_name = body1_name
                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact_normal,
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )
            # hand and hand
            elif (
                body1_id > world_id
                and body1_id < hand_id
                and body2_id > world_id
                and body2_id < hand_id
            ):
                hh_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact.frame[0:3],
                        "body1_name": body1_name,
                        "body2_name": body2_name,
                    }
                )
            # else:
            #     print(body1_name, body2_name, body1_id, body2_id)

        # Set margin and gap back
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = 0
        return ho_contact, hh_contact

    def set_ext_force_on_obj(self, ext_force):
        self.data.xfrc_applied[-1] = ext_force
        return

    def reset_pose_qpos(self, hand_qpos, obj_pose):
        # set key frame
        self.model.key_qpos[0] = np.concatenate([hand_qpos, obj_pose], axis=0)
        self.model.key_ctrl[0] = self._qpos2ctrl(hand_qpos)
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        if self.hand_mocap:
            self.model.key_mpos[0] = hand_qpos[:3]
            self.model.key_mquat[0] = hand_qpos[3:7]

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return

    def control_hand_with_interp(
        self, hand_qpos1, hand_qpos2, partial_points, step_outer=10, step_inner=10, scale=0.06
    ):
        if self.hand_mocap:
            pose_interp = interplote_pose(hand_qpos1[:7], hand_qpos2[:7], step_outer)
        qpos_interp = interplote_qpos(
            self._qpos2ctrl(hand_qpos1), self._qpos2ctrl(hand_qpos2), step_outer
        )
        for j in range(step_outer):
            if self.hand_mocap:
                self.data.mocap_pos[0] = pose_interp[j, :3]
                self.data.mocap_quat[0] = pose_interp[j, 3:7]
            self.data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.model, self.data)
            self.control_hand_step(step_inner, partial_points, scale=scale)
        return
    
    def _draw_partial_pcd(self, partial_points, scale):
            # transform point cloud from object frame ➜ world frame
            obj_pos, obj_quat = self.get_obj_pose()[:3], self.get_obj_pose()[3:]
            # print(f"Object pose: {obj_pos}, {obj_quat}")
            rot_mat = quat_to_mat(obj_quat)       # (3,3)
            # world_pts = obj_pos + partial_points @ rot_mat.T
            world_pts = partial_points

            scene = self.debug_render.scene
            # print(len(world_pts), scene.ngeom, scene.maxgeom)
            for i, p in enumerate(world_pts):
                # print(p)
                # p = p * 10
                # if scene.ngeom >= scene.maxgeom:
                #     break              # avoid overflow; raise if you prefer
                geom = scene.geoms[scene.ngeom]        # get write-slot
                rgba = [1.0, 0.2, 0.4, 1.0] if i <= 22 else [0.1, 0.6, 1.0, 1.0]
                
                mujoco.mjv_initGeom(                   # low-level C helper
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,      # type
                    np.array([scale, scale, scale], float),   # size; only x used
                    p.astype(float),                   # position
                    np.eye(3, dtype=np.float64).ravel(),                     # orientation
                    np.array(rgba, float)              # colour
                )
                geom.dataid = -1       # -- not linked to existing mesh/texture
                geom.objtype = -1
                scene.ngeom += 1
                # print(i)


    def control_hand_step(self, step_inner, partial_points, scale=0.06):
        for _ in range(step_inner):
            mujoco.mj_step(self.model, self.data)

        if self.debug_render is not None:
            self._move_camera()  
            self.debug_render.update_scene(self.data, camera="closeup", scene_option=self.debug_options)
            if not self.configs.task.simulation_test:
                self._draw_partial_pcd(partial_points, scale=scale)
            scene = self.debug_render.scene
            pixels = self.debug_render.render()
            self.debug_images.append(pixels)

        if self.debug_viewer is not None:
            raise NotImplementedError
        return
    
    def save_render(self, gif_path, png_path, fps: int = 30):
        """
        Write the first frame as PNG and the whole roll-out as an animated GIF.

        Call once after your simulation loop finishes.
        """
        if not self.debug_images:
            raise RuntimeError("Nothing rendered  run the sim first.")

        # MuJoCo gives uint8 RGB already; guard against float.
        frames = [f.astype(np.uint8) if f.dtype != np.uint8 else f
                  for f in self.debug_images]

        # imageio.imwrite(png_path, frames[0])      # first frame
        imageio.mimsave(gif_path,  frames, fps=fps)
        
    


class RobotKinematics:
    def __init__(self, xml_path):
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mesh_geom_info = {}
        for i in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(i)
            mesh_id = geom.dataid
            if mesh_id != -1:
                mjm = self.mj_model.mesh(mesh_id)
                vert = self.mj_model.mesh_vert[
                    mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]
                ]
                face = self.mj_model.mesh_face[
                    mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]
                ]
                body_name = self.mj_model.body(geom.bodyid).name
                mesh_name = mjm.name
                self.mesh_geom_info[f"{body_name}_{mesh_name}"] = {
                    "vert": vert,
                    "face": face,
                    "geom_id": i,
                }

        return

    def forward_kinematics(self, q):
        self.mj_data.qpos = q
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        return

    def get_init_meshes(self):
        init_mesh_lst = []
        mesh_name_lst = []
        for k, v in self.mesh_geom_info.items():
            mesh_name_lst.append(k)
            init_mesh_lst.append(trimesh.Trimesh(vertices=v["vert"], faces=v["face"]))
        return mesh_name_lst, init_mesh_lst

    def get_poses(self, root_pose):
        geom_poses = np.zeros((len(self.mesh_geom_info), 7))
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        for i, v in enumerate(self.mesh_geom_info.values()):
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_poses[i, :3] = root_rot @ geom_trans + root_trans
            geom_poses[i, 3:] = tq.mat2quat(root_rot @ geom_rot)
        return geom_poses

    def get_posed_meshes(self, root_pose):
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        full_tm = []
        for k, v in self.mesh_geom_info.items():
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            posed_vert = (v["vert"] @ geom_rot.T + geom_trans) @ root_rot.T + root_trans
            posed_tm = trimesh.Trimesh(vertices=posed_vert, faces=v["face"])
            full_tm.append(posed_tm)
        full_tm = trimesh.util.concatenate(full_tm)
        return full_tm


if __name__ == "__main__":
    xml_path = os.path.join(
        os.path.dirname(__file__), "../../assets/hand/shadow/customized.xml"
    )
    kinematic = RobotKinematics(xml_path)
    hand_qpos = np.zeros((22))
    kinematic.forward_kinematics(hand_qpos)
    visual_mesh = kinematic.get_posed_meshes()
    visual_mesh.export(f"debug_hand.obj")
