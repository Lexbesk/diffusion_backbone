import os
from os.path import join as pjoin
from glob import glob
import random

import numpy as np
from torch.utils.data import Dataset

from .utils import numpy_quaternion_to_matrix, quat2mat
from .utils import load_json, load_scene_cfg

from pathlib import Path
import json
import trimesh                     # pip install trimesh
import mujoco


# In hand (grasp) frame, millimetres.
PALM_CENTER_H      = np.array([  0.0,   0.0,   0.0])
THUMB_TIP_H        = np.array([ 58.4, -25.7,  11.2])
MIDDLE_TIP_H       = np.array([ 78.3,   0.0,  19.6])

MID_TIP_H          = 0.5 * (THUMB_TIP_H + MIDDLE_TIP_H)
DIR_H              = (MID_TIP_H - PALM_CENTER_H)           # “heading/approach”
DIR_H /= np.linalg.norm(DIR_H)                             # normalise
DIR_H = np.array([0, 0, 1])
I = 0


def anchor_point(full_pcd_w, R_w_h, t_w_h):
    """
    full_pcd_w : (N,3) np.ndarray   scene/object cloud in WORLD frame
    R_w_h      : (3,3) np.ndarray   grasp rotation, hand→world
    t_w_h      : (3,)  np.ndarray   grasp translation, hand→world
    Returns
    -------
    anchor_point_w : (3,) np.ndarray  the anchor point in WORLD frame
    """
    # 1⃣  transform the two constant vectors just once
    # R_w_h = np.linalg.inv(R_w_h)  # hand→world

    # 2⃣  vector from palm to every point
    
    distances = np.linalg.norm(full_pcd_w - t_w_h, axis=1)  # (N,)
    # select the nearest point without threshold
    min_index = np.argmin(distances)
    anchor_pt = full_pcd_w[min_index]  # (3,)
    # global I
    # if I == 0:  # Debugging
    #     np.savez(
    #         "anchor_debug.npz",
    #         full_pcd_w=full_pcd_w,          # (N,3)   original cloud
    #         wrist=t_w_h,                  # (3,)    the grasp translation
    #         anchor_pt=anchor_pt             # (3,)    the chosen anchor
    #         )
    #     print(f"Anchor debug saved to anchor_debug.npz")
    # I += 1

    return anchor_pt

def pick_cloud_with_anchor(clouds, anchor_w, tol=1e-3):
    """
    clouds    : list[ np.ndarray ]       # [(N₀,3), (N₁,3), …]   world-frame partial PCs
    anchor_w  : (3,) np.ndarray          # the target 3-D anchor in the same frame
    tol       : float                    # ≡ 1 mm  (in metres)

    Returns
    --------
    view_idx  : int                      # which cloud in `clouds` contains the point
    point_idx : int                      # index of that point inside clouds[view_idx]
    """
    tol_sq = tol * tol
    for view_idx, pc in enumerate(clouds):
        # squared Euclidean distance to every point (vectorised, no loops)
        d2 = ((pc - anchor_w) ** 2).sum(axis=1)
        # indices whose distance ≤ 1 mm
        hit = np.flatnonzero(d2 <= tol_sq)
        if hit.size:                     # found our anchor
            return view_idx, int(hit[0]) # first match is enough

    raise ValueError("Anchor not found in any candidate point cloud")



def load_object(scene_cfg: dict,
                repo_root=".",     # project root on your disk
                n_sample=4096):    # how many points to sample if no PC file
    """
    Returns:
        mesh  : trimesh.Trimesh  already scaled & posed
        points: (N,3) float32    point cloud in the same frame
    """
    # ---- 1. parse the dict --------------------------------------------------
    obj_uid, obj_entry = next(iter(scene_cfg["scene"].items()))

    # Relative paths inside the repo → absolute, with ".." collapsed
    mesh_path = Path(repo_root, obj_entry["file_path"]).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    # ---- 2. load & transform the mesh ---------------------------------------
    mesh = trimesh.load(mesh_path, process=False)

    # scale (isotropic here but stored as 3-vector)
    mesh.apply_scale(obj_entry["scale"])

    # # pose: [tx, ty, tz, qw, qx, qy, qz]
    # pose = obj_entry["pose"]
    # trans, quat = pose[:3], pose[3:]
    # # trimesh expects [w, x, y, z]
    # T = trimesh.transformations.quaternion_matrix(quat)
    # T[:3, 3] = trans
    # mesh.apply_transform(T)

    # ---- 3. get / build a point cloud ---------------------------------------
    # Most processed objects have a point cloud under processed_data/<uid>/pc
    pc_candidates = list(mesh_path.parents[1].glob("pc/*.*"))
    points = None
    if pc_candidates:
        pc_path = pc_candidates[0]
        if pc_path.suffix == ".npy":
            points = np.load(pc_path)[:, :3]
        elif pc_path.suffix in {".ply", ".xyz"}:
            points = trimesh.load(pc_path).vertices
    if points is None:
        # Fallback: uniform random samples on the surface
        points = mesh.sample(n_sample)

    return mesh, points.astype(np.float32)


class DexDataset(Dataset):
    def __init__(self, config: dict, mode: str, sc_voxel_size: float = None):
        self.config = config
        self.sc_voxel_size = sc_voxel_size
        self.mode = mode

        if self.config.grasp_type_lst is not None:
            self.grasp_type_lst = self.config.grasp_type_lst
        else:
            self.grasp_type_lst = os.listdir(self.config.grasp_path)
        self.grasp_type_num = len(self.grasp_type_lst)
        self.object_pc_folder = pjoin(self.config.object_path, self.config.pc_path)

        if mode == "train":
            self.init_train(mode)
        elif mode == "eval":
            self.init_eval(mode)
        elif mode == "test":
            self.init_test()
        

        MODEL_PATH = '/home/austinz/Projects/manipulation/Regrasping/diffusion_backbone/DexGraspBench/assets/hand/shadow/right_hand.xml'
        self.hand_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.hand_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.sid_palm  = mujoco.mj_name2id(self.hand_model, mujoco.mjtObj.mjOBJ_SITE, b"palm_centre")
        self.sid_thumb = mujoco.mj_name2id(self.hand_model, mujoco.mjtObj.mjOBJ_SITE, b"thumb_tip")
        self.sid_mid   = mujoco.mj_name2id(self.hand_model, mujoco.mjtObj.mjOBJ_SITE, b"middle_tip")
        self.hand_model_data  = mujoco.MjData(self.hand_model)
        # print(self.hand_model_data.qpos.shape, 'qpos of mujoco model') # (22,)
        
        return

    def init_train(self, mode):
        split_name = "test" if mode == "eval" else "train"
        self.obj_id_lst = load_json(
            pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json")
        )

        self.grasp_obj_dict = {}
        self.data_num = 0
        self.index = []
        for grasp_type in self.grasp_type_lst:
            self.grasp_obj_dict[grasp_type] = []
            for obj_id in self.obj_id_lst:
                obj_grasp_data = len(
                    glob(
                        pjoin(self.config.grasp_path, grasp_type, obj_id, "**/**.npy"),
                        recursive=True,
                    )
                )
                if obj_grasp_data == 0:
                    continue
                self.data_num += obj_grasp_data
                self.grasp_obj_dict[grasp_type].append(obj_id)
                for gpath in glob(pjoin(self.config.grasp_path, grasp_type, obj_id, "**/**.npy"), recursive=True):
                    gdata = np.load(gpath, allow_pickle=True).item()
                    n_poses = gdata["pregrasp_qpos"].shape[0]

                    # point-cloud views for this object
                    scene_cfg = load_scene_cfg(gdata["scene_path"])
                    pc_paths = sorted(
                        glob(pjoin(self.config.object_path, self.config.pc_path,
                                   scene_cfg["scene_id"], "partial_pc**.npy"))
                    )

                    for pose_idx in range(n_poses):
                        for pc_path in pc_paths:           # enumerate every view
                            self.index.append((grasp_type,
                                               obj_id,
                                               gpath,
                                               pose_idx,
                                               pc_path))     
                    if split_name == "test":
                        # for test split, we only need one grasp pose per object
                        break            
            if len(self.grasp_obj_dict[grasp_type]) == 0:
                self.grasp_obj_dict.pop(grasp_type)
        print(
            f"mode: {mode}, grasp type number: {self.grasp_type_num}, grasp data num: {self.data_num}, all data: {len(self.index)}"
            
        )
        return
    
    def init_eval(self, mode):
        assert mode == "eval"
        split_name = "test"
        self.obj_id_lst = load_json(
            pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json")
        )

        self.grasp_obj_dict = {}
        self.data_num = 0
        self.index = []
        for grasp_type in self.grasp_type_lst:
            self.grasp_obj_dict[grasp_type] = []
            for obj_id in self.obj_id_lst:
                obj_grasp_data = len(
                    glob(
                        pjoin(self.config.grasp_path, grasp_type, obj_id, "**/**.npy"),
                        recursive=True,
                    )
                )
                if obj_grasp_data == 0:
                    continue
                self.data_num += obj_grasp_data
                self.grasp_obj_dict[grasp_type].append(obj_id)
                for gpath in glob(pjoin(self.config.grasp_path, grasp_type, obj_id, "**/**.npy"), recursive=True):
                    gdata = np.load(gpath, allow_pickle=True).item()
                    n_poses = gdata["pregrasp_qpos"].shape[0]

                    # point-cloud views for this object
                    scene_cfg = load_scene_cfg(gdata["scene_path"])
                    pc_paths = sorted(
                        glob(pjoin(self.config.object_path, self.config.pc_path,
                                   scene_cfg["scene_id"], "partial_pc**.npy"))
                    )

                    for pose_idx in range(n_poses):
                        for pc_path in pc_paths:           # enumerate every view
                            self.index.append((grasp_type,
                                               obj_id,
                                               gpath,
                                               pose_idx,
                                               pc_path))     
                    if split_name == "test":
                        # for test split, we only need one grasp pose per object
                        break            
            if len(self.grasp_obj_dict[grasp_type]) == 0:
                self.grasp_obj_dict.pop(grasp_type)
        print(
            f"mode: {mode}, grasp type number: {self.grasp_type_num}, grasp data num: {self.data_num}, all data: {len(self.index)}"
            
        )
        return

    def init_test(self):
        split_name = self.config.test_split
        self.obj_id_lst = []
        self.test_cfg_lst = []
        self.obj_id_lst = load_json(
            pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json")
        )
        if self.config.mini_test:
            self.obj_id_lst = self.obj_id_lst[:100]
        for o in self.obj_id_lst:
            self.test_cfg_lst.extend(
                glob(
                    pjoin(
                        self.config.object_path,
                        "scene_cfg",
                        o,
                        self.config.test_scene_cfg,
                    )
                )
            )
        self.data_num = self.grasp_type_num * len(self.test_cfg_lst)
        print(
            f"Test split: {split_name}, grasp type number: {self.grasp_type_num}, object cfg num: {len(self.test_cfg_lst)}"
        )
        return

    def __len__(self):
        return len(self.index)
    
    def fk_points_from_grasp1(self, grasp29):
        """
        Input
        grasp29 : (29,)  [ xyz  qwqxqyqz  22 angles ]
        Returns
        p_c, t_th, t_mid  : each (3,) world-frame numpy arrays
        """

        self.hand_model_data.qpos[:]  = grasp29[7:29]       # 22 joint angles

        # propagate FK
        mujoco.mj_forward(self.hand_model, self.hand_model_data)

        # --- grab the 3 sites ---------------------------------------------------
        # sid_palm   = self.hand_model.site_name2id("palm_centre")
        # sid_thumb  = self.hand_model.site_name2id("thumb_tip")
        # sid_middle = self.hand_model.site_name2id("middle_tip")
        
        sid_palm   = mujoco.mj_name2id(self.hand_model, mujoco.mjtObj.mjOBJ_SITE,
                               b"palm_centre")
        sid_thumb  = mujoco.mj_name2id(self.hand_model, mujoco.mjtObj.mjOBJ_SITE,
                                    b"thumb_tip")
        sid_middle = mujoco.mj_name2id(self.hand_model, mujoco.mjtObj.mjOBJ_SITE,
                                    b"middle_tip")

        p_c   = self.hand_model_data.site_xpos[sid_palm].copy()
        t_th  = self.hand_model_data.site_xpos[sid_thumb].copy()
        t_mid = self.hand_model_data.site_xpos[sid_middle].copy()

        return p_c, t_th, t_mid
    
    def fk_points_from_grasp(self, grasp29):
        """
        grasp29 : (29,) = [xyz, qwqxqyqz, 22 joint angles]
        returns  : palm centre, thumb tip, middle-finger tip  (each (3,))
        """
        d = self.hand_model_data          # local alias

        # --- set joint configuration -------------------------------------------
        d.qpos[:] = grasp29[7:29]          # 22 DoF
        d.qvel[:] = 0                      # ensure clean state (optional but cheap)

        # --- fast FK: positions only, no contacts ------------------------------
        mujoco.mj_forwardSkip(
            self.hand_model,
            d,
            mujoco.mjtStage.mjSTAGE_POS,
            0
        )

        # --- fetch pre-cached sites -------------------------------------------
        p_c   = d.site_xpos[self.sid_palm]
        t_th  = d.site_xpos[self.sid_thumb]
        t_mid = d.site_xpos[self.sid_mid]

        return p_c.copy(), t_th.copy(), t_mid.copy()
    
    def fk_points_from_grasp_nofree(self, grasp29):
        pos = grasp29[:3]            # x,y,z
        quat= grasp29[3:7]           # w,x,y,z

        p_c_l, t_th_l, t_mid_l = self.fk_points_from_grasp(grasp29)
        R   = quat2mat(quat)
        p_c   = pos + R @ p_c_l
        t_th  = pos + R @ t_th_l
        t_mid = pos + R @ t_mid_l
        return p_c, t_th, t_mid
    

    def anchor_point_from_hand(self, p_c, t_th, t_mid, pts):
        """
        Args
        ----
        p_c   : (3,)  palm-centre in world frame  (numpy)
        t_th  : (3,)  thumb-tip  "
        t_mid : (3,)  middle-tip "
        pts   : (N,3) object point-cloud in the *same* world frame

        Returns
        -------
        anchor : (3,)  the point-cloud point chosen as anchor
        idx    : int   its index in pts  (handy for debugging / weighting)
        """

        # 1. heading direction  (palm -> midpoint between tips)
        midpoint = 0.5*(t_th + t_mid)
        v        = midpoint - p_c
        v_norm   = np.linalg.norm(v)
        # if v_norm < 1e-8:
        #     raise ValueError("Degenerate heading vector: thumb & middle coincide with palm")
        if v_norm < 1e-8:
            # choose the point nearest to the palm centre
            idx = np.argmin(np.linalg.norm(pts - p_c[None, :], axis=1))
            return pts[idx], int(idx)
        v /= v_norm                                    # unit vector (3,)

        # 2. vectors from palm-centre to every cloud point
        d        = pts - p_c[None, :]                  # (N,3)

        # 3. signed distance *along the ray*  (dot product with v)
        s        = d @ v                               # (N,)

        # only keep points in *front* of the palm (optional but typical)
        mask     = s > 0
        if not np.any(mask):
            raise ValueError("No points in front of the palm centre")
            # fall back to nearest point in any direction
            idx = np.argmin(np.linalg.norm(d, axis=1))
            return pts[idx], idx

        d_fwd    = d[mask]
        s_fwd    = s[mask]                   # (M,1)

        # find the *minimum* positive projection
        s_min   = np.min(s_fwd)
        # allow tiny numeric wiggle room
        close   = np.abs(s_fwd - s_min) < 1e-6          # boolean mask (M,)

        if np.sum(close) == 1:
            idx_fwd = np.where(close)[0][0]

        else:
            # ----------   TIE-BREAK   ----------
            # perpendicular distance for the ties only
            d_perp2 = np.linalg.norm(d_fwd[close] - (s_min * v), axis=1)**2
            idx_local = np.argmin(d_perp2)
            idx_fwd   = np.flatnonzero(close)[idx_local]

        # map back to original cloud index
        idx_global = np.flatnonzero(mask)[idx_fwd]
        return pts[idx_global], idx_global

    def __getitem__(self, idx: int):
        gtype, obj_id, gpath, pose_idx, pc_path = self.index[idx]

        gdata = np.load(gpath, allow_pickle=True).item()
        scene_cfg = load_scene_cfg(gdata["scene_path"])

        # ---------- grasp pose & scale ------------------------------------
        robot_pose = np.stack([gdata["pregrasp_qpos"],
                               gdata["grasp_qpos"],
                               gdata["squeeze_qpos"]], axis=-2)[pose_idx]   # (3,J)
        scene_scale = gdata["scene_scale"][pose_idx]

        # ---------- object point cloud ------------------------------------
        mesh, pts_full = load_object(scene_cfg, n_sample=4096)
        pts_full *= scene_scale

        raw_pc = np.load(pc_path)                      # (V,3)
        idxs = np.random.choice(raw_pc.shape[0], 4096, replace=True)
        pc = raw_pc[idxs] * scene_scale

        # ---------- anchor -------------------------------------------------
        p_c, t_th, t_mid = self.fk_points_from_grasp_nofree(robot_pose[1])
        anchor_w, anchor_idx = self.anchor_point_from_hand(p_c, t_th, t_mid, pts_full)
        dists = np.linalg.norm(pc - anchor_w, axis=1)
        visible_anchor = dists.min() < 0.001  # 1 mm threshold
        # print(f"Anchor point {anchor_w} is visible: {visible_anchor}")
        if visible_anchor:
            # print('yes')
            pc[0] = anchor_w                                    # overwrite first point
        # else:
            # print('no')

        return {
            "partial_points": pc.astype(np.float16),
            "full_points"   : pts_full.astype(np.float32),
            "pregrasp_qpos" : robot_pose[0],       # etc.
            "grasp_qpos"    : robot_pose[1],
            "squeeze_qpos"  : robot_pose[2],
            "grasp_type_id" : int(gtype.split("_")[0]),
            "anchor_visible": visible_anchor,
            "obj_pose"      : scene_cfg["scene"][scene_cfg["task"]["obj_name"]]["pose"],
            "obj_scale"    : scene_scale * scene_cfg["scene"][scene_cfg["task"]["obj_name"]]["scale"][0],
            "obj_path"     : os.path.dirname(
                os.path.dirname(scene_cfg["scene"][scene_cfg["task"]["obj_name"]]["file_path"]))
        }

    # def __getitem__(self, id: int):
    #     ret_dict = {}

    #     if self.mode == "train" or self.mode == "eval":
    #         # random select grasp data
    #         rand_grasp_type = random.choice(self.grasp_type_lst)
    #         grasp_obj_lst = self.grasp_obj_dict[rand_grasp_type]
    #         rand_obj_id = random.choice(grasp_obj_lst)
    #         grasp_npy_lst = glob(
    #             pjoin(
    #                 self.config.grasp_path, rand_grasp_type, rand_obj_id, "**/**.npy"
    #             ),
    #             recursive=True,
    #         )
    #         grasp_path = random.choice(sorted(grasp_npy_lst))
    #         grasp_data = np.load(grasp_path, allow_pickle=True).item()
    #         # print(f"grasp_path: {grasp_path}")
    #         robot_pose = np.stack(
    #             [
    #                 grasp_data["pregrasp_qpos"],
    #                 grasp_data["grasp_qpos"],
    #                 grasp_data["squeeze_qpos"],
    #             ],
    #             axis=-2,
    #         )
    #         # if len(robot_pose.shape) == 3:
    #         rand_pose_id = np.random.randint(robot_pose.shape[0])
    #         robot_pose = robot_pose[rand_pose_id : rand_pose_id + 1]  # 1, 3, J
    #         # for rand_pose_id in range(robot_pose.shape[0]):
    #         #     robot_pose = robot_pose[rand_pose_id : rand_pose_id + 1]  # 1, 3, J

    #         scene_cfg = load_scene_cfg(grasp_data["scene_path"])
            
    #         mesh, pts = load_object(scene_cfg, n_sample=4096)
    #         target_obj = scene_cfg["task"]["obj_name"]
    #         ret_dict["obj_pose"] = scene_cfg["scene"][target_obj]["pose"]
    #         if not np.allclose(ret_dict["obj_pose"], [0, 0, 0, 1, 0, 0, 0]):
    #             raise NotImplementedError(
    #                 f"Object pose is not identity: {ret_dict['obj_pose']}"
    #             )
    #         obj_scale_in_scene = scene_cfg["scene"][target_obj]["scale"][0]
    #         # ret_dict["obj_scale"] = obj_scale_in_scene * grasp_data["scene_scale"][rand_pose_id]
    #         # pts = pts * ret_dict["obj_scale"]
    #         pts *= grasp_data["scene_scale"][rand_pose_id]
    #         ret_dict["full_points"] = pts
    #         ret_dict["obj_path"] = os.path.dirname(
    #                 os.path.dirname(scene_cfg["scene"][target_obj]["file_path"])
    #             )
    #         ret_dict["pregrasp_qpos"] = grasp_data["pregrasp_qpos"][rand_pose_id]  # (J)
    #         ret_dict["grasp_qpos"] = grasp_data["grasp_qpos"][rand_pose_id]
    #         ret_dict["squeeze_qpos"] = grasp_data["squeeze_qpos"][rand_pose_id]
            
    #         p_c, t_th, t_mid = self.fk_points_from_grasp_nofree(ret_dict["grasp_qpos"])
    #         # print("Palm centre:", p_c)
    #         # print("Thumb tip  :", t_th)
    #         # print("Middle tip :", t_mid)
            
    #         # ret_dict["palm_centre"] = p_c
    #         # ret_dict["thumb_tip"] = t_th
    #         # ret_dict["middle_tip"] = t_mid
            
    #         anchor_w, anchor_idx = self.anchor_point_from_hand(p_c, t_th, t_mid, ret_dict["full_points"])
            
    #         # temp_rot = numpy_quaternion_to_matrix(
    #         #     grasp_data["grasp_qpos"][rand_pose_id][3:7]
    #         # )  # (K, n, 3, 3)
            
            
    #         # anchor_w = anchor_point(
    #         #     pts,
    #         #     temp_rot,  # (3, 3)
    #         #     grasp_data["grasp_qpos"][rand_pose_id][:3],  # (3,)
    #         # )  # (3,)
            
    #         pc_path_lst = glob(
    #             pjoin(self.object_pc_folder, scene_cfg["scene_id"], "partial_pc**.npy")
    #         )
    #         pc_path = random.choice(sorted(pc_path_lst))
    #         raw_pc = np.load(pc_path, allow_pickle=True) # 4096, 643, 1000
    #         # print(raw_pc.shape, 'raw_pc shape')  # (N, 3)
    #         # print(pts.shape, 'pts shape')  # (N, 3)
    #         idx = np.random.choice(
    #             raw_pc.shape[0], self.config.num_points, replace=True # 1024
    #         )
    #         pc = raw_pc[idx]
    #         pc *= grasp_data["scene_scale"][rand_pose_id]
    #         # concat with anchor point
    #         pc = np.concatenate(
    #             [anchor_w[None, :], pc[1:]], axis=0
    #         )  # (N+1, 3)
    #         # print(pc.shape) (1024, 3)
            
    #         ret_dict["partial_points"] = pc


    #     elif self.mode == "test":
    #         rand_grasp_type = self.grasp_type_lst[id // len(self.test_cfg_lst)]
    #         scene_path = self.test_cfg_lst[id % len(self.test_cfg_lst)]
    #         scene_cfg = load_scene_cfg(scene_path)

    #         # read point cloud
    #         pc_path_lst = glob(
    #             pjoin(self.object_pc_folder, scene_cfg["scene_id"], "partial_pc**.npy"),
    #         )
    #         pc_path = random.choice(sorted(pc_path_lst))
    #         raw_pc = np.load(pc_path, allow_pickle=True)
    #         idx = np.random.choice(
    #             raw_pc.shape[0], self.config.num_points, replace=True
    #         )
    #         pc = raw_pc[idx]

    #         ret_dict["save_path"] = pjoin(
    #             rand_grasp_type, scene_cfg["scene_id"], os.path.basename(pc_path)
    #         )
    #         ret_dict["scene_path"] = scene_path

    #     # ret_dict["point_clouds"] = pc  # (N, 3)
    #     ret_dict["grasp_type_id"] = (
    #         int(rand_grasp_type.split("_")[0])
    #     )
    #     return ret_dict


