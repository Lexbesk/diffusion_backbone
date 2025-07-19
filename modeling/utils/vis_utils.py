# # utils/vis.py
# import os, tempfile, pathlib
# import numpy as np
# import mujoco
# import open3d as o3d
# import trimesh
# from matplotlib import cm
# import torch
# from mujoco.usd import exporter 

# # --------------------------------------------------------------------- #
# # Helper: build an Open3D mesh of the posed hand ---------------------- #
# # --------------------------------------------------------------------- #

# import numpy as np
# import open3d as o3d
# import mujoco
# from mujoco import mj_name2id, mjtObj
# from scipy.spatial.transform import Rotation as R
# from mujoco import mjtGeom, mj_id2name, mjtObj

# def _mesh_from_grasp(model: mujoco.MjModel,
#                      grasp: np.ndarray,
#                      color=(0.8, 0.8, 0.8),
#                      mesh_dir=None) -> o3d.geometry.TriangleMesh:
#     """
#     Constructs a TriangleMesh of the Shadow Hand in a given grasp pose,
#     using basic MuJoCo geom info (box, sphere, capsule).
#     """
#     data = mujoco.MjData(model)
#     data.qpos[:] = grasp[7:]     # finger joints only
#     mujoco.mj_forward(model, data)

#     full_mesh = o3d.geometry.TriangleMesh()

#     for geom_id in range(model.ngeom):
#         geom_type = model.geom_type[geom_id]
#         print(geom_type, 'geom type')
#         size = model.geom_size[geom_id]  # (3,)
#         pos  = data.geom_xpos[geom_id]   # (3,)
#         mat = data.geom_xmat[geom_id].reshape(3, 3)
#         # if geom_type == mjtGeom.mjGEOM_BOX:
#         #     mesh = o3d.geometry.TriangleMesh.create_box(
#         #         width=2*size[0], height=2*size[1], depth=2*size[2])
#         # elif geom_type == mjtGeom.mjGEOM_SPHERE:
#         #     mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size[0])
#         # elif geom_type == mjtGeom.mjGEOM_CAPSULE:
#         #     mesh = o3d.geometry.TriangleMesh.create_cylinder(
#         #         radius=size[0], height=2*size[1])
#         # elif geom_type == mjtGeom.mjGEOM_CYLINDER:
#         #     mesh = o3d.geometry.TriangleMesh.create_cylinder(
#         #         radius=size[0], height=2*size[1])
#         if geom_type == mjtGeom.mjGEOM_MESH:
#             mesh_id = model.geom_dataid[geom_id]
#             mesh_name = mj_id2name(model, mjtObj.mjOBJ_MESH, mesh_id)

#             if mesh_dir is None:
#                 print(f"[warn] mesh_dir not provided — cannot load mesh '{mesh_name}'")
#                 continue

#             for ext in [".stl", ".obj", ".ply"]:
#                 mesh_path = os.path.join(mesh_dir, mesh_name + ext)
#                 if os.path.exists(mesh_path):
#                     mesh = o3d.io.read_triangle_mesh(mesh_path)
#                     scale = model.mesh_scale[mesh_id]
#                     mesh.scale(scale=1.0, center=(0, 0, 0))  # ensures init scale = 1
#                     mesh.vertices = o3d.utility.Vector3dVector(
#                         np.asarray(mesh.vertices) * scale
#                     )
#                     break
#             else:
#                 print(f"[warn] mesh file for '{mesh_name}' not found in {mesh_dir}")
#                 continue
#         else:
#             print(f"[skip] unsupported geom type: {geom_type}")
#             continue

#         mesh.compute_vertex_normals()
#         mesh.paint_uniform_color(color)
#         mesh.rotate(mat, center=(0, 0, 0))
#         mesh.translate(pos)
#         full_mesh += mesh

#     # apply global grasp transform
#     xyz, quat = grasp[:3], grasp[3:7]
#     R_world = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
#     full_mesh.rotate(R_world, center=(0, 0, 0))
#     full_mesh.translate(xyz)

#     return full_mesh


# def _mesh_from_grasp1(model: mujoco.MjModel,
#                      grasp: np.ndarray,            # (29,)
#                      color=(0.8, 0.8, 0.8)):
#     """
#     Builds an Open3D mesh of the Shadow Hand at `grasp`.

#     If the MJCF has *no* free joint, we:
#       • set only the 22 finger angles in qpos
#       • export the mesh
#       • apply the global (xyz + quaternion) transform afterwards
#     """
#     # ------------------------------------------------------------------
#     # 1) Forward-kinematics for finger joints only
#     # ------------------------------------------------------------------
#     data = mujoco.MjData(model)
#     finger_angles = grasp[7:]                # len 22
#     if finger_angles.shape[0] != model.nq:
#         raise ValueError(
#             f"Model has {model.nq} qpos DOFs, "
#             f"but grasp specifies {finger_angles.shape[0]} finger angles."
#         )
#     data.qpos[:] = finger_angles
#     mujoco.mj_forward(model, data)

#     # ------------------------------------------------------------------
#     # 2) Export posed mesh to GLB
#     # ------------------------------------------------------------------
#     with tempfile.TemporaryDirectory() as tmpdir:
#         tmpdir = 'tmp'
#         os.makedirs(tmpdir, exist_ok=True)
#         print(f"Exporting to temporary directory: {tmpdir}")
#         exp = exporter.USDExporter(
#             model=model,
#             max_geom=model.ngeom,
#             output_directory_root=tmpdir,
#             verbose=True,
#         )
#         exp.update_scene(data=data)
#         usd_path = exp.save_scene(filetype="usd")  # or "usda"

#         # Trimesh or Open3D can load USD
#         # usd_path = exp.save_scene(filetype="usdc")
#         usd_path = '/data/user_data/austinz/Robots/manipulation/analogical_manipulation/tmp/mujoco_usdpkg/frames/frame_1.usdc'
#         if not usd_path or not os.path.exists(usd_path):
#             raise RuntimeError("USDExporter failed to export a valid file.")
#         # mesh = o3d.io.read_triangle_mesh(usd_path)
#         trimesh_mesh = trimesh.load(usd_path, force='mesh')
#         mesh = o3d.geometry.TriangleMesh(
#             vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
#             triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces)
#         )

#     # ------------------------------------------------------------------
#     # 3) Apply global pose (translate + rotate)
#     # ------------------------------------------------------------------
#     xyz  = grasp[:3]
#     quat = grasp[3:7]                  # [w, x, y, z]
#     # Open3D expects (w, x, y, z); ok as-is
#     R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
#     mesh.rotate(R, center=(0, 0, 0))
#     mesh.translate(xyz)

#     mesh.paint_uniform_color(color)
#     return mesh


# # --------------------------------------------------------------------- #
# # Main API: render PNG (+ opt. HTML) ---------------------------------- #
# # --------------------------------------------------------------------- #
# def render_grasps(xml_path: str,
#                   pcd: np.ndarray,           # (N,3)
#                   grasps: np.ndarray,        # (M,29) 1≤M≤11
#                   png_out: os.PathLike,
#                   html_out: os.PathLike | None = None,
#                   img_size=(1280 // 2, 960 // 2),
#                   mesh_dir=None):
#     """
#     Produce `png_out` (always) and `html_out` (if not None).

#     Example
#     -------
#     render_grasps("assets/shadow_hand.xml",
#                   pcd_xyz, poses[ :11], "iter020.png", "iter020.html")
#     """
#     assert grasps.ndim == 2 and grasps.shape[1] == 29
#     model = mujoco.MjModel.from_xml_path(xml_path)  # ~2–3 ms, do once

#     # 1) build geometry list
#     cmap = cm.get_cmap('viridis', len(grasps))
#     geoms = [
#         _mesh_from_grasp(model, g, cmap(i)[:3], mesh_dir=mesh_dir)
#         for i, g in enumerate(grasps)
#     ]
#     # point cloud
#     pcd_o3d = o3d.geometry.PointCloud(
#         o3d.utility.Vector3dVector(pcd.astype(np.float32)))
#     pcd_o3d.paint_uniform_color([1.0, 0.2, 0.2])
#     geoms.append(pcd_o3d)

#     # 2) off-screen raster → PNG
#     r = o3d.visualization.rendering.OffscreenRenderer(
#         img_size[0], img_size[1])
#     mat = o3d.visualization.rendering.MaterialRecord()
#     mat.shader = "defaultLit"

#     for j, g in enumerate(geoms):
#         r.scene.add_geometry(f"g{j}", g, mat)

#     # simple orbit camera
#     bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
#         o3d.utility.Vector3dVector(pcd))
#     center = bbox.get_center()
#     eye    = center + np.array([0.30, 0.35, 0.25])   # tweak if needed
#     r.scene.camera.look_at(center, eye, [0, 0, 1])

#     img = r.render_to_image()
#     o3d.io.write_image(str(png_out), img)
#     # r.release()

#     # 3) optional HTML
#     if html_out:
#         scene = trimesh.Scene()
#         for k, g in enumerate(geoms):
#             if isinstance(g, o3d.geometry.PointCloud):
#                 scene.add_geometry(trimesh.PointCloud(np.asarray(g.points)),
#                                    node_name=f"pcd{k}")
#             else:
#                 scene.add_geometry(
#                     trimesh.Trimesh(vertices=np.asarray(g.vertices),
#                                     faces=np.asarray(g.triangles)),
#                     node_name=f"hand{k}")
#         scene.export(html_out)   # embeds three.js viewer

import numpy as np, tempfile, textwrap, mujoco, pathlib
from lxml import etree   # pip install lxml
import torch
import copy

def xml_with_pcd_and_hands(base_hand_xml: str,
                           pcd: np.ndarray,         # (N,3)
                           n_hands: int,
                           pcd_radius: float = 0.005,
                           asset_dir=None) -> pathlib.Path:
    """
    Generates a temporary XML scene:
      • <asset> : a mesh called "pcd_mesh"
      • <worldbody> : a single geom that displays the mesh
      •             : n_hands bodies, each a copy of the ShadowHand subtree
    Returns path to the XML on disk.
    """
    # 1) Build <mesh> OBJ for point cloud  -------------
    obj_path = tempfile.NamedTemporaryFile(delete=False,
                                           suffix=".obj").name
    with open(obj_path, "w") as f:
        f.write("# generated OBJ\n")
        for v in pcd:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # ---- NEW: non-zero-area face --------------------------------- #
        # Use the first three vertices, which are guaranteed to differ.
        # If the cloud is extremely small (<1 mm), multiply by 100 here
        # or add an explicit 'scale="100 100 100"' attribute on <mesh>.
        f.write("f 1 2 3\n")

    # 2) Parse the base hand file and extract the <body> --------------------
    hand_tree = etree.parse(base_hand_xml)
    base_root  = hand_tree.getroot()
    hand_body  = base_root.find(".//worldbody/body")
    # hand_body = hand_tree.find(".//worldbody/body")
    # if hand_body is None:
    #     raise RuntimeError("Couldn’t locate top <body> of the hand.")

    # 3) Start a fresh XML ---------------------------------------------------
    root = etree.Element("mujoco", model="pcd+hands")
    compiler = etree.SubElement(root, "compiler", angle="degree", coordinate="local", meshdir=asset_dir, texturedir=asset_dir)
    option   = etree.SubElement(root, "option", gravity="0 0 0")

    
    for blk_name in ("default", "asset"):
        blk = base_root.find(blk_name)
        if blk is not None:
            root.append(copy.deepcopy(blk))
    asset = root.find("asset")
    if asset is None:
        asset = etree.SubElement(root, "asset")
        
    # asset    = etree.SubElement(root, "asset")
    world    = etree.SubElement(root, "worldbody")

    # # 3a) add point-cloud mesh asset
    # mesh = etree.SubElement(asset, "mesh", name="pcd_mesh", file=obj_path)
    # # 3b) geom that shows it
    # etree.SubElement(world, "geom",
    #                  type="mesh", mesh="pcd_mesh",
    #                  rgba="0 0.6 1 1",
    #                  contype="0", conaffinity="0",
    #                  # MuJoCo scales mesh by “size”; we’ll use it as a radius for GL_POINTS style
    #                  size=f"{pcd_radius}")

    # 3c) add N copies of the hand
    for k in range(n_hands):
        new_body = etree.fromstring(etree.tostring(hand_body))
        new_body.set("name", f"hand{k}")
        # Give each body its own free-joint so we can set pos+quat independently
        free = etree.SubElement(new_body, "joint", type="free", name=f"hand{k}_root")
        world.append(new_body)

    # 4) write XML
    tmp_dir  = pathlib.Path("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_xml  = tmp_dir / "tmphands.xml"
    print(f"Writing temporary XML to {tmp_xml}")
    tmp_xml.write_text(etree.tostring(root, pretty_print=True, encoding="unicode"))
    print(tmp_xml)
    return tmp_xml


import mujoco, imageio
import numpy as np, os


if __name__ == "__main__":
    # Example usage
    xml_path = "/data/user_data/austinz/Robots/DexGraspBench/assets/hand/shadow/right_hand.xml"
    input_pt_path = '/data/user_data/austinz/Robots/manipulation/analogical_manipulation/train_logs/DEXONOMY_7k_pcdcentric/run_Jul13_grasp_denoiser-Dexonomy-lr1e-4-constant-rectified_flow-B32-Bval8-DT10/batch_inspect.pt'
    mesh_dir = '/data/user_data/austinz/Robots/DexGraspBench/third_party/mujoco_menagerie/shadow_hand/assets'
    grasp_data0 = torch.load(input_pt_path, map_location="cpu")
    
    # for ip in iterable_params:
    num = -1
    for i in range(grasp_data0['grasp_qpos'].shape[0]):
        grasp_data = {}
        num += 1
        for key, val in grasp_data0.items():
            if torch.is_tensor(val):                       # case 1
                item = val[i]                 # (B, …) → (…)
            elif isinstance(val, (list, tuple)) and val:   # case 2
                item = val[i]                 # list/tuple → first element
            else:                                          # metadata or already single
                item = val
            
            if torch.is_tensor(item):
                item = item.detach().cpu().numpy()         # safe even if already on CPU

            grasp_data[key] = item
        pcd = grasp_data['partial_points']
        print(pcd.shape, 'pcd shape')
        grasps = grasp_data['grasp_qpos'][None]
        M      = len(grasps)
        xml_path = xml_with_pcd_and_hands(xml_path, pcd, n_hands=M, asset_dir=mesh_dir)
        model    = mujoco.MjModel.from_xml_path(str(xml_path))
        data     = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 360, 480)     # off-screen
        # ---------- helper: index map ---------------------------------------------
        root_qpos_addr = [model.joint(name).qposadr for name in
                        [f"hand{i}_root" for i in range(M)]]
        # print(root_qpos_addr, 'root qpos addr') 
        # ---------- drop every grasp pose -----------------------------------------
        for i, g in enumerate(grasps):
            pos   = g[:3]                         # (x,y,z)
            quatW = g[3:7]                        # (w,x,y,z) incoming
            quatX = [quatW[1], quatW[2], quatW[3], quatW[0]]  # MuJoCo is (x,y,z,w)

            # free-joint layout: [x y z qw qx qy qz]
            qpos_block = np.array([*pos, *quatX], dtype=np.float64)
            idx        = int(root_qpos_addr[i])
            data.qpos[idx:idx+7] = qpos_block

            # joint angles start at index idx+7 (order exactly as in XML)
            data.qpos[idx+7 : idx+7+22] = g[7:]

        # ---------- render ---------------------------------------------------------
        mujoco.mj_forward(model, data)
        img = renderer.render()
        imageio.imwrite("tmp/all_grasps.png", img[..., ::-1])
        print("wrote all_grasps.png – check the result!")

