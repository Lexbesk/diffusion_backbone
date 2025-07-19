import torch
from torch import Tensor


class DexonomyDataPreprocessor:
    """
    Geometry-consistent point-cloud + grasp-pose augmentation, torch version.

    Expected batch dict  (shapes may have a leading batch dim):
        'partial_points' :  (B, N, 3) or (N, 3)
        'pregrasp_qpos'  :  (B, 29)   or (29)
        'grasp_qpos'     :  (B, 29)
        'squeeze_qpos'   :  (B, 29)
    """

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def quat_normalize(q: Tensor) -> Tensor:
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def quat_to_matrix(q: Tensor) -> Tensor:
        """
        Convert w-x-y-z quaternion(s) to 3×3 rotation matrix/matrices.
        Vectorised:  q.shape == (..., 4)  →  (..., 3, 3)
        """
        q = DexonomyDataPreprocessor.quat_normalize(q)
        w, x, y, z = q.unbind(-1)

        two = 2.0
        R = torch.stack((
            1 - two * (y * y + z * z),  two * (x * y - w * z),         two * (x * z + w * y),
            two * (x * y + w * z),      1 - two * (x * x + z * z),     two * (y * z - w * x),
            two * (x * z - w * y),      two * (y * z + w * x),         1 - two * (x * x + y * y),
        ), dim=-1)

        return R.view(q.shape[:-1] + (3, 3))

    @staticmethod
    def quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
        """
        Hamilton product q = q1 * q2 (all quats in w-x-y-z ordering).
        Shapes must be broadcast-compatible.
        """
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=-1)

    @staticmethod
    def random_quaternion(batch: int, device=None, dtype=None) -> Tensor:
        """
        Uniform SO(3) sampling   (Müller, 1959).
        Returns tensor (batch, 4)  in w-x-y-z order, unit-norm.
        """
        u1, u2, u3 = torch.rand(3, batch, device=device, dtype=dtype)
        q = torch.stack((
            torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2),
            torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2),
            torch.sqrt(u1)     * torch.sin(2 * torch.pi * u3),
            torch.sqrt(u1)     * torch.cos(2 * torch.pi * u3),
        ), dim=-1)
        return DexonomyDataPreprocessor.quat_normalize(q)

    # ------------------------------------------------------------------- public
    def wild_parallel_augment(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # print(batch['partial_points'].shape, 'partial_points shape')  # Debugging line
        # print(batch['grasp_type_id'].shape, 'grasp_type_id shape')  # Debugging line
        # print(batch['anchor_visible'].shape, 'anchor_visible shape')  # Debugging line
        # print(batch['obj_pose'].shape, 'obj_pose shape')  # Debugging line
        # print(batch['obj_scale'].shape, 'obj_scale shape')  # Debugging line
        # print(batch['obj_path'], 'obj_path shape')  # Debugging line
        # print(batch['pregrasp_qpos'].shape, 'pregrasp_qpos shape')  # Debugging line
        # print(batch['grasp_qpos'].shape, 'grasp_qpos shape')
        # print(batch['squeeze_qpos'].shape, 'squeeze_qpos shape')
        # print('Batch keys:', batch.keys())  # Debugging line
        """
        Translates each sample so the point-cloud centroid becomes the origin,
        then applies a shared random rotation to the cloud **and** the three
        grasp poses  preserving their relative geometry.

        Returns a new dict (original tensors remain untouched).
        """
        pts: Tensor = batch["partial_points"]            # (B,N,3) or (N,3)
        batched = pts.dim() == 3
        B = pts.size(0) if batched else 1
        device, dtype = pts.device, pts.dtype

        # ---------- 1. translate to centroid ----------
        center = pts.mean(dim=-2, keepdim=True)          # (B,1,3) or (1,3)
        pts_centered = pts - center

        # ---------- 2. sample & apply random rotation ----------
        rand_q = self.random_quaternion(B, device, dtype)          # (B,4)
        R = self.quat_to_matrix(rand_q)                            # (B,3,3)

        if batched:
            pts_rot = torch.einsum("bij,bnj->bni", R, pts_centered)
        else:
            pts_rot = (R @ pts_centered.T).T.squeeze(0)

        out = {"partial_points": pts_rot}
        out["grasp_type_id"] = batch["grasp_type_id"] # (B,) or None
        out["anchor_visible"] = batch["anchor_visible"]  # (B

        # helper: transform one qpos or a batch thereof
        def transform_qpos(qpos: Tensor, idx: int | slice) -> Tensor:
            # -------- translate ----------
            if batched:
                pos = qpos[..., :3] - center[idx, 0]     # (3,)
                pos = R[idx] @ pos                       # rotate
            else:
                pos = qpos[..., :3] - center[0]          # (3,)
                pos = R @ pos

            # -------- rotate quaternion ----------
            quat = qpos[..., 3:7]
            new_quat = self.quat_mul(rand_q[idx], quat)
            new_quat = self.quat_normalize(new_quat)

            # -------- concat ----------
            return torch.cat((pos, new_quat, qpos[..., 7:]), dim=-1)

        for key in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos"):
            q = batch[key]
            if batched:
                transformed = torch.stack(
                    [transform_qpos(q[i], i) for i in range(B)], dim=0
                )
            else:
                transformed = transform_qpos(q, 0)
            out[key] = transformed
            
        if "obj_pose" in batch:
            obj_pose = batch["obj_pose"]          # (B,7)  (tx,ty,tz, qw qx qy qz)
            obj_pose = obj_pose.to('cuda').float()
            pos_centered = obj_pose[:, :3] - center.squeeze(-2)          # (B,3)

            # rotate it with the same R
            pos_rot = torch.einsum("bij,bj->bi", R, pos_centered)        # (B,3)

            # ---------- 2. rotate quaternion ----------
            quat_orig = obj_pose[:, 3:7]                                 # (B,4)
            quat_new  = self.quat_mul(rand_q, quat_orig)                 # (B,4)
            quat_new  = self.quat_normalize(quat_new)

            # ---------- 3. concat ----------
            obj_pose_new = torch.cat((pos_rot, quat_new), dim=-1)        # (B,7)

            out["obj_pose"] = obj_pose_new
            out["obj_path"] = batch["obj_path"]  # pass through unchanged
            out["obj_scale"] = batch["obj_scale"]  # pass through unchanged

        return out
    
    def translate_to_center_frame(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Rigidly translate the whole sample so that the centroid of the partial
        point-cloud is moved to the origin **without applying any rotation**.
        The grasp poses therefore keep their original orientation – only the
        positional part of each q-pos is shifted.

        Parameters
        ----------
        batch : dict
            Must contain at least the keys
            - "partial_points" : (B, N, 3) or (N, 3)
            - "pregrasp_qpos"  : (B, D)   or (D,)
            - "grasp_qpos"     : (B, D)   or (D,)
            - "squeeze_qpos"   : (B, D)   or (D,)
            plus any meta-fields you want to copy through unchanged.

        Returns
        -------
        dict
            Same structure as *batch* with every geometry tensor expressed
            in the object-centered frame (no rotation applied).
        """
        pts = batch["partial_points"]                 # (B,N,3) or (N,3)
        batched = pts.dim() == 3
        B = pts.size(0) if batched else 1

        # ---- 1. translate point cloud so centroid → origin --------------------
        center = pts.mean(dim=-2, keepdim=True)       # (B,1,3) or (1,3)
        pts_centered = pts - center

        out: dict[str, torch.Tensor] = {"partial_points": pts_centered}

        # pass through meta information that is unaffected by the transform
        for k in ("grasp_type_id", "anchor_visible"):
            if k in batch:
                out[k] = batch[k]

        # ---- 2. helper to translate a single q-pos (no rotation) -------------
        def translate_qpos(qpos: torch.Tensor, idx: int | slice) -> torch.Tensor:
            if batched:
                pos = qpos[..., :3] - center[idx, 0]   # subtract that sample’s centroid
            else:
                pos = qpos[..., :3] - center[0]
            # orientation and joint angles stay exactly the same
            return torch.cat((pos, qpos[..., 3:]), dim=-1)

        # ---- 3. translate every grasp pose -----------------------------------
        for key in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos"):
            q = batch[key]
            if batched:
                translated = torch.stack(
                    [translate_qpos(q[i], i) for i in range(B)], dim=0
                )
            else:
                translated = translate_qpos(q, 0)
            out[key] = translated

        return out
