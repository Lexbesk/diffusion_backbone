import torch
import torch.nn as nn
from typing import Dict, List, Optional
import pytorch_kinematics as pk  # pip install pytorch-kinematics

class FKLayer(nn.Module):
    """
    Fast, differentiable FK from a URDF using pytorch-kinematics.
    - Works with q shaped (..., J) where J == len(joint_names)
    - Returns positions/rotations for a set of link frames
    - Optional fixed probe points on links (local-frame offsets)
    """
    def __init__(
        self,
        urdf_path: str,
        joint_names: List[str],
        out_links: List[str],
        probe_points_local: Optional[Dict[str, torch.Tensor]] = None,  # e.g. {"rh_fftip": (K,3) local offsets}
        device: Optional[torch.device] = None,
        compile_torch: bool = False,   # set True if on torch>=2.0 for extra speed
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Build chain (parses URDF, orders joints as in file)
        if hasattr(pk, "build_chain_from_urdf_path"):
            self.chain = pk.build_chain_from_urdf_path(urdf_path).to(self.device)
        else:
            with open(urdf_path, "r") as f:
                urdf_xml = f.read()
            self.chain = pk.build_chain_from_urdf(urdf_xml).to(device=self.device)

        # Map your model's joint order -> chain's internal joint order
        chain_joint_names = self.chain.get_joint_parameter_names()  # list of names
        name_to_idx = {n: i for i, n in enumerate(chain_joint_names)}
        try:
            self.reorder_idx = torch.tensor([name_to_idx[n] for n in joint_names], device=self.device, dtype=torch.long)
        except KeyError as e:
            missing = str(e).strip("'")
            raise ValueError(
                f"Joint '{missing}' not found in URDF. "
                f"Available joints include: {chain_joint_names[:10]} ... (total {len(chain_joint_names)})"
            )

        # Verify output links exist; if not, print all links for quick fix
        link_names = set(self.chain.get_link_names())
        missing_links = [L for L in out_links if L not in link_names]
        if len(missing_links) > 0:
            raise ValueError(
                f"These out_links are not in URDF: {missing_links}\n"
                f"Try one of: {list(link_names)[:12]} ... (total {len(link_names)})"
            )
        self.out_links = out_links

        # Optional probe points (local-frame points to transform)
        self.has_probes = probe_points_local is not None and len(probe_points_local) > 0
        if self.has_probes:
            # Store as buffers so they move with the module/device and are saved in state_dict
            self.register_buffer("_one", torch.tensor(1.0))  # just to have a device/dtype anchor
            self.probe_points_local = {k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
                                       for k, v in probe_points_local.items()}

        # Optional compile for speed (PyTorch 2.0+)
        self._compiled = None
        if compile_torch:
            try:
                self._compiled = torch.compile(self._forward_impl)
            except Exception:
                self._compiled = None  # fallback silently

    @torch.no_grad()
    def print_summary(self):
        print("URDF joints:", self.chain.get_joint_parameter_names())
        print("URDF links:", self.chain.get_link_names())

    def forward(self, q: torch.Tensor, need_rot: bool = False):
        """
        q: (..., J) in the order of `joint_names` passed at init.
        Returns:
          pos_dict: {link: (..., 3)} world (base) positions of link frames
          rot_dict: {link: (..., 3,3)} rotation matrices (if need_rot)
          probe_dict (optional): {link: (..., K, 3)} world coords of local probe points per link
        """
        if self._compiled is not None:
            return self._compiled(q, need_rot)
        else:
            return self._forward_impl(q, need_rot)

    def _forward_impl(self, q: torch.Tensor, need_rot: bool = False):
        orig_shape = q.shape[:-1]     # e.g., (B, T)
        J = q.shape[-1]
        assert J == self.reorder_idx.numel(), f"q has {J} dofs but expected {self.reorder_idx.numel()}"

        # Reorder q to match the chain's joint order and flatten batch/time
        q_chain = q[..., torch.arange(J, device=q.device)]
        q_chain = q_chain.index_select(-1, self.reorder_idx)
        q_flat = q_chain.reshape(-1, J)

        # Forward kinematics for all links (batched)
        # returns dict: {link_name: Transform3d} with batch dimension = q_flat.shape[0]
        fk_all = self.chain.forward_kinematics(q_flat)

        # Gather outputs
        pos_dict, rot_dict, probe_dict = {}, {}, {}
        for L in self.out_links:
            T = fk_all[L]  # Transform3d

            # --- Version-agnostic split of R, t ---
            if hasattr(T, "get_matrix"):
                M = T.get_matrix()            # (..., 4, 4)
            elif hasattr(T, "matrix"):
                M = T.matrix                   # (..., 4, 4)
            else:
                raise RuntimeError("Transform3d has no get_matrix()/matrix")

            R = M[..., :3, :3]                # (..., 3, 3)
            t = M[..., :3, 3]                 # (..., 3)

            pos_dict[L] = t.reshape(*orig_shape, 3)

            if need_rot:
                rot_dict[L] = R.reshape(*orig_shape, 3, 3)

            # Optional probe points in local link frame -> world
            if self.has_probes and L in self.probe_points_local:
                P_local = self.probe_points_local[L]         # (K,3)
                N = t.shape[0]
                P_local_b = P_local.unsqueeze(0).expand(N, -1, -1)  # (N,K,3)

                # Prefer library method if available; fallback to manual transform
                if hasattr(T, "transform_points"):
                    Xw = T.transform_points(P_local_b)       # (N,K,3)
                else:
                    Xw = torch.matmul(P_local_b, R.transpose(-1, -2)) + t.unsqueeze(1)

                probe_dict[L] = Xw.reshape(*orig_shape, P_local.shape[0], 3)

        return (pos_dict,
                rot_dict if need_rot else None,
                probe_dict if self.has_probes else None)
