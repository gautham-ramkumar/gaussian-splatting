"""3D Gaussian Splatting model."""
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def _build_rotation(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (N,4) → rotation matrix (N,3,3)."""
    qr, qi, qj, qk = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1-2*(qj*qj+qk*qk),   2*(qi*qj-qr*qk),    2*(qi*qk+qr*qj),
          2*(qi*qj+qr*qk), 1-2*(qi*qi+qk*qk),    2*(qj*qk-qr*qi),
          2*(qi*qk-qr*qj),   2*(qj*qk+qr*qi),  1-2*(qi*qi+qj*qj),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _strip_symmetric(L: torch.Tensor) -> torch.Tensor:
    """Return upper-triangle of Σ = L @ L.T as (N,6)."""
    # L is (N,3,3)
    return torch.stack([
        L[:, 0, 0], L[:, 0, 1], L[:, 0, 2],
        L[:, 1, 1], L[:, 1, 2], L[:, 2, 2],
    ], dim=1)


def _build_L(scale: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Σ = R @ S @ S.T @ R.T as upper-triangle."""
    R = _build_rotation(rot)                          # (N,3,3)
    S = torch.diag_embed(scale)                       # (N,3,3)
    L = R @ S                                         # (N,3,3)
    return _strip_symmetric(L)


class GaussianModel(nn.Module):
    """
    Learnable 3D Gaussian parameters representing a radiance field.
    All parameter tensors live on CUDA.
    """

    def __init__(self, sh_degree: int = 3):
        super().__init__()
        self.max_sh_degree  = sh_degree
        self.active_sh_degree = 0

        # Learnable parameters (initialised by create_from_pcd)
        self._xyz           = nn.Parameter(torch.empty(0, 3))
        self._features_dc   = nn.Parameter(torch.empty(0, 1,  3))
        self._features_rest = nn.Parameter(torch.empty(0, 15, 3))
        self._scaling       = nn.Parameter(torch.empty(0, 3))
        self._rotation      = nn.Parameter(torch.empty(0, 4))
        self._opacity       = nn.Parameter(torch.empty(0, 1))

        # Gradient accumulators for densification (not parameters)
        self._xyz_grad_accum: Optional[torch.Tensor] = None
        self._denom:          Optional[torch.Tensor] = None
        self._max_radii:      Optional[torch.Tensor] = None

        self.optimizer: Optional[torch.optim.Optimizer] = None

    # ── Properties ───────────────────────────────────────────
    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_covariance(self) -> torch.Tensor:
        return _build_L(self.get_scaling, self.get_rotation)

    # ── Initialisation ───────────────────────────────────────
    def create_from_pcd(self, xyzs: np.ndarray, rgbs: np.ndarray,
                        scene_radius: float = 1.0):
        """Initialise Gaussians from a sparse COLMAP point cloud."""
        n = xyzs.shape[0]
        print(f"[Gaussian] Initialising {n} splats from point cloud")

        points = torch.tensor(xyzs, dtype=torch.float32).cuda()

        # SH DC init: convert sRGB → SH coefficient  (f = (rgb - 0.5) / C0)
        from src.rendering.sh_utils import C0
        rgb_t = torch.tensor(rgbs / 255.0, dtype=torch.float32).cuda()
        features = torch.zeros(n, 16, 3, dtype=torch.float32).cuda()
        features[:, 0, :] = (rgb_t - 0.5) / C0       # DC component

        # Scale init: mean nearest-neighbour distance (log space)
        from simple_knn._C import distCUDA2
        dists = torch.clamp_min(distCUDA2(points), 1e-7)
        scales = torch.log(torch.sqrt(dists))[..., None].repeat(1, 3)

        rots    = torch.zeros(n, 4).cuda(); rots[:, 0] = 1.0   # identity quaternion
        opacs   = _inverse_sigmoid(0.1 * torch.ones(n, 1).cuda())

        self._xyz           = nn.Parameter(points)
        self._features_dc   = nn.Parameter(features[:, :1,  :])
        self._features_rest = nn.Parameter(features[:, 1:,  :])
        self._scaling       = nn.Parameter(scales)
        self._rotation      = nn.Parameter(rots)
        self._opacity       = nn.Parameter(opacs)

        self._xyz_grad_accum = torch.zeros(n, 1, device="cuda")
        self._denom          = torch.zeros(n, 1, device="cuda")
        self._max_radii      = torch.zeros(n,    device="cuda")

    # ── Optimizer setup ───────────────────────────────────────
    def setup_optimizer(self, position_lr_init=1.6e-4, position_lr_final=1.6e-6,
                        feature_lr=2.5e-3, opacity_lr=5e-2,
                        scaling_lr=5e-3, rotation_lr=1e-3):
        self._pos_lr_init  = position_lr_init
        self._pos_lr_final = position_lr_final

        params = [
            {"params": [self._xyz],           "lr": position_lr_init, "name": "xyz"},
            {"params": [self._features_dc],   "lr": feature_lr,       "name": "f_dc"},
            {"params": [self._features_rest], "lr": feature_lr / 20,  "name": "f_rest"},
            {"params": [self._opacity],       "lr": opacity_lr,       "name": "opacity"},
            {"params": [self._scaling],       "lr": scaling_lr,       "name": "scaling"},
            {"params": [self._rotation],      "lr": rotation_lr,      "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

    def update_position_lr(self, step: int, max_steps: int):
        """Exponential decay for position LR only."""
        t = step / max_steps
        lr = self._pos_lr_final + (self._pos_lr_init - self._pos_lr_final) * math.exp(-t * 7)
        for g in self.optimizer.param_groups:
            if g["name"] == "xyz":
                g["lr"] = lr

    # ── Densification ─────────────────────────────────────────
    def update_stats(self, viewspace_pts: torch.Tensor, radii: torch.Tensor, visible: torch.Tensor):
        """Accumulate 2D gradient norms for densification."""
        self._xyz_grad_accum[visible] += torch.norm(
            viewspace_pts.grad[visible, :2], dim=-1, keepdim=True)
        self._denom[visible] += 1
        self._max_radii[visible] = torch.max(
            self._max_radii[visible], radii[visible].float())

    def densify_and_prune(self, grad_threshold=2e-4, min_opacity=5e-3,
                          scene_extent=1.0, max_gaussians=500_000,
                          percent_dense=0.01):
        """Clone small high-gradient splats, split large ones, prune transparent ones."""
        grads = self._xyz_grad_accum / (self._denom + 1e-7)
        grads[grads.isnan()] = 0.0

        # Compute both masks BEFORE any mutations (based on original N Gaussians)
        high_grad   = (grads > grad_threshold).squeeze()
        large_scale = torch.max(self.get_scaling, dim=1).values > percent_dense * scene_extent

        # ──── Clone (small Gaussians with high gradient) ──────
        clone_mask = high_grad & ~large_scale
        self._clone(clone_mask)

        # ──── Split (large Gaussians with high gradient) ──────
        split_mask = high_grad & large_scale
        # Pad split_mask with False for the Gaussians appended by _clone
        n_new = self.num_gaussians - split_mask.shape[0]
        if n_new > 0:
            split_mask = torch.cat([split_mask,
                                    torch.zeros(n_new, dtype=torch.bool, device="cuda")])
        if split_mask.any():
            self._split(split_mask)

        # ──── Prune ───────────────────────────────────────────
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_gaussians > 0 and self.num_gaussians > max_gaussians:
            # Prune by lowest opacity until under limit
            extra = self.num_gaussians - max_gaussians
            topk_low = torch.topk(self.get_opacity.squeeze(), extra, largest=False).indices
            prune_mask[topk_low] = True
        self._prune(prune_mask)

        # Reset accumulators
        n = self.num_gaussians
        self._xyz_grad_accum = torch.zeros(n, 1, device="cuda")
        self._denom          = torch.zeros(n, 1, device="cuda")
        self._max_radii      = torch.zeros(n,    device="cuda")

    def _clone(self, mask: torch.Tensor):
        if not mask.any():
            return
        new_xyz   = self._xyz[mask].detach()
        new_fdc   = self._features_dc[mask].detach()
        new_frest = self._features_rest[mask].detach()
        new_scale = self._scaling[mask].detach()
        new_rot   = self._rotation[mask].detach()
        new_opa   = self._opacity[mask].detach()
        self._cat_tensors(new_xyz, new_fdc, new_frest, new_scale, new_rot, new_opa)

    def _split(self, mask: torch.Tensor, n_split: int = 2):
        if not mask.any():
            return
        scale  = self.get_scaling[mask]
        rot    = self.get_rotation[mask]
        # Sample points around original Gaussian
        samples = torch.randn(mask.sum() * n_split, 3, device="cuda")
        R = _build_rotation(rot.repeat(n_split, 1))
        stds = scale.repeat(n_split, 1)
        means = torch.bmm(R, (samples * stds).unsqueeze(-1)).squeeze(-1)
        new_xyz   = self._xyz[mask].repeat(n_split, 1) + means
        new_scale = torch.log(scale.repeat(n_split, 1) / (0.8 * n_split))
        new_rot   = self._rotation[mask].repeat(n_split, 1)
        new_fdc   = self._features_dc[mask].repeat(n_split, 1, 1)
        new_frest = self._features_rest[mask].repeat(n_split, 1, 1)
        new_opa   = self._opacity[mask].repeat(n_split, 1)
        # Remove originals via _prune (handles optimizer state correctly)
        self._prune(mask)
        self._cat_tensors(new_xyz, new_fdc, new_frest, new_scale, new_rot, new_opa)

    def _prune(self, mask: torch.Tensor):
        keep = ~mask
        # Save old param refs before reassignment so we can migrate optimizer state
        if self.optimizer is not None:
            old_refs = {g["name"]: g["params"][0] for g in self.optimizer.param_groups}
        self._xyz           = nn.Parameter(self._xyz[keep])
        self._features_dc   = nn.Parameter(self._features_dc[keep])
        self._features_rest = nn.Parameter(self._features_rest[keep])
        self._scaling       = nn.Parameter(self._scaling[keep])
        self._rotation      = nn.Parameter(self._rotation[keep])
        self._opacity       = nn.Parameter(self._opacity[keep])
        if self.optimizer is not None:
            new_refs = {
                "xyz": self._xyz, "f_dc": self._features_dc, "f_rest": self._features_rest,
                "opacity": self._opacity, "scaling": self._scaling, "rotation": self._rotation,
            }
            for g in self.optimizer.param_groups:
                name  = g["name"]
                old_p = old_refs[name]
                new_p = new_refs[name]
                if old_p in self.optimizer.state and "exp_avg" in self.optimizer.state[old_p]:
                    st = self.optimizer.state.pop(old_p)
                    self.optimizer.state[new_p] = {
                        "step":        st["step"],
                        "exp_avg":     st["exp_avg"][keep],
                        "exp_avg_sq":  st["exp_avg_sq"][keep],
                    }
                g["params"][0] = new_p

    def _cat_tensors(self, xyz, fdc, frest, scale, rot, opa):
        """Concatenate new tensors and update optimiser state."""
        n = xyz.shape[0]
        # Save old param refs before reassignment
        if self.optimizer is not None:
            old_refs = {g["name"]: g["params"][0] for g in self.optimizer.param_groups}
        with torch.no_grad():
            self._xyz           = nn.Parameter(torch.cat([self._xyz,           xyz],   dim=0))
            self._features_dc   = nn.Parameter(torch.cat([self._features_dc,   fdc],   dim=0))
            self._features_rest = nn.Parameter(torch.cat([self._features_rest, frest], dim=0))
            self._scaling       = nn.Parameter(torch.cat([self._scaling,       scale], dim=0))
            self._rotation      = nn.Parameter(torch.cat([self._rotation,      rot],   dim=0))
            self._opacity       = nn.Parameter(torch.cat([self._opacity,       opa],   dim=0))
            # Also extend accumulators
            if self._xyz_grad_accum is not None:
                self._xyz_grad_accum = torch.cat([self._xyz_grad_accum, torch.zeros(n,1,device="cuda")])
                self._denom          = torch.cat([self._denom,          torch.zeros(n,1,device="cuda")])
                self._max_radii      = torch.cat([self._max_radii,      torch.zeros(n,  device="cuda")])
        if self.optimizer is not None:
            new_refs = {
                "xyz": self._xyz, "f_dc": self._features_dc, "f_rest": self._features_rest,
                "opacity": self._opacity, "scaling": self._scaling, "rotation": self._rotation,
            }
            ext_data = {
                "xyz": xyz, "f_dc": fdc, "f_rest": frest,
                "scaling": scale, "rotation": rot, "opacity": opa,
            }
            for g in self.optimizer.param_groups:
                name  = g["name"]
                old_p = old_refs[name]
                new_p = new_refs[name]
                if old_p in self.optimizer.state and "exp_avg" in self.optimizer.state[old_p]:
                    st    = self.optimizer.state.pop(old_p)
                    ext_t = ext_data[name]
                    zeros = torch.zeros(n, *ext_t.shape[1:], device="cuda")
                    self.optimizer.state[new_p] = {
                        "step":        st["step"],
                        "exp_avg":     torch.cat([st["exp_avg"],    zeros]),
                        "exp_avg_sq":  torch.cat([st["exp_avg_sq"], zeros]),
                    }
                g["params"][0] = new_p

    def reset_opacity(self, max_val: float = 0.01):
        """Clamp opacity to prevent floaters."""
        with torch.no_grad():
            new_val = _inverse_sigmoid(torch.clamp(self.get_opacity, max=max_val))
            self._opacity.data.copy_(new_val)
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                if g["name"] == "opacity":
                    for p in g["params"]:
                        if p in self.optimizer.state:
                            self.optimizer.state[p]["exp_avg"]    .zero_()
                            self.optimizer.state[p]["exp_avg_sq"] .zero_()

    def one_up_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print(f"[Gaussian] SH degree → {self.active_sh_degree}")

    # ── I/O ───────────────────────────────────────────────────
    def save_ply(self, path: str):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        xyz    = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        fdc    = self._features_dc.detach().cpu().numpy().reshape(len(xyz), -1)
        frest  = self._features_rest.detach().cpu().numpy().reshape(len(xyz), -1)
        opa    = self._opacity.detach().cpu().numpy()
        scale  = self._scaling.detach().cpu().numpy()
        rot    = self._rotation.detach().cpu().numpy()

        attrs = np.concatenate([xyz, normals, fdc, frest, opa, scale, rot], axis=1)
        names = (["x","y","z","nx","ny","nz"] +
                 [f"f_dc_{i}"   for i in range(fdc.shape[1])] +
                 [f"f_rest_{i}" for i in range(frest.shape[1])] +
                 ["opacity"] +
                 [f"scale_{i}"    for i in range(3)] +
                 [f"rot_{i}"      for i in range(4)])
        dtype = [(n, "f4") for n in names]
        elements = np.empty(len(xyz), dtype=dtype)
        for i, n in enumerate(names):
            elements[n] = attrs[:, i].astype(np.float32)
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(str(path))
        print(f"[Gaussian] Saved {len(xyz)} splats → {path}")

    def load_ply(self, path: str):
        ply = PlyData.read(path)
        v   = ply["vertex"]
        xyz    = np.stack([v["x"], v["y"], v["z"]], axis=1)
        f_dc   = np.stack([v[f"f_dc_{i}"]   for i in range(3)], axis=1)[:, np.newaxis, :]
        f_rest = np.stack([v[f"f_rest_{i}"] for i in range(45)], axis=1).reshape(-1, 15, 3)
        opa    = v["opacity"][:, np.newaxis]
        scale  = np.stack([v[f"scale_{i}"]  for i in range(3)], axis=1)
        rot    = np.stack([v[f"rot_{i}"]    for i in range(4)], axis=1)
        self._xyz           = nn.Parameter(torch.tensor(xyz,    dtype=torch.float32).cuda())
        self._features_dc   = nn.Parameter(torch.tensor(f_dc,   dtype=torch.float32).cuda())
        self._features_rest = nn.Parameter(torch.tensor(f_rest, dtype=torch.float32).cuda())
        self._opacity       = nn.Parameter(torch.tensor(opa,    dtype=torch.float32).cuda())
        self._scaling       = nn.Parameter(torch.tensor(scale,  dtype=torch.float32).cuda())
        self._rotation      = nn.Parameter(torch.tensor(rot,    dtype=torch.float32).cuda())
        n = len(xyz)
        self._xyz_grad_accum = torch.zeros(n, 1, device="cuda")
        self._denom          = torch.zeros(n, 1, device="cuda")
        self._max_radii      = torch.zeros(n,    device="cuda")


