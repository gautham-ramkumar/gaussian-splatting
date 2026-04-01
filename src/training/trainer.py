"""Training loop for 3D Gaussian Splatting."""
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn
from tqdm import tqdm

from src.data.scene import Camera, Scene
from src.gaussian.gaussian_model import GaussianModel
from src.rendering.renderer import render


class Trainer:
    """Full 3DGS training loop.

    Loss:  L = (1 - λ) · L1(Î, I) + λ · (1 - SSIM(Î, I))    λ = 0.2
    Schedule from Kerbl et al. (2023):
      - Densification every 100 iters (100 → 15k)
      - Opacity reset at iter 1k
      - SH degree steps at 3k / 7.5k / 15k
      - Checkpoints at 7k / 15k / final
    """

    def __init__(
        self,
        scene: Scene,
        output_dir: str,
        n_iters: int = 30_000,
        lambda_dssim: float = 0.2,
        densify_from: int = 100,
        densify_until: int = 15_000,
        densify_every: int = 100,
        opacity_reset_every: int = 3_000,
        opacity_reset_value: float = 0.01,
        sh_degree_steps: tuple = (3_000, 7_500, 15_000),
        grad_threshold: float = 2e-4,
        min_opacity: float = 5e-3,
        max_gaussians: int = 500_000,
        percent_dense: float = 0.01,
        bg_white: bool = True,
        checkpoint_iters: tuple = (7_000, 15_000),
        resolution_scale: float = 1.0,
    ):
        self.scene       = scene
        self.output_dir  = Path(output_dir)
        self.n_iters     = n_iters
        self.lambda_dssim = lambda_dssim
        self.densify_from   = densify_from
        self.densify_until  = densify_until
        self.densify_every  = densify_every
        self.opacity_reset_every  = opacity_reset_every
        self.opacity_reset_value  = opacity_reset_value
        self.sh_steps    = sh_degree_steps
        self.grad_thr    = grad_threshold
        self.min_opacity = min_opacity
        self.max_gaussians = max_gaussians
        self.percent_dense = percent_dense
        self.bg_color    = torch.tensor([1., 1., 1.] if bg_white else [0., 0., 0.],
                                        dtype=torch.float32).cuda()
        self.ckpt_iters  = set(checkpoint_iters)
        self.res_scale   = resolution_scale

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.losses: List[float] = []

    def run(self) -> GaussianModel:
        scene   = self.scene
        gaussians = GaussianModel(sh_degree=3)

        # Initialise from COLMAP point cloud
        xyzs = torch.tensor([p.xyz for p in scene.point_cloud], dtype=torch.float32).numpy()
        rgbs = torch.tensor([p.rgb for p in scene.point_cloud], dtype=torch.float32).numpy()
        import numpy as np
        gaussians.create_from_pcd(
            np.array([p.xyz for p in scene.point_cloud]),
            np.array([p.rgb for p in scene.point_cloud]),
            scene.scene_radius,
        )
        gaussians.cuda()
        gaussians.setup_optimizer()

        train_cams: List[Camera] = scene.cameras
        sh_step_idx = 0

        pbar = tqdm(range(1, self.n_iters + 1), desc="Training")
        t0   = time.time()

        for step in pbar:
            # ── LR schedule ─────────────────────────────────
            gaussians.update_position_lr(step, self.n_iters)

            # ── SH degree schedule ───────────────────────────
            if sh_step_idx < len(self.sh_steps) and step == self.sh_steps[sh_step_idx]:
                gaussians.one_up_sh_degree()
                sh_step_idx += 1

            # ── Pick random training camera ───────────────────
            cam = random.choice(train_cams)

            # ── Render ───────────────────────────────────────
            out = render(cam, gaussians, self.bg_color)
            img_pred = out["render"]                         # (3, H, W)

            # ── Load GT image ────────────────────────────────
            img_gt = cam.load_image().cuda()                 # (3, H, W)
            if self.res_scale != 1.0:
                h = int(img_gt.shape[1] * self.res_scale)
                w = int(img_gt.shape[2] * self.res_scale)
                img_gt   = F.interpolate(img_gt.unsqueeze(0),   (h, w), mode="bilinear", align_corners=False).squeeze(0)
                img_pred = F.interpolate(img_pred.unsqueeze(0), (h, w), mode="bilinear", align_corners=False).squeeze(0)

            # ── Loss (Kerbl et al. Eq. 7) ────────────────────
            l1   = F.l1_loss(img_pred, img_gt)
            ssim = 1.0 - ssim_fn(img_pred.unsqueeze(0), img_gt.unsqueeze(0),
                                  data_range=1.0, size_average=True)
            loss = (1 - self.lambda_dssim) * l1 + self.lambda_dssim * ssim

            loss.backward()
            self.losses.append(loss.item())

            with torch.no_grad():
                # ── Accumulate densification stats ───────────
                if self.densify_from <= step <= self.densify_until:
                    gaussians.update_stats(
                        out["viewspace_points"],
                        out["radii"],
                        out["visibility_filter"],
                    )

                # ── Densification ────────────────────────────
                if (self.densify_from <= step <= self.densify_until and
                        step % self.densify_every == 0):
                    gaussians.densify_and_prune(
                        self.grad_thr, self.min_opacity,
                        scene.scene_radius, self.max_gaussians,
                        self.percent_dense,
                    )
                    torch.cuda.empty_cache()

                # ── Opacity reset ────────────────────────────
                if step % self.opacity_reset_every == 0:
                    gaussians.reset_opacity(self.opacity_reset_value)

                # ── Optimiser step ───────────────────────────
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # ── Progress bar ─────────────────────────────────
            if step % 100 == 0:
                avg = sum(self.losses[-100:]) / 100
                pbar.set_postfix({
                    "loss": f"{avg:.4f}",
                    "N":    gaussians.num_gaussians,
                    "SH":   gaussians.active_sh_degree,
                })

            # ── Checkpoint ───────────────────────────────────
            if step in self.ckpt_iters:
                ckpt = self.output_dir / f"point_cloud/iteration_{step}/point_cloud.ply"
                gaussians.save_ply(str(ckpt))

        # ── Final save ────────────────────────────────────────
        final = self.output_dir / f"point_cloud/iteration_{self.n_iters}/point_cloud.ply"
        gaussians.save_ply(str(final))
        elapsed = time.time() - t0
        print(f"\n[Trainer] Done in {elapsed/60:.1f} min | "
              f"Final loss {self.losses[-1]:.4f} | {gaussians.num_gaussians} splats")

        # Save loss log
        import json
        with open(self.output_dir / "loss_log.json", "w") as f:
            json.dump(self.losses, f)

        return gaussians
