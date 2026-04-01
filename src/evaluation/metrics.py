"""Evaluation metrics: PSNR, SSIM, LPIPS."""
import math
import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn

try:
    import lpips as lpips_lib
    _lpips_model = None
    def _get_lpips():
        global _lpips_model
        if _lpips_model is None:
            _lpips_model = lpips_lib.LPIPS(net="alex").cuda()
        return _lpips_model
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR of (3,H,W) tensors in [0,1]."""
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM of (3,H,W) tensors in [0,1]."""
    return ssim_fn(pred.unsqueeze(0), gt.unsqueeze(0),
                   data_range=1.0, size_average=True).item()


def compute_lpips(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """LPIPS of (3,H,W) tensors in [0,1]."""
    if not HAS_LPIPS:
        return float("nan")
    model = _get_lpips()
    with torch.no_grad():
        # LPIPS expects [-1,1]
        p = pred.unsqueeze(0) * 2 - 1
        g = gt.unsqueeze(0)   * 2 - 1
        return model(p, g).item()


def evaluate(
    model_path: str,
    scene,
    iteration: int = -1,
) -> dict:
    """
    Render all test cameras from a saved checkpoint and compute metrics.

    Args:
        model_path: directory containing point_cloud/ subdirs
        scene:      Scene loaded with split="test"
        iteration:  which checkpoint iteration. -1 = latest.

    Returns:
        dict with per-image metrics and averages.
    """
    from src.gaussian.gaussian_model import GaussianModel
    from src.rendering.renderer import render

    model_path = Path(model_path)

    # Find checkpoint
    if iteration == -1:
        ply_dirs = sorted((model_path / "point_cloud").glob("iteration_*"),
                          key=lambda p: int(p.name.split("_")[1]))
        if not ply_dirs:
            raise FileNotFoundError(f"No checkpoints found in {model_path}")
        ply_path = ply_dirs[-1] / "point_cloud.ply"
        iteration = int(ply_dirs[-1].name.split("_")[1])
    else:
        ply_path = model_path / f"point_cloud/iteration_{iteration}/point_cloud.ply"

    print(f"[Eval] Loading checkpoint: {ply_path}")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_path))
    gaussians.cuda()
    gaussians.active_sh_degree = 3

    bg = torch.tensor([1., 1., 1.], dtype=torch.float32).cuda()

    results = {"iteration": iteration, "images": [], "avg": {}}
    psnrs, ssims, lpips_vals = [], [], []

    for cam in scene.cameras:
        out = render(cam, gaussians, bg)
        pred = out["render"].clamp(0, 1)
        gt   = cam.load_image().cuda()

        p = psnr(pred, gt)
        s = compute_ssim(pred, gt)
        l = compute_lpips(pred, gt)

        psnrs.append(p); ssims.append(s); lpips_vals.append(l)
        results["images"].append({"name": cam.image_name, "psnr": p, "ssim": s, "lpips": l})

    results["avg"] = {
        "psnr":  sum(psnrs)  / len(psnrs),
        "ssim":  sum(ssims)  / len(ssims),
        "lpips": sum(lpips_vals) / len(lpips_vals),
    }

    out_json = model_path / "eval_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] PSNR {results['avg']['psnr']:.2f} | "
          f"SSIM {results['avg']['ssim']:.4f} | "
          f"LPIPS {results['avg']['lpips']:.4f}  → {out_json}")
    return results
