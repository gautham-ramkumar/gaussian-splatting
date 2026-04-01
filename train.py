#!/usr/bin/env python3
"""train.py — 3D Gaussian Splatting training entry point."""
import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.data.scene import Scene
from src.training.trainer import Trainer


def auto_hyperparams(n_cams: int, n_iters: int) -> dict:
    """
    Scale densification hyperparameters to dataset size.

    < 100 images  → small scene  (e.g. Buddha 26-cam)
    100–200 images → medium scene
    > 200 images  → large scene  (paper defaults)
    """
    if n_cams < 100:
        return dict(
            grad_threshold      = 5e-5,
            min_opacity         = 1e-3,
            densify_until       = int(n_iters * 0.75),
            opacity_reset_every = 5_000,
            opacity_reset_value = 0.1,
        )
    elif n_cams < 200:
        return dict(
            grad_threshold      = 1e-4,
            min_opacity         = 2e-3,
            densify_until       = int(min(20_000, n_iters * 0.70)),
            opacity_reset_every = 3_000,
            opacity_reset_value = 0.05,
        )
    else:
        # Paper defaults
        return dict(
            grad_threshold      = 2e-4,
            min_opacity         = 5e-3,
            densify_until       = 15_000,
            opacity_reset_every = 3_000,
            opacity_reset_value = 0.01,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Train 3D Gaussian Splatting")
    p.add_argument("--data",    required=True,  help="Dataset root dir (e.g. data/buddha)")
    p.add_argument("--sparse",  default="sparse/0",
                   help="Sparse subdir inside --data (default: sparse/0). "
                        "Use 'sparse_hw4/0' or 'sparse_colmap/0' for Buddha.")
    p.add_argument("--output",  required=True,  help="Output directory for checkpoints")
    p.add_argument("--iters",   type=int, default=30_000, help="Number of training iterations")
    p.add_argument("--images",  default=None,
                   help="Image subdirectory name inside --data (auto-detected if omitted)")
    p.add_argument("--white_bg", action="store_true", default=True,
                   help="Use white background (default True; use --no-white_bg for black)")
    p.add_argument("--no_white_bg", dest="white_bg", action="store_false")
    p.add_argument("--resolution_scale", type=float, default=1.0,
                   help="Downscale images during training (e.g. 0.5)")
    p.add_argument("--max_gaussians", type=int, default=500_000)
    p.add_argument("--test_every", type=int, default=8,
                   help="Hold out every Nth camera for evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(f"  3D Gaussian Splatting — Training")
    print(f"  Data   : {args.data}  [{args.sparse}]")
    print(f"  Output : {args.output}")
    print(f"  Iters  : {args.iters}")
    print("=" * 60)

    scene = Scene(
        data_path=args.data,
        sparse_subdir=args.sparse,
        split="train",
        test_every=args.test_every,
        image_subdir=args.images,
    )

    # Adapt default iterations for small scenes
    n_iters = args.iters
    ckpt_iters = (n_iters // 4, n_iters // 2)
    sh_steps   = (
        min(3_000,  n_iters // 10),
        min(7_500,  n_iters // 4),
        min(15_000, n_iters // 2),
    )

    hp = auto_hyperparams(len(scene.cameras), n_iters)
    tier = "small" if len(scene.cameras) < 100 else "medium" if len(scene.cameras) < 200 else "large"
    print(f"  Cameras: {len(scene.cameras)} → {tier} scene hyperparameters")
    print(f"  grad_threshold={hp['grad_threshold']:.0e}  min_opacity={hp['min_opacity']:.0e}  "
          f"densify_until={hp['densify_until']}  opacity_reset_every={hp['opacity_reset_every']}  "
          f"opacity_reset_value={hp['opacity_reset_value']}")
    print("=" * 60)

    trainer = Trainer(
        scene=scene,
        output_dir=args.output,
        n_iters=n_iters,
        bg_white=args.white_bg,
        resolution_scale=args.resolution_scale,
        max_gaussians=args.max_gaussians,
        checkpoint_iters=ckpt_iters,
        sh_degree_steps=sh_steps,
        **hp,
    )
    trainer.run()


if __name__ == "__main__":
    main()
