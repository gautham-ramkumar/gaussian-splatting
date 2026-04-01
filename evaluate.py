#!/usr/bin/env python3
"""evaluate.py — Evaluate 3DGS checkpoints across runs and iterations."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.scene import Scene
from src.evaluation.metrics import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate 3DGS checkpoints")
    p.add_argument("--data",       required=True, help="Dataset root dir")
    p.add_argument("--sparse",     default="sparse/0")
    p.add_argument("--images",     default=None)
    p.add_argument("--test_every", type=int, default=8)
    p.add_argument("--white_bg",   action="store_true", default=True)
    p.add_argument("--no_white_bg", dest="white_bg", action="store_false")
    p.add_argument("--runs",  nargs="+", required=True,
                   help="Output dirs to evaluate (e.g. output/buddha_colmap_v3 output/buddha_hw4_v2)")
    p.add_argument("--iters", nargs="+", type=int, default=[7500, 15000, 30000],
                   help="Checkpoint iterations to evaluate")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading test cameras from {args.data} [{args.sparse}] ...")
    scene = Scene(
        data_path=args.data,
        sparse_subdir=args.sparse,
        split="test",
        test_every=args.test_every,
        image_subdir=args.images,
    )
    print(f"  {len(scene.cameras)} test cameras\n")

    # results[run_label][iter] = avg metrics dict or None
    all_results = {}

    for run_dir in args.runs:
        run_dir = Path(run_dir)
        label = run_dir.name
        all_results[label] = {}

        for it in args.iters:
            ply = run_dir / f"point_cloud/iteration_{it}/point_cloud.ply"
            if not ply.exists():
                print(f"[SKIP] {ply} — not found\n")
                all_results[label][it] = None
                continue

            print(f"── {label}  iter {it} ──")
            res = evaluate(str(run_dir), scene, iteration=it)
            all_results[label][it] = res["avg"]
            print()

    # ── Print table ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EVALUATION TABLE")
    print("=" * 72)
    hdr = f"  {'Run':<26} {'Iter':>6}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'LPIPS':>8}"
    print(hdr)
    print("  " + "-" * 68)

    for label, iter_dict in all_results.items():
        for it in args.iters:
            m = iter_dict.get(it)
            if m is None:
                row = f"  {label:<26} {it:>6}  {'N/A':>10}  {'N/A':>8}  {'N/A':>8}"
            else:
                lpips_str = f"{m['lpips']:>8.4f}" if not (m['lpips'] != m['lpips']) else "    N/A "
                row = (f"  {label:<26} {it:>6}"
                       f"  {m['psnr']:>10.2f}"
                       f"  {m['ssim']:>8.4f}"
                       f"  {lpips_str}")
            print(row)
        print("  " + "-" * 68)

    print("=" * 72)

    # Save combined JSON
    out = Path("eval_combined.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
