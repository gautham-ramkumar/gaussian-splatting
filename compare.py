#!/usr/bin/env python3
"""compare.py — Side-by-side PSNR/SSIM/LPIPS for HW4 SfM vs COLMAP GT."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_results(model_path: str) -> dict:
    p = Path(model_path) / "eval_results.json"
    if not p.exists():
        raise FileNotFoundError(
            f"eval_results.json not found in {model_path}.\n"
            f"Run: python evaluate.py --model {model_path} --data <data_path> --sparse <sparse>"
        )
    with open(p) as f:
        return json.load(f)


def fmt(v, fmt_str=".2f") -> str:
    try:
        return format(float(v), fmt_str)
    except (TypeError, ValueError):
        return "N/A"


def main():
    p = argparse.ArgumentParser(description="Compare HW4 SfM vs COLMAP GT results")
    p.add_argument("--hw4",   required=True, help="Output dir of HW4-SfM trained model")
    p.add_argument("--colmap",required=True, help="Output dir of COLMAP-GT trained model")
    args = p.parse_args()

    hw4_r  = load_results(args.hw4)
    col_r  = load_results(args.colmap)

    print("\n" + "═" * 56)
    print("  3DGS Pose Source Comparison — Buddha Scene")
    print("═" * 56)
    print(f"  {'Metric':<12} {'HW4 SfM':>12} {'COLMAP GT':>12}  {'Δ (GT - HW4)':>14}")
    print("─" * 56)

    for metric, fmtstr in [("psnr", ".2f"), ("ssim", ".4f"), ("lpips", ".4f")]:
        h = hw4_r["avg"][metric]
        c = col_r["avg"][metric]
        delta = c - h
        sign = "+" if delta >= 0 else ""
        unit  = " dB" if metric == "psnr" else "   "
        print(f"  {metric.upper():<12} {fmt(h, fmtstr):>10}{unit} {fmt(c, fmtstr):>10}{unit}  "
              f"{sign}{fmt(abs(delta), fmtstr):>10}{unit}")
    print("─" * 56)

    n_cams = len(hw4_r.get("images", []))
    print(f"  Iters (HW4)   : {hw4_r['iteration']}")
    print(f"  Iters (COLMAP): {col_r['iteration']}")
    print(f"  Test cameras  : {n_cams}")
    print("═" * 56 + "\n")


if __name__ == "__main__":
    main()
