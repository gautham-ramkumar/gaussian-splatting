# 3D Gaussian Splatting

A from-scratch PyTorch implementation of [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (Kerbl et al., 2023), tested on a 26-image Buddha dataset using both COLMAP and custom SfM poses.

## Result

**COLMAP poses @ 7,500 iterations** — PSNR 31.4 dB, SSIM 0.87

![COLMAP Output](assets/colmap_output.png)

https://github.com/gautham-ramkumar/gaussian-splatting/raw/main/assets/colmap_orbit.webm

## Evaluation

| Run | Iter | PSNR (dB) | SSIM |
|---|---|---|---|
| COLMAP poses | 7,500 | **31.40** | **0.8725** |
| COLMAP poses | 15,000 | 10.09 | 0.6046 |
| COLMAP poses | 30,000 | 14.99 | 0.7119 |
| Custom SfM (HW4) | 7,500 | 8.11 | 0.6017 |
| Custom SfM (HW4) | 15,000 | 5.65 | 0.5812 |
| Custom SfM (HW4) | 30,000 | 6.33 | 0.5986 |

> Best checkpoint is 7,500 iterations — not 30,000. See [Why early stopping matters](#why-early-stopping-matters).

## Project Structure

```
gaussian_splatting_project/
├── src/
│   ├── data/
│   │   ├── scene.py          # COLMAP loader, Camera/Point3D dataclasses
│   │   └── colmap_utils.py   # Binary COLMAP reader
│   ├── gaussian/
│   │   └── gaussian_model.py # Learnable 3D Gaussian parameters + densification
│   ├── rendering/
│   │   ├── renderer.py       # Differentiable rasterizer wrapper
│   │   └── sh_utils.py       # Spherical harmonics evaluation
│   └── evaluation/
│       └── metrics.py        # PSNR, SSIM, LPIPS
├── train.py                  # Training entry point
├── evaluate.py               # Multi-run / multi-iter evaluation table
├── render_video.py           # Orbit video renderer
├── Dockerfile
└── .github/workflows/        # Docker CI/CD
```

## How It Works

Each scene is represented as a cloud of **3D Gaussian ellipsoids**, each with:
- `xyz` — 3D position in world space
- `scale` (×3) + `rotation` (quaternion) — shape of the ellipsoid
- `opacity` — transparency
- `features` — view-dependent color encoded as Spherical Harmonics (up to degree 3)

At render time, each Gaussian is projected onto the image plane using the **EWA splatting** formula:

```
Σ_2D = J · W · Σ_3D · Wᵀ · Jᵀ
```

where `J` is the Jacobian of the perspective projection and `W` is the camera rotation. The resulting 2D ellipses are depth-sorted and alpha-composited front-to-back to produce the final image.

Loss follows Kerbl et al. Eq. 7:
```
L = (1 - λ) · L1 + λ · (1 - SSIM),   λ = 0.2
```

Densification runs every 100 iterations: Gaussians with high screen-space gradient are **cloned** (if small) or **split** (if large). Transparent Gaussians are pruned.

## Why Early Stopping Matters

With only 26 training cameras, the model overfits after ~10k iterations:

- At 7,500 iters: Gaussians represent real geometry → **PSNR 31.4 dB**
- At 30,000 iters: model memorizes training views, places floater Gaussians in empty space → **PSNR drops to 15 dB** from novel viewpoints

Densification is **additive and irreversible** — Gaussians are only cloned/split, never merged. Once the model saturates its capacity on 26 views, held-out quality degrades. Evaluating on held-out cameras is the only way to catch this.

Fix: cap at `--max_gaussians 100000` (instead of 500k default) to reduce memorization.

## Auto-Hyperparameter Tuning

Hyperparameters are automatically scaled by dataset size:

| Scene size | grad_threshold | min_opacity | densify_until | opacity_reset_value |
|---|---|---|---|---|
| < 100 cams | 5e-5 | 1e-3 | 75% of iters | 0.1 |
| 100–200 cams | 1e-4 | 2e-3 | 70% of iters | 0.05 |
| > 200 cams | 2e-4 | 5e-3 | 15,000 | 0.01 (paper default) |

## Installation

### Docker (recommended)

```bash
docker build -t gaussian-splatting .
docker run --gpus all -v /path/to/data:/app/data gaussian-splatting \
    python train.py --data /app/data/buddha --sparse sparse/0 --output /app/output
```

### Local

```bash
git clone --recursive https://github.com/gautham-ramkumar/gaussian-splatting
cd gaussian-splatting
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
pip install -e .
```

## Usage

**Train:**
```bash
python train.py --data data/buddha --sparse sparse_colmap/0 --output output/buddha --iters 30000
```

**Evaluate across checkpoints:**
```bash
python evaluate.py --data data/buddha --sparse sparse_colmap/0 \
    --runs output/buddha_colmap output/buddha_hw4 \
    --iters 7500 15000 30000
```

**Render orbit video:**
```bash
python render_video.py --model output/buddha_colmap --data data/buddha \
    --sparse sparse_colmap/0 --iter 7500 --frames 240 --fps 30
```

## Key Bugs Fixed

| Bug | Symptom | Fix |
|---|---|---|
| Missing `.transpose(0,1)` on projection matrix | Loss flatlined at 0.497 for 30k iters | `world_view_transform @ proj.T` |
| `split_mask` computed after `_clone` mutated N | `RuntimeError: tensor size mismatch` | Compute all masks before mutations |
| Adam state keyed by tensor identity | Weights stopped updating after prune | Explicit state migration on reassignment |
| 500k Gaussians on 26 images | Overfitting — later checkpoints worse | `--max_gaussians 100000` |

## Dataset

Buddha dataset: 26 images captured with a handheld camera, reconstructed with COLMAP.

---

**Reference:** Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
