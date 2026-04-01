"""Differentiable Gaussian rasterization renderer."""
import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from src.data.scene import Camera
from src.gaussian.gaussian_model import GaussianModel
from .sh_utils import eval_sh


def render(camera: Camera,
           gaussians: GaussianModel,
           bg_color: torch.Tensor,
           scaling_modifier: float = 1.0) -> dict:
    """
    Render a single view using the differentiable Gaussian rasterizer.

    Args:
        camera:           Camera object (with world_view_transform etc.)
        gaussians:        GaussianModel (all params on CUDA)
        bg_color:         (3,) background colour tensor on CUDA
        scaling_modifier: global scale multiplier (1.0 during training)

    Returns:
        dict with keys:
            "render"           – (3, H, W) rendered image
            "viewspace_points" – screen-space points (for grad accumulation)
            "visibility_filter"– boolean mask of visible Gaussians
            "radii"            – 2D pixel radii per Gaussian
    """
    # ── Screen-space placeholder for 2D gradients ─────────────
    means2D = torch.zeros_like(gaussians.get_xyz,
                               dtype=torch.float32, requires_grad=True, device="cuda")
    try:
        means2D.retain_grad()
    except Exception:
        pass

    # ── SH → RGB at this viewpoint ────────────────────────────
    sh_degree = gaussians.active_sh_degree
    if sh_degree > 0:
        # View direction: world space  (normalised)
        dirs = gaussians.get_xyz.detach() - camera.camera_center
        dirs = dirs / (torch.norm(dirs, dim=1, keepdim=True) + 1e-7)

        shs = gaussians.get_features.transpose(1, 2)   # (N, 3, 16)
        sh = shs[:, :, :(sh_degree + 1) ** 2]          # active coeffs only
        rgb = eval_sh(sh_degree, sh.transpose(1, 2), dirs)
        colors_precomp = torch.clamp_min(rgb + 0.5, 0.0)
        shs_arg = None
    else:
        # Degree 0: let rasterizer use DC only (avoids Python SH eval)
        colors_precomp = None
        shs_arg = gaussians.get_features

    # ── Rasteriser settings ────────────────────────────────────
    tan_fov_x = math.tan(camera.fov_x / 2)
    tan_fov_y = math.tan(camera.fov_y / 2)

    settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    rendered, radii = rasterizer(
        means3D=gaussians.get_xyz,
        means2D=means2D,
        shs=shs_arg,
        colors_precomp=colors_precomp,
        opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        cov3D_precomp=None,
    )

    return {
        "render":            rendered,
        "viewspace_points":  means2D,
        "visibility_filter": radii > 0,
        "radii":             radii,
    }
