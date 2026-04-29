"""3D inpainting glue: prune Gaussians by 2D masks, render fixed cameras,
compose multi-stage comparison sheets.

Designed against the gsplat training stack used in
`examples/gsplat/simple_trainer_difix3d.py`. No nerfstudio dependency.
"""

from .prune import prune_splats_by_masks, project_means_to_pixels
from .render import render_camera, render_views
from .compose import grid, label, compose_four_panel

__all__ = [
    "prune_splats_by_masks",
    "project_means_to_pixels",
    "render_camera",
    "render_views",
    "grid",
    "label",
    "compose_four_panel",
]
