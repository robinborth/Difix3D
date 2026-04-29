"""Render gsplat splats from a fixed list of cameras to disk.

Used at the three stages of the inpainting pipeline (original, post-prune,
post-finetune) so we can build the 4-panel comparison sheet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import torch
from torch import Tensor

from gsplat.rendering import rasterization


@torch.no_grad()
def render_camera(
    splats: torch.nn.ParameterDict,
    K: Tensor,
    cam_to_world: Tensor,
    image_size: Tuple[int, int],
    sh_degree: int = 3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    backgrounds: Optional[Tensor] = None,
) -> Tensor:
    """Render a single camera. Returns an RGB tensor in [0, 1], shape [H, W, 3]."""
    device = splats["means"].device
    W, H = image_size
    view = torch.linalg.inv(cam_to_world.to(device))[None]  # [1, 4, 4]
    K_t = K.to(device)[None]                                  # [1, 3, 3]

    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])

    if "sh0" in splats:
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]
        out, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=view,
            Ks=K_t,
            width=W,
            height=H,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            backgrounds=backgrounds,
        )
    else:  # appearance-opt path: features+colors
        colors = torch.sigmoid(splats["colors"])
        out, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=view,
            Ks=K_t,
            width=W,
            height=H,
            near_plane=near_plane,
            far_plane=far_plane,
            backgrounds=backgrounds,
        )

    img = out[0].clamp(0.0, 1.0)
    return img


def render_views(
    splats: torch.nn.ParameterDict,
    Ks: Sequence[Tensor],
    cam_to_worlds: Sequence[Tensor],
    image_sizes: Sequence[Tuple[int, int]],
    out_dir: Path,
    stems: Sequence[str],
    sh_degree: int = 3,
) -> None:
    """Render every camera in the list and save PNGs to out_dir/<stem>.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for K, c2w, sz, stem in zip(Ks, cam_to_worlds, image_sizes, stems):
        img = render_camera(splats, K, c2w, sz, sh_degree=sh_degree)
        arr = (img.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        iio.imwrite(out_dir / f"{stem}.png", arr)
