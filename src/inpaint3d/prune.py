"""Project gsplat Gaussian centers into camera views, sample 2D masks, and
prune the Gaussians that fall inside the masks in a configurable fraction
of their visible views.

The pruning is done in-place on a `torch.nn.ParameterDict` (the standard
gsplat `splats`) and on the matching dict of per-key Adam optimizers,
preserving Adam state for the kept Gaussians. Densification strategy
state (DefaultStrategy / MCMCStrategy) is reset by the caller after
pruning — the indices it tracks (running_stats, etc.) are no longer
valid.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def project_means_to_pixels(
    means: Tensor,                # [N, 3] world-space
    K: Tensor,                    # [3, 3] pinhole intrinsics
    world_to_cam: Tensor,         # [4, 4]
    image_size: Tuple[int, int],  # (W, H)
) -> Tuple[Tensor, Tensor]:
    """Project Gaussian centers into one camera.

    Returns
    -------
    pix : Tensor[N, 2]  pixel coords (x, y), in float
    visible : Tensor[N] bool  True where in front of camera AND inside the image
    """
    N = means.shape[0]
    if N == 0:
        return means.new_zeros((0, 2)), torch.zeros(0, dtype=torch.bool, device=means.device)

    homog = torch.cat([means, means.new_ones((N, 1))], dim=1)        # [N, 4]
    cam = (world_to_cam @ homog.T).T                                  # [N, 4]
    z = cam[:, 2]
    in_front = z > 1e-6

    cam_xy = cam[:, :2] / z.clamp(min=1e-6).unsqueeze(1)              # [N, 2]
    pix = (K[:2, :2] @ cam_xy.T).T + K[:2, 2]                         # [N, 2]

    W, H = image_size
    in_bounds = (
        (pix[:, 0] >= 0) & (pix[:, 0] < W) &
        (pix[:, 1] >= 0) & (pix[:, 1] < H)
    )
    visible = in_front & in_bounds
    return pix, visible


def _sample_mask(mask: Tensor, pix: Tensor, visible: Tensor) -> Tensor:
    """Sample a HxW bool/float mask at the (rounded) projected pixels.

    Returns a Tensor[N] bool where True = mask hit. Invisible projections
    return False.
    """
    H, W = mask.shape[:2]
    N = pix.shape[0]
    out = torch.zeros(N, dtype=torch.bool, device=mask.device)
    if not visible.any():
        return out
    ix = pix[:, 0].clamp(0, W - 1).long()
    iy = pix[:, 1].clamp(0, H - 1).long()
    sampled = mask[iy[visible], ix[visible]]
    if sampled.dtype != torch.bool:
        sampled = sampled > 0.5
    out[visible] = sampled
    return out


def hit_rate_across_views(
    means: Tensor,                            # [N, 3]
    Ks: Sequence[Tensor],                     # K per view, [3,3] each
    world_to_cams: Sequence[Tensor],          # w2c per view, [4,4] each
    masks: Sequence[Tensor],                  # H,W per view (bool or float)
    image_sizes: Sequence[Tuple[int, int]],   # (W, H) per view
) -> Tuple[Tensor, Tensor]:
    """Returns (hit_rate[N], visible_count[N])

    hit_rate[i] = #views where the projection of mean i falls inside the mask
                  / #views where mean i is visible (in-frame, in-front).
                  0 when visible_count == 0.
    """
    device = means.device
    N = means.shape[0]
    hits = torch.zeros(N, dtype=torch.long, device=device)
    visible_count = torch.zeros(N, dtype=torch.long, device=device)

    for K, w2c, mask, size in zip(Ks, world_to_cams, masks, image_sizes):
        pix, vis = project_means_to_pixels(
            means, K.to(device), w2c.to(device), size
        )
        in_mask = _sample_mask(mask.to(device), pix, vis)
        visible_count += vis.to(torch.long)
        hits += in_mask.to(torch.long)

    rate = torch.where(
        visible_count > 0,
        hits.float() / visible_count.float(),
        torch.zeros_like(hits, dtype=torch.float),
    )
    return rate, visible_count


def prune_splats_by_masks(
    splats: torch.nn.ParameterDict,
    optimizers: Dict[str, torch.optim.Optimizer],
    Ks: Sequence[Tensor],
    world_to_cams: Sequence[Tensor],
    masks: Sequence[Tensor],
    image_sizes: Sequence[Tuple[int, int]],
    threshold: float = 0.4,
    min_visible_views: int = 3,
) -> dict:
    """Drop every Gaussian whose mask hit-rate is >= threshold across the
    views it is visible in (with at least `min_visible_views` visible views,
    so we don't kill Gaussians the views never see).

    Mutates `splats` and the underlying optimizer state in place. Returns
    a small report dict.
    """
    means = splats["means"].detach()
    rate, vis_count = hit_rate_across_views(
        means, Ks, world_to_cams, masks, image_sizes
    )
    keep = ~((rate >= threshold) & (vis_count >= min_visible_views))
    pruned = int((~keep).sum().item())

    # Boolean-index every key, preserving Adam state for the kept rows.
    for key, param in list(splats.items()):
        new_param = torch.nn.Parameter(param.detach()[keep].clone())
        splats[key] = new_param
        optim = optimizers[key]
        # Adam keeps exp_avg / exp_avg_sq tied to the param tensor — rebuild
        # a fresh optimizer of the same class with the same hyperparams,
        # then copy in the sliced state.
        old_state = optim.state.get(param, None)
        param_groups = optim.param_groups
        optim.param_groups = []  # clear before re-adding
        optim.state.clear()
        new_pg = {**param_groups[0], "params": [new_param]}
        optim.add_param_group(new_pg)
        if old_state is not None:
            new_state = {}
            for k, v in old_state.items():
                if isinstance(v, torch.Tensor) and v.shape[: 1] == param.shape[: 1]:
                    new_state[k] = v[keep].clone()
                else:
                    new_state[k] = v
            optim.state[new_param] = new_state

    return {
        "n_total": int(rate.numel()),
        "n_pruned": pruned,
        "n_kept": int(keep.sum().item()),
        "n_visible_in_any_view": int((vis_count > 0).sum().item()),
        "threshold": threshold,
        "min_visible_views": min_visible_views,
    }
