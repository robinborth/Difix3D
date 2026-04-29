"""3D inpainting debug trainer (gsplat-only, full control loop).

Pipeline:
  1. (optional) train splatfacto from scratch on a colmap scene
  2. load (or resume) checkpoint
  3. render every subset view -> 01_original/
  4. project Gaussian centers into the subset views, prune those that fall
     inside the SAM3 masks in >= prune_threshold of their visible views
  5. render -> 02_after_removal/
  6. swap the dataparser image_paths to images_inpainted/, fine-tune for
     `finetune_steps` steps with the existing densification strategy
  7. render -> 03_after_finetune/
  8. compose 4-panel sheets per view

Forked from `examples/gsplat/simple_trainer_difix3d.py`; the Difix branch,
appearance opt, bilateral grid, depth loss, and PNG compression hooks are
all stripped to keep the loop legible. Densification (DefaultStrategy /
MCMCStrategy) is kept fully in the user's hands via Config.

Run from repo root:

    PYTHONPATH=src:examples/gsplat python examples/gsplat/simple_trainer_inpaint3d.py default \\
        --data_dir /cluster/angmar/rborth/difix3d/data/garden \\
        --result_dir outputs/inpaint3d_garden \\
        --masks_subdir sam3_masks \\
        --inpainted_subdir images_inpainted \\
        --train_steps 30000 \\
        --finetune_steps 2000 \\
        --view_stride 6
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from PIL import Image
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Literal, assert_never

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from datasets.colmap import Dataset, Parser
from utils import knn, rgb_to_sh, set_random_seed

# inpaint3d helpers (src/inpaint3d/)
from inpaint3d.prune import prune_splats_by_masks
from inpaint3d.render import render_views
from inpaint3d.compose import compose_four_panel


# ============================ config ============================


@dataclass
class Config:
    # --- data ---
    data_dir: str = "data/garden"
    images_subdir: str = "images"
    masks_subdir: str = "sam3_masks"
    inpainted_subdir: str = "images_inpainted"
    data_factor: int = 4
    test_every: int = 8
    normalize_world_space: bool = True
    global_scale: float = 1.0

    # --- subset for prune + comparison sheets ---
    view_stride: int = 6
    view_limit: Optional[int] = None

    # --- output ---
    result_dir: str = "outputs/inpaint3d"
    save_ckpts: bool = True

    # --- model ---
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_type: Literal["sfm", "random"] = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    init_opa: float = 0.1
    init_scale: float = 1.0

    # --- pipeline phases ---
    ckpt: Optional[str] = None       # if set, skip training, jump to step 3
    train_steps: int = 30_000        # used when ckpt is None
    finetune_steps: int = 2_000      # after pruning

    # --- prune ---
    prune_threshold: float = 0.4
    prune_min_visible_views: int = 3
    # During finetune, drop training views that have no inpainted counterpart so
    # the loss isn't fighting un-edited vase pixels. Set False to keep originals.
    drop_uninpainted_views: bool = True

    # --- training knobs (kept gsplat-style) ---
    batch_size: int = 1
    ssim_lambda: float = 0.2
    near_plane: float = 0.01
    far_plane: float = 1e10
    random_bkgd: bool = False
    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    # densification strategy
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False

    # logging
    tb_every: int = 200


# ============================ splats helpers ============================


def create_splats(
    parser: Parser,
    cfg: Config,
    device: str,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if cfg.init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    else:
        scene_scale = parser.scene_scale * 1.1 * cfg.global_scale
        points = cfg.init_extent * scene_scale * (
            torch.rand((cfg.init_num_pts, 3)) * 2 - 1
        )
        rgbs = torch.rand((cfg.init_num_pts, 3))

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), cfg.init_opa))

    sh_K = (cfg.sh_degree + 1) ** 2
    colors = torch.zeros((N, sh_K, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)

    scene_scale = parser.scene_scale * 1.1 * cfg.global_scale
    params = [
        ("means",     torch.nn.Parameter(points),    1.6e-4 / 10 * scene_scale),
        ("scales",    torch.nn.Parameter(scales),    5e-3 / 5),
        ("quats",     torch.nn.Parameter(quats),     1e-3 / 5),
        ("opacities", torch.nn.Parameter(opacities), 5e-2 / 5),
        ("sh0",       torch.nn.Parameter(colors[:, :1, :]),  2.5e-3 / 50),
        ("shN",       torch.nn.Parameter(colors[:, 1:, :]),  2.5e-3 / 20 / 50),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    BS = cfg.batch_size
    optimizer_class = (
        torch.optim.SparseAdam if cfg.sparse_grad else torch.optim.Adam
    )
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def save_ckpt(path: Path, splats: torch.nn.ParameterDict, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "splats": {k: v.detach().cpu() for k, v in splats.items()},
            "step": step,
        },
        path,
    )


def load_ckpt_into(splats: torch.nn.ParameterDict, ckpt_path: Path, device: str) -> int:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    for k in splats.keys():
        splats[k].data = ckpt["splats"][k].to(device)
    return int(ckpt.get("step", 0))


# ============================ render ============================


@torch.no_grad()
def render_camera_inproc(
    splats: torch.nn.ParameterDict,
    K: Tensor,
    cam_to_world: Tensor,
    image_size: Tuple[int, int],
    cfg: Config,
    device: str,
) -> Tensor:
    """Render a single camera and return RGB Tensor[H, W, 3] in [0, 1]."""
    W, H = image_size
    view = torch.linalg.inv(cam_to_world.to(device))[None]
    K_t = K.to(device)[None]
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

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
        sh_degree=cfg.sh_degree,
        near_plane=cfg.near_plane,
        far_plane=cfg.far_plane,
        packed=cfg.packed,
    )
    return out[0].clamp(0.0, 1.0)


def render_subset_to_dir(
    splats: torch.nn.ParameterDict,
    parser: Parser,
    indices: Sequence[int],
    out_dir: Path,
    cfg: Config,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        cam_id = parser.camera_ids[idx]
        K = torch.from_numpy(parser.Ks_dict[cam_id]).float()
        c2w = torch.from_numpy(parser.camtoworlds[idx]).float()
        W, H = parser.imsize_dict[cam_id]
        img = render_camera_inproc(splats, K, c2w, (W, H), cfg, device)
        arr = (img.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        stem = Path(parser.image_paths[idx]).stem
        iio.imwrite(out_dir / f"{stem}.png", arr)


def render_cached_to_dir(
    splats: torch.nn.ParameterDict,
    Ks: Sequence[Tensor],
    cam_to_worlds: Sequence[Tensor],
    image_sizes: Sequence[Tuple[int, int]],
    stems: Sequence[str],
    out_dir: Path,
    cfg: Config,
    device: str,
) -> None:
    """Like render_subset_to_dir but doesn't read from a Parser — works after
    the parser has been shrunk by swap_image_paths_to_inpainted(drop_missing=True)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for K, c2w, sz, stem in zip(Ks, cam_to_worlds, image_sizes, stems):
        img = render_camera_inproc(splats, K, c2w, sz, cfg, device)
        arr = (img.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        iio.imwrite(out_dir / f"{stem}.png", arr)


# ============================ training ============================


def train_loop(
    splats: torch.nn.ParameterDict,
    optimizers: Dict[str, torch.optim.Optimizer],
    parser: Parser,
    trainset: Dataset,
    cfg: Config,
    device: str,
    start_step: int,
    end_step: int,
    strategy_state,
    writer: Optional[SummaryWriter] = None,
    label: str = "train",
) -> None:
    """Tight gsplat training loop. Single-GPU. Densification via cfg.strategy."""
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4,
        persistent_workers=True, pin_memory=True,
    )
    train_iter = iter(trainloader)

    pbar = tqdm.tqdm(range(start_step, end_step), desc=label)
    for step in pbar:
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            data = next(train_iter)

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0  # [B, H, W, 3]
        H, W = pixels.shape[1:3]

        viewmats = torch.linalg.inv(camtoworlds)
        means = splats["means"]
        quats = splats["quats"]
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        sh_deg = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        renders, alphas, info = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks, width=W, height=H,
            sh_degree=sh_deg,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
            packed=cfg.packed,
        )
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            renders = renders + (1.0 - alphas) * bkgd

        l1 = F.l1_loss(renders, pixels)
        # SSIM as 1 - SSIM via fused if available, else simple proxy
        try:
            from fused_ssim import fused_ssim  # local import — optional
            ssim = fused_ssim(
                renders.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2),
                padding="valid",
            )
            ssim_loss = 1.0 - ssim
        except Exception:
            ssim_loss = renders.new_zeros(())

        loss = (1.0 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * ssim_loss

        if cfg.opacity_reg > 0:
            loss = loss + cfg.opacity_reg * torch.sigmoid(splats["opacities"]).mean()
        if cfg.scale_reg > 0:
            loss = loss + cfg.scale_reg * torch.exp(splats["scales"]).mean()

        cfg.strategy.step_pre_backward(
            params=splats, optimizers=optimizers, state=strategy_state,
            step=step, info=info,
        )
        loss.backward()

        if writer is not None and step % cfg.tb_every == 0:
            writer.add_scalar(f"{label}/loss", loss.item(), step)
            writer.add_scalar(f"{label}/l1", l1.item(), step)
            writer.add_scalar(f"{label}/n_gauss", splats["means"].shape[0], step)

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        cfg.strategy.step_post_backward(
            params=splats, optimizers=optimizers, state=strategy_state,
            step=step, info=info, packed=cfg.packed,
        )

        if step % 50 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", n=f"{splats['means'].shape[0]}",
            )


# ============================ orchestration ============================


def pick_subset(parser: Parser, stride: int, limit: Optional[int]) -> List[int]:
    n = len(parser.image_paths)
    idxs = list(range(0, n, max(1, stride)))
    if limit is not None:
        idxs = idxs[:limit]
    return idxs


def load_masks_for_indices(
    parser: Parser,
    indices: Sequence[int],
    masks_dir: Path,
    image_sizes: Sequence[Tuple[int, int]],
    device: str,
) -> List[Tensor]:
    """Returns one HxW bool mask per index (resized to that image's HxW)."""
    out: List[Tensor] = []
    for idx, (W, H) in zip(indices, image_sizes):
        stem = Path(parser.image_paths[idx]).stem
        mp = masks_dir / f"{stem}.png"
        if not mp.exists():
            print(f"[warn] mask missing for {stem}, treating as all-zero")
            out.append(torch.zeros((H, W), dtype=torch.bool, device=device))
            continue
        m = Image.open(mp).convert("L")
        if m.size != (W, H):
            m = m.resize((W, H), Image.NEAREST)
        arr = np.asarray(m) > 127
        out.append(torch.from_numpy(arr).to(device))
    return out


def swap_image_paths_to_inpainted(
    parser: Parser, inpainted_dir: Path, drop_missing: bool = False,
) -> None:
    """In-place: redirect parser.image_paths entries to images_inpainted/<stem>.<ext>.

    When `drop_missing` is True (default during debug runs), views without an
    inpainted counterpart are removed from the parser entirely so the finetune
    loss isn't fighting un-inpainted vase pixels.

    Mutates: image_paths, image_names, camtoworlds, camera_ids.
    """
    keep_indices: List[int] = []
    new_paths: List[str] = []
    n_swapped = 0
    n_missing = 0
    for i, p in enumerate(parser.image_paths):
        stem = Path(p).stem
        replacement = None
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = inpainted_dir / f"{stem}{ext}"
            if cand.exists():
                replacement = cand
                break
        if replacement is None:
            n_missing += 1
            if drop_missing:
                continue
            new_paths.append(p)
            keep_indices.append(i)
        else:
            new_paths.append(str(replacement))
            keep_indices.append(i)
            n_swapped += 1

    if drop_missing and n_missing:
        parser.image_paths = new_paths
        parser.image_names = [parser.image_names[i] for i in keep_indices]
        parser.camtoworlds = parser.camtoworlds[keep_indices]
        parser.camera_ids = [parser.camera_ids[i] for i in keep_indices]
    else:
        parser.image_paths = new_paths
    print(f"[swap] redirected {n_swapped} image paths to {inpainted_dir} "
          f"(missing {n_missing}; drop_missing={drop_missing})")


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config) -> None:
    set_random_seed(42 + local_rank)
    device = f"cuda:{local_rank}"

    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ---- 1. data ----
    parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=cfg.test_every,
    )
    trainset = Dataset(parser, split="train")
    print(f"[data] {len(parser.image_paths)} images, "
          f"{parser.points.shape[0]} sfm points, scene_scale={parser.scene_scale:.3f}")

    splats, optimizers = create_splats(parser, cfg, device)
    cfg.strategy.check_sanity(splats, optimizers)
    if isinstance(cfg.strategy, DefaultStrategy):
        strategy_state = cfg.strategy.initialize_state(
            scene_scale=parser.scene_scale * 1.1 * cfg.global_scale,
        )
    elif isinstance(cfg.strategy, MCMCStrategy):
        strategy_state = cfg.strategy.initialize_state()
    else:
        assert_never(cfg.strategy)

    # ---- 2. train or load ----
    if cfg.ckpt is not None:
        ckpt_path = Path(cfg.ckpt)
        print(f"[step 1] loading ckpt {ckpt_path}")
        # If shape mismatch, replace splats wholesale.
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        for k in splats.keys():
            splats[k] = torch.nn.Parameter(ck["splats"][k].to(device))
        # Rebuild optimizers tied to the new tensors.
        for k, opt in optimizers.items():
            opt.param_groups = []
            opt.state.clear()
            opt.add_param_group({"params": [splats[k]], "lr": opt.defaults["lr"], "name": k})
        start_step = int(ck.get("step", 0))
        cfg.strategy.check_sanity(splats, optimizers)
        if isinstance(cfg.strategy, DefaultStrategy):
            strategy_state = cfg.strategy.initialize_state(
                scene_scale=parser.scene_scale * 1.1 * cfg.global_scale,
            )
        elif isinstance(cfg.strategy, MCMCStrategy):
            strategy_state = cfg.strategy.initialize_state()
        print(f"[step 1] resumed at step {start_step} with "
              f"{splats['means'].shape[0]} gaussians")
    else:
        print(f"[step 1] training from sfm init for {cfg.train_steps} steps")
        train_loop(
            splats, optimizers, parser, trainset, cfg, device,
            start_step=0, end_step=cfg.train_steps,
            strategy_state=strategy_state, writer=writer, label="train",
        )
        if cfg.save_ckpts:
            save_ckpt(out_dir / "ckpts" / "step-train.ckpt", splats, cfg.train_steps)
        start_step = cfg.train_steps

    # ---- 3. choose subset ----
    subset = pick_subset(parser, cfg.view_stride, cfg.view_limit)
    print(f"[subset] {len(subset)} views (stride={cfg.view_stride})")

    subset_image_sizes: List[Tuple[int, int]] = []
    subset_Ks: List[Tensor] = []
    subset_c2w: List[Tensor] = []
    subset_w2c: List[Tensor] = []
    subset_stems: List[str] = []
    for idx in subset:
        cam_id = parser.camera_ids[idx]
        W, H = parser.imsize_dict[cam_id]
        K = torch.from_numpy(parser.Ks_dict[cam_id]).float()
        c2w = torch.from_numpy(parser.camtoworlds[idx]).float()
        subset_image_sizes.append((W, H))
        subset_Ks.append(K)
        subset_c2w.append(c2w)
        subset_w2c.append(torch.linalg.inv(c2w))
        subset_stems.append(Path(parser.image_paths[idx]).stem)

    # ---- 4. render original ----
    render_subset_to_dir(splats, parser, subset, out_dir / "01_original", cfg, device)

    # ---- 5. prune by mask ----
    masks_dir = Path(cfg.data_dir) / cfg.masks_subdir
    masks = load_masks_for_indices(
        parser, subset, masks_dir, subset_image_sizes, device,
    )
    print(f"[prune] using {len(masks)} masks from {masks_dir}")
    report = prune_splats_by_masks(
        splats, optimizers,
        Ks=subset_Ks,
        world_to_cams=subset_w2c,
        masks=masks,
        image_sizes=subset_image_sizes,
        threshold=cfg.prune_threshold,
        min_visible_views=cfg.prune_min_visible_views,
    )
    (out_dir / "prune_report.json").write_text(json.dumps(report, indent=2))
    print(f"[prune] {report}")

    # Strategy state has stale per-Gaussian arrays after pruning — reset.
    if isinstance(cfg.strategy, DefaultStrategy):
        strategy_state = cfg.strategy.initialize_state(
            scene_scale=parser.scene_scale * 1.1 * cfg.global_scale,
        )
    elif isinstance(cfg.strategy, MCMCStrategy):
        strategy_state = cfg.strategy.initialize_state()

    # ---- 6. render after removal ----
    render_subset_to_dir(splats, parser, subset, out_dir / "02_after_removal", cfg, device)

    # ---- 7. swap to inpainted, finetune ----
    inpainted_dir = Path(cfg.data_dir) / cfg.inpainted_subdir
    swap_image_paths_to_inpainted(parser, inpainted_dir, drop_missing=cfg.drop_uninpainted_views)
    trainset_inp = Dataset(parser, split="train")
    train_loop(
        splats, optimizers, parser, trainset_inp, cfg, device,
        start_step=start_step, end_step=start_step + cfg.finetune_steps,
        strategy_state=strategy_state, writer=writer, label="finetune",
    )
    if cfg.save_ckpts:
        save_ckpt(
            out_dir / "ckpts" / "step-finetune.ckpt",
            splats, start_step + cfg.finetune_steps,
        )

    # ---- 8. render after finetune ----
    # Use cached cams/stems because parser.* may have been shrunk by the swap.
    render_cached_to_dir(
        splats, subset_Ks, subset_c2w, subset_image_sizes, subset_stems,
        out_dir / "03_after_finetune", cfg, device,
    )

    # ---- 9. compose 4-panel sheets ----
    panels_dir = out_dir / "comparison"
    panels_dir.mkdir(parents=True, exist_ok=True)
    for stem in subset_stems:
        try:
            orig = Image.open(out_dir / "01_original" / f"{stem}.png").convert("RGB")
            after_rm = Image.open(out_dir / "02_after_removal" / f"{stem}.png").convert("RGB")
            inp_gt = Image.open(inpainted_dir / next(
                (f"{stem}{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp")
                 if (inpainted_dir / f"{stem}{ext}").exists()),
                f"{stem}.png",
            )).convert("RGB")
            after_ft = Image.open(out_dir / "03_after_finetune" / f"{stem}.png").convert("RGB")
        except FileNotFoundError as e:
            print(f"[compose] skipping {stem}: {e}")
            continue
        # Normalize all panels to the same size.
        target = orig.size
        panels = [orig, after_rm.resize(target), inp_gt.resize(target), after_ft.resize(target)]
        compose_four_panel(*panels, out_path=panels_dir / f"{stem}.png", cols=2)

    print(f"[done] {out_dir}\n  comparisons -> {panels_dir}")


if __name__ == "__main__":
    configs = {
        "default": (
            "Inpaint-3D pipeline with DefaultStrategy densification.",
            Config(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "Inpaint-3D pipeline with MCMCStrategy densification.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cli(main, cfg, verbose=True)
