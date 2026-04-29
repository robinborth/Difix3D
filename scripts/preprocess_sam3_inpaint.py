"""Pre-process a scene's training views: produce per-view SAM3 masks and
flux-inpainted RGB images.

Outputs (parallel to data_dir/<images_subdir>):
    <data_dir>/sam3_masks/<stem>.png       L-mode binary mask, 255 = inpaint
    <data_dir>/images_inpainted/<stem>.<ext>  kontext result composited at orig res

Both folders mirror the source filenames so downstream code can lookup
masks/inpaints by image stem.

Single long-lived process: SAM3 + Qwen3 text encoder + FLUX flow + AE are
loaded once each, then we iterate views. Re-running with the same args
skips views that already have outputs (idempotent).

Usage:
    PYTHONPATH=src python scripts/preprocess_sam3_inpaint.py \
        data_dir=/cluster/angmar/rborth/difix3d/data/garden \
        view_stride=6 \
        sam3_prompt="vase on top of table" \
        inpaint_prompt="empty wooden garden table, leafy hedge background, no objects on the table"
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image, ImageFilter
from tqdm import tqdm

# Reuse the inpaint helpers from scripts/inpaint.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inpaint import (  # noqa: E402
    decode_latent_to_pil,
    grey_out,
    round_to_multiple,
    run_denoise_with_ctx,
)

from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder  # noqa: E402
from sam3 import Sam3Segmenter  # noqa: E402


def list_views(images_dir: Path, stride: int, limit: Optional[int]) -> List[Path]:
    paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    paths = paths[::max(1, stride)]
    if limit is not None:
        paths = paths[:limit]
    return paths


def dilate_feather(mask_pil: Image.Image, dilate: int, feather: int) -> Image.Image:
    if dilate > 0:
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=2 * int(dilate) + 1))
    if feather > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask_pil


@hydra.main(version_base=None, config_path="../conf", config_name="preprocess_sam3_inpaint")
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    images_dir = data_dir / cfg.images_subdir
    masks_dir = data_dir / cfg.masks_subdir
    inpaint_dir = data_dir / cfg.inpainted_subdir
    masks_dir.mkdir(parents=True, exist_ok=True)
    inpaint_dir.mkdir(parents=True, exist_ok=True)

    views = list_views(images_dir, cfg.view_stride, cfg.limit)
    if not views:
        raise SystemExit(f"No images under {images_dir}")
    print(f"[preprocess] {len(views)} views from {images_dir}", flush=True)

    info = FLUX2_MODEL_INFO[cfg.model_name.lower()]
    defaults = info.get("defaults", {})
    num_steps = cfg.num_steps if cfg.num_steps is not None else defaults["num_steps"]
    guidance = cfg.guidance if cfg.guidance is not None else defaults["guidance"]

    device = torch.device("cuda")

    # ---------------- SAM3 ----------------
    print("[preprocess] loading SAM3 ...", flush=True)
    seg = Sam3Segmenter.from_pretrained(cfg.sam3_model_id)

    print("[preprocess] running SAM3 over all views ...", flush=True)
    mask_paths_by_view: dict[Path, Path] = {}
    for vp in tqdm(views, desc="sam3"):
        mp = masks_dir / f"{vp.stem}.png"
        mask_paths_by_view[vp] = mp
        if mp.exists() and not cfg.overwrite_masks:
            continue
        img = Image.open(vp).convert("RGB")
        union, masks, boxes, scores = seg.segment(
            img,
            text=cfg.sam3_prompt,
            score_threshold=cfg.sam3_score_threshold,
            mask_threshold=cfg.sam3_mask_threshold,
        )
        Image.fromarray((union * 255).astype(np.uint8)).save(mp)

    # Free SAM3 before loading FLUX (they don't need to coexist).
    del seg
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- FLUX text encode ----------------
    print("[preprocess] loading flux text encoder ...", flush=True)
    text_encoder = load_text_encoder(cfg.model_name.lower(), device=device)
    text_encoder.eval()
    with torch.no_grad():
        if info["guidance_distilled"]:
            ctx = text_encoder([cfg.inpaint_prompt]).to(torch.bfloat16)
        else:
            empty = text_encoder([""]).to(torch.bfloat16)
            full = text_encoder([cfg.inpaint_prompt]).to(torch.bfloat16)
            ctx = torch.cat([empty, full], dim=0)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- FLUX flow + AE ----------------
    print("[preprocess] loading flux flow + AE ...", flush=True)
    model = load_flow_model(cfg.model_name.lower(), device=device)
    ae = load_ae(cfg.model_name.lower())
    model.eval()
    ae.eval()

    # ---------------- per-view kontext inpaint ----------------
    print("[preprocess] running flux kontext inpainting ...", flush=True)
    width = round_to_multiple(cfg.width)
    height = round_to_multiple(cfg.height)

    for vp in tqdm(views, desc="inpaint"):
        out_path = inpaint_dir / f"{vp.stem}{vp.suffix.lower()}"
        if out_path.exists() and not cfg.overwrite_inpaints:
            continue
        mp = mask_paths_by_view[vp]

        full_pil = Image.open(vp).convert("RGB")
        orig_size = full_pil.size  # (W, H)
        mask_full = Image.open(mp).convert("L")
        if mask_full.size != orig_size:
            mask_full = mask_full.resize(orig_size, Image.NEAREST)
        mask_full = dilate_feather(mask_full, dilate=cfg.dilate, feather=cfg.feather)
        if np.asarray(mask_full).max() == 0:
            # Nothing to inpaint — copy original through so downstream code
            # has a valid file at this stem.
            full_pil.save(out_path)
            continue

        # Run flux at (width, height); composite the result back at full res.
        base_small = full_pil.resize((width, height), Image.LANCZOS)
        mask_small = mask_full.resize((width, height), Image.LANCZOS)
        ref = grey_out(base_small, mask_small, grey=128)

        latent = run_denoise_with_ctx(
            model, ae, info, ctx, width, height, cfg.seed,
            num_steps, guidance, ref_images=[ref],
        )
        kontext_small = decode_latent_to_pil(ae, latent)
        kontext_full = kontext_small.resize(orig_size, Image.LANCZOS)
        composite_full = Image.composite(kontext_full, full_pil, mask_full)
        composite_full.save(out_path)

    print(f"[preprocess] done.\n  masks   -> {masks_dir}\n  inpaints -> {inpaint_dir}", flush=True)


if __name__ == "__main__":
    main()
