"""
FLUX.2 inpainting conditioned on a SAM3 mask (object-removal style).

Same three methods as scripts/inpaint.py:
  A) kontext         — feed the greyed-out image as a reference, prompt for the
                       new content (FLUX.2 native edit pathway).
  B) blended_latent  — RePaint / blended-latent diffusion. Synthesise inside the
                       mask, keep the original outside.
  C) pixel_composite — pure T2I + alpha-blend at the mask. Baseline.

The only thing that changes vs. scripts/inpaint.py is mask construction:
the mask is loaded from a SAM3 annotation directory produced by
`scripts/sam3_app.py` (or supplied directly via --image / --mask), instead of
being built from a bbox. Everything else is unchanged.

Annotation folder layout (matches scripts/sam3_app.py output):
    data/sam3_<stem>_<ts>/
        image.<jpg|png|webp>   the source image
        mask.png               L-mode binary mask (255 = inpaint here)
        overlay.jpg            (ignored)
        meta.json              (ignored — kept for traceability)

Usage:
    PYTHONPATH=src python scripts/inpaint_sam3.py \\
        annotation_dir=data/sam3_DSC07956_20260428_144157 \\
        inpaint_prompt="empty wooden garden table, no objects on it, leafy hedge background" \\
        method=all width=1024 height=768 dilate=12 feather=8 seed=7

Or pass image+mask directly:
    PYTHONPATH=src python scripts/inpaint_sam3.py \\
        image=data/foo.jpg mask=data/foo_mask.png inpaint_prompt="..."
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter

# Reuse helpers and stages from the bbox script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inpaint import (  # noqa: E402
    decode_latent_to_pil,
    denoise_blended,
    grey_out,
    grid,
    label,
    round_to_multiple,
    run_denoise_with_ctx,
)

from flux2.sampling import batched_prc_txt  # noqa: E402
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder  # noqa: E402


# ------------------------------- mask helpers ------------------------------


def find_image_in_dir(d: Path) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = d / f"image{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image.* found in {d}")


def load_mask(path: Path, size: tuple[int, int], dilate: int, feather: int) -> Image.Image:
    """Load a binary mask, resize to (W, H), optionally dilate then feather. White=inpaint."""
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    if dilate > 0:
        # MaxFilter with kernel = 2*dilate+1 expands white regions by `dilate` pixels.
        k = 2 * int(dilate) + 1
        m = m.filter(ImageFilter.MaxFilter(size=k))
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather))
    # Re-binarise (then keep grey edge from feather only)
    arr = np.asarray(m)
    if arr.max() == 0:
        raise ValueError(f"Mask {path} is empty after preprocessing.")
    return m


def build_overlay(img: Image.Image, mask: Image.Image,
                  color=(255, 64, 64), alpha=0.55) -> Image.Image:
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    m = np.asarray(mask).astype(np.float32) / 255.0
    tint = np.array(color, dtype=np.float32)
    out = a * (1 - alpha * m[..., None]) + tint * (alpha * m[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ------------------------------- main --------------------------------------


def main(
    inpaint_prompt: str,
    annotation_dir: Optional[str] = None,
    image: Optional[str] = None,
    mask: Optional[str] = None,
    method: str = "all",  # "kontext" | "blended_latent" | "pixel_composite" | "all"
    model_name: str = "flux.2-klein-4b",
    width: int = 1024,
    height: int = 768,
    num_steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: int = 42,
    dilate: int = 12,
    feather: int = 8,
    output_dir: Optional[str] = None,
):
    if annotation_dir:
        adir = Path(annotation_dir)
        image_path = find_image_in_dir(adir)
        mask_path = adir / "mask.png"
        assert mask_path.exists(), f"{mask_path} not found"
    else:
        assert image and mask, "Pass --annotation_dir, or both --image and --mask."
        image_path = Path(image)
        mask_path = Path(mask)

    assert model_name.lower() in FLUX2_MODEL_INFO, (
        f"{model_name} not in {list(FLUX2_MODEL_INFO.keys())}"
    )
    model_name = model_name.lower()
    info = FLUX2_MODEL_INFO[model_name]
    defaults = info.get("defaults", {})
    if num_steps is None:
        num_steps = defaults["num_steps"]
    if guidance is None:
        guidance = defaults["guidance"]

    width = round_to_multiple(width)
    height = round_to_multiple(height)

    methods = (
        ["kontext", "blended_latent", "pixel_composite"]
        if method == "all" else [method]
    )
    for m in methods:
        assert m in {"kontext", "blended_latent", "pixel_composite"}, f"unknown method {m}"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) if output_dir else Path("output") / f"inpaint_sam3_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)
    print(f"image:  {image_path}", flush=True)
    print(f"mask:   {mask_path}", flush=True)

    device = torch.device("cuda")

    # ---- Stage 1: encode prompt(s), drop encoder ----
    print("[1/4] encoding prompt...", flush=True)
    text_encoder = load_text_encoder(model_name, device=device)
    text_encoder.eval()
    with torch.no_grad():
        if info["guidance_distilled"]:
            ctx_inpaint = text_encoder([inpaint_prompt]).to(torch.bfloat16)
        else:
            ctx_empty = text_encoder([""]).to(torch.bfloat16)
            ctx_p = text_encoder([inpaint_prompt]).to(torch.bfloat16)
            ctx_inpaint = torch.cat([ctx_empty, ctx_p], dim=0)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Load flow + AE ----
    print("[2/4] loading flow + AE...", flush=True)
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name)
    model.eval()
    ae.eval()

    # ---- Stage 3: base image + mask, both at (width, height) ----
    base_pil = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    base_pil.save(out_dir / "base.png")

    mask_pil = load_mask(mask_path, (width, height), dilate=dilate, feather=feather)
    mask_pil.save(out_dir / "mask.png")
    masked_input_pil = grey_out(base_pil, mask_pil, grey=128)
    masked_input_pil.save(out_dir / "masked_input.png")
    build_overlay(base_pil, mask_pil).save(out_dir / "base_with_mask.png")

    results: dict[str, Image.Image] = {}

    # ---- Method A: kontext (reference-image edit) ----
    if "kontext" in methods:
        print("[3/4] method: kontext...", flush=True)
        ref = grey_out(base_pil, mask_pil, grey=128)
        latent = run_denoise_with_ctx(
            model, ae, info, ctx_inpaint, width, height, seed,
            num_steps, guidance, ref_images=[ref],
        )
        kontext_raw = decode_latent_to_pil(ae, latent)
        kontext_raw.save(out_dir / "kontext_raw.png")
        kontext_pil = Image.composite(kontext_raw, base_pil, mask_pil)
        kontext_pil.save(out_dir / "kontext.png")
        results["kontext"] = kontext_pil

    # ---- Method B: blended_latent ----
    if "blended_latent" in methods:
        print("[3/4] method: blended_latent (RePaint)...", flush=True)
        with torch.no_grad():
            ae_dtype = next(ae.parameters()).dtype
            base_t = (2 * T.ToTensor()(base_pil) - 1).to(ae_dtype).cuda()[None]
            base_clean_latent = ae.encode(base_t).to(torch.bfloat16)
            mask_np = np.asarray(mask_pil, dtype=np.float32) / 255.0
            mask_t = torch.from_numpy(mask_np)[None, None].cuda()
            mask_latent = F.interpolate(
                mask_t, size=(height // 16, width // 16), mode="bilinear", align_corners=False
            ).clamp(0, 1)

        ctx_b, ctx_ids = batched_prc_txt(ctx_inpaint)
        with torch.no_grad():
            x_final = denoise_blended(
                model, ctx_b, ctx_ids, info,
                base_latent=base_clean_latent,
                mask_latent=mask_latent,
                width=width, height=height,
                seed=seed, num_steps=num_steps, guidance=guidance,
            )
        blended_pil = decode_latent_to_pil(ae, x_final)
        blended_pil = Image.composite(blended_pil, base_pil, mask_pil)
        blended_pil.save(out_dir / "blended_latent.png")
        results["blended_latent"] = blended_pil

    # ---- Method C: pixel_composite ----
    if "pixel_composite" in methods:
        print("[3/4] method: pixel_composite...", flush=True)
        latent = run_denoise_with_ctx(
            model, ae, info, ctx_inpaint, width, height, seed,
            num_steps, guidance, ref_images=None,
        )
        fresh_pil = decode_latent_to_pil(ae, latent)
        composite_pil = Image.composite(fresh_pil, base_pil, mask_pil)
        composite_pil.save(out_dir / "pixel_composite.png")
        fresh_pil.save(out_dir / "pixel_composite_fresh.png")
        results["pixel_composite"] = composite_pil

    del model, ae
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Stage 4: comparison sheet ----
    print("[4/4] writing comparison + params.json...", flush=True)
    panels = [
        label(base_pil, "base"),
        label(build_overlay(base_pil, mask_pil), "mask"),
        label(masked_input_pil, "masked input"),
    ]
    for name in ["kontext", "blended_latent", "pixel_composite"]:
        if name in results:
            panels.append(label(results[name], name))
    sheet = grid(panels, cols=3)
    sheet.save(out_dir / "comparison.png")

    params = {
        "inpaint_prompt": inpaint_prompt,
        "image": str(image_path),
        "mask": str(mask_path),
        "annotation_dir": str(annotation_dir) if annotation_dir else None,
        "method": method,
        "model_name": model_name,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "guidance": guidance,
        "seed": seed,
        "dilate": dilate,
        "feather": feather,
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
    print(f"done -> {out_dir}", flush=True)


import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402


@hydra.main(version_base=None, config_path="../conf", config_name="inpaint_sam3")
def hydra_main(cfg: DictConfig) -> None:
    main(
        inpaint_prompt=cfg.inpaint_prompt,
        annotation_dir=cfg.get("annotation_dir"),
        image=cfg.get("image"),
        mask=cfg.get("mask"),
        method=cfg.method,
        model_name=cfg.model_name,
        width=cfg.width,
        height=cfg.height,
        num_steps=cfg.get("num_steps"),
        guidance=cfg.get("guidance"),
        seed=cfg.seed,
        dilate=cfg.dilate,
        feather=cfg.feather,
        output_dir=cfg.get("output_dir"),
    )


if __name__ == "__main__":
    hydra_main()
