"""
FLUX.2 inpainting demo with three methods.

FLUX.2 has no native mask-input inpainting head. This script layers three
standard mask-based inpainting strategies on top of the existing flow model:

  A) kontext         — feed the original image with the bbox region greyed out
                       as a reference image, prompt for the new content. Uses
                       FLUX.2's native image-edit pathway (img_cond_seq).
  B) blended_latent  — RePaint / blended-latent diffusion. At every denoise step
                       replace the unmasked region of the model's latent with
                       the AE-encoded original forward-noised to the current
                       timestep. Masked region is fully synthesized.
  C) pixel_composite — pure T2I with the new prompt, then alpha-blend into the
                       original at the bbox using a feathered mask. Baseline.

End-to-end pipeline:
  1. generate base image from --base_prompt
  2. construct a bbox mask
  3. run each requested method
  4. write all visuals (base, mask, masked input, per-method results,
     side-by-side comparison) + params.json to outputs/inpaint_<ts>/

Usage:
    PYTHONPATH=src python scripts/inpaint.py \
        --base_prompt "a wooden desk with a red apple, soft window light" \
        --inpaint_prompt "a small potted cactus" \
        --bbox "300,260,520,500" \
        --method all \
        --width 768 --height 768 \
        --seed 7
"""

from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder


# ------------------------------- helpers -----------------------------------


def round_to_multiple(x: int, m: int = 16) -> int:
    return max(m, (x // m) * m)


def parse_bbox(bbox: str | tuple, width: int, height: int) -> tuple[int, int, int, int]:
    if isinstance(bbox, (tuple, list)):
        x1, y1, x2, y2 = bbox
    else:
        parts = [p.strip() for p in str(bbox).split(",")]
        x1, y1, x2, y2 = (int(p) for p in parts)
    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox is empty after clamping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def build_pixel_mask(width: int, height: int, bbox, feather: int) -> Image.Image:
    """White (255) inside the bbox, black outside. Optional Gaussian feather."""
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle(bbox, fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask


def grey_out(img: Image.Image, mask: Image.Image, grey: int = 128) -> Image.Image:
    """Replace mask>0 region with neutral grey."""
    grey_img = Image.new("RGB", img.size, (grey, grey, grey))
    return Image.composite(grey_img, img, mask)


def label(img: Image.Image, text: str) -> Image.Image:
    """Add a small text label at the top-left for the contact sheet."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def grid(images: list[Image.Image], cols: int, pad: int = 8, bg=(20, 20, 20)) -> Image.Image:
    if not images:
        raise ValueError("empty images")
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad) + (w - im.width) // 2
        y = pad + r * (h + pad) + (h - im.height) // 2
        out.paste(im, (x, y))
    return out


def decode_latent_to_pil(ae, x_latent: torch.Tensor) -> Image.Image:
    """x_latent: (1, 128, h, w) bfloat16 -> PIL image."""
    with torch.no_grad():
        x = ae.decode(x_latent).float()
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())


# ------------------------------- core stages -------------------------------


def run_denoise_with_ctx(
    model,
    ae,
    info: dict,
    ctx: torch.Tensor,
    width: int,
    height: int,
    seed: int,
    num_steps: int,
    guidance: float,
    ref_images: Optional[list[Image.Image]] = None,
) -> torch.Tensor:
    """Run denoising and return the *latent* (1, 128, h/16, w/16) — caller decodes."""
    with torch.no_grad():
        if ref_images:
            ref_tokens, ref_ids = encode_image_refs(ae, ref_images)
        else:
            ref_tokens, ref_ids = None, None

        ctx_b, ctx_ids = batched_prc_txt(ctx)
        shape = (1, 128, height // 16, width // 16)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(shape, generator=gen, dtype=torch.bfloat16, device="cuda")
        x, x_ids = batched_prc_img(randn)
        timesteps = get_schedule(num_steps, x.shape[1])

        if info["guidance_distilled"]:
            x = denoise(
                model, x, x_ids, ctx_b, ctx_ids,
                timesteps=timesteps, guidance=guidance,
                img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )
        else:
            x = denoise_cfg(
                model, x, x_ids, ctx_b, ctx_ids,
                timesteps=timesteps, guidance=guidance,
                img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)  # (1, 128, h, w)
    return x


def denoise_blended(
    model,
    ctx_b: torch.Tensor,
    ctx_ids: torch.Tensor,
    info: dict,
    base_latent: torch.Tensor,         # (1, 128, h, w) — clean latent of the original image
    mask_latent: torch.Tensor,         # (1, 1, h, w) in [0,1], 1 = inpaint here
    width: int,
    height: int,
    seed: int,
    num_steps: int,
    guidance: float,
):
    """Blended-latent diffusion. mask_latent==1 -> synthesize, ==0 -> keep original."""
    shape = (1, 128, height // 16, width // 16)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    randn = torch.randn(shape, generator=gen, dtype=torch.bfloat16, device="cuda")
    x, x_ids = batched_prc_img(randn)

    timesteps = get_schedule(num_steps, x.shape[1])
    guidance_distilled = info["guidance_distilled"]

    # Bring base/mask into token-flat form so we can blend in token space.
    # batched_prc_img flattens (c,h,w) as (h w) c with t/h/w/l ids; do the same to base.
    base_flat, base_ids = batched_prc_img(base_latent.to(torch.bfloat16))
    # mask: broadcast over channels by repeating to 128 then prc with same shape.
    mask_full = mask_latent.expand(-1, 128, -1, -1).to(torch.bfloat16)
    mask_flat, _ = batched_prc_img(mask_full)
    # All three (x, base_flat, mask_flat) share the same x_ids token ordering since
    # batched_prc_img is deterministic for identical shapes.

    if guidance_distilled:
        guidance_vec = torch.full((x.shape[0],), guidance, device=x.device, dtype=x.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)

            # Forward-noise the clean base to t_curr in flow form: x_t = (1-t)*z + t*eps
            eps = torch.randn(base_flat.shape, generator=gen, dtype=torch.bfloat16, device="cuda")
            base_at_t = (1.0 - t_curr) * base_flat + t_curr * eps

            # Composite: keep base outside mask, keep model x inside mask.
            x = mask_flat * x + (1.0 - mask_flat) * base_at_t

            pred = model(
                x=x, x_ids=x_ids, timesteps=t_vec,
                ctx=ctx_b, ctx_ids=ctx_ids, guidance=guidance_vec,
            )
            x = x + (t_prev - t_curr) * pred
        # Final blend so the unmasked region is exactly the clean base.
        x = mask_flat * x + (1.0 - mask_flat) * base_flat
    else:
        # CFG path: duplicate batch.
        x = torch.cat([x, x], dim=0)
        x_ids_d = torch.cat([x_ids, x_ids], dim=0)
        base_flat_d = torch.cat([base_flat, base_flat], dim=0)
        mask_flat_d = torch.cat([mask_flat, mask_flat], dim=0)

        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
            eps = torch.randn(base_flat_d.shape, generator=gen, dtype=torch.bfloat16, device="cuda")
            base_at_t = (1.0 - t_curr) * base_flat_d + t_curr * eps
            x = mask_flat_d * x + (1.0 - mask_flat_d) * base_at_t

            pred = model(
                x=x, x_ids=x_ids_d, timesteps=t_vec,
                ctx=ctx_b, ctx_ids=ctx_ids, guidance=None,
            )
            pred_u, pred_c = pred.chunk(2)
            pred = pred_u + guidance * (pred_c - pred_u)
            pred = torch.cat([pred, pred], dim=0)
            x = x + (t_prev - t_curr) * pred
        x = x.chunk(2)[0]
        x = mask_flat * x + (1.0 - mask_flat) * base_flat
        x_ids = x_ids  # unchanged

    x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
    return x


# ------------------------------- main --------------------------------------


def _run(
    base_prompt: str,
    inpaint_prompt: str,
    bbox: str,
    base_image: Optional[str] = None,
    method: str = "all",          # "kontext" | "blended_latent" | "pixel_composite" | "all"
    model_name: str = "flux.2-klein-4b",
    width: int = 768,
    height: int = 768,
    num_steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: int = 42,
    feather: int = 16,
    output_dir: Optional[str] = None,
):
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
    out_dir = Path(output_dir) if output_dir else Path("outputs") / f"inpaint_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)

    device = torch.device("cuda")

    # ---- Stage 1: encode all prompts up-front, then drop the text encoder ----
    print("[1/5] encoding prompts...", flush=True)
    text_encoder = load_text_encoder(model_name, device=device)
    text_encoder.eval()
    with torch.no_grad():
        if info["guidance_distilled"]:
            ctx_base = text_encoder([base_prompt]).to(torch.bfloat16)
            ctx_inpaint = text_encoder([inpaint_prompt]).to(torch.bfloat16)
        else:
            ctx_empty = text_encoder([""]).to(torch.bfloat16)
            ctx_base_p = text_encoder([base_prompt]).to(torch.bfloat16)
            ctx_inpaint_p = text_encoder([inpaint_prompt]).to(torch.bfloat16)
            ctx_base = torch.cat([ctx_empty, ctx_base_p], dim=0)
            ctx_inpaint = torch.cat([ctx_empty, ctx_inpaint_p], dim=0)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Load flow model + AE (kept alive across stages) ----
    print("[2/5] loading flow + AE...", flush=True)
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name)
    model.eval()
    ae.eval()

    # ---- Stage 2: base image ----
    if base_image:
        print(f"[3/5] using provided base image {base_image}", flush=True)
        base_pil = Image.open(base_image).convert("RGB").resize((width, height), Image.LANCZOS)
    else:
        print(f"[3/5] generating base image (steps={num_steps}, guidance={guidance}, seed={seed})...",
              flush=True)
        base_latent = run_denoise_with_ctx(
            model, ae, info, ctx_base, width, height, seed, num_steps, guidance,
            ref_images=None,
        )
        base_pil = decode_latent_to_pil(ae, base_latent)

    base_pil.save(out_dir / "base.png")

    # ---- Mask construction ----
    bbox_t = parse_bbox(bbox, width, height)
    mask_pil = build_pixel_mask(width, height, bbox_t, feather=feather)
    mask_pil.save(out_dir / "mask.png")
    masked_input_pil = grey_out(base_pil, mask_pil, grey=128)
    masked_input_pil.save(out_dir / "masked_input.png")

    # Also save a base + bbox overlay for quick visual check.
    overlay = base_pil.copy()
    ImageDraw.Draw(overlay).rectangle(bbox_t, outline=(255, 0, 0), width=4)
    overlay.save(out_dir / "base_with_bbox.png")

    results: dict[str, Image.Image] = {}

    # ---- Method A: kontext (reference-image edit) ----
    if "kontext" in methods:
        print("[4/5] method: kontext (reference-image edit)...", flush=True)
        # Use the greyed-out image as the reference. Augment the prompt to point at the hole.
        ref = grey_out(base_pil, mask_pil, grey=128)
        latent = run_denoise_with_ctx(
            model, ae, info, ctx_inpaint, width, height, seed,
            num_steps, guidance, ref_images=[ref],
        )
        kontext_raw = decode_latent_to_pil(ae, latent)
        kontext_raw.save(out_dir / "kontext_raw.png")
        # Re-composite outside the mask so global luminance shifts don't bleed.
        kontext_pil = Image.composite(kontext_raw, base_pil, mask_pil)
        kontext_pil.save(out_dir / "kontext.png")
        results["kontext"] = kontext_pil

    # ---- Method B: blended_latent ----
    if "blended_latent" in methods:
        print("[4/5] method: blended_latent (RePaint)...", flush=True)
        with torch.no_grad():
            base_t = (2 * T.ToTensor()(base_pil) - 1).to(torch.bfloat16).cuda()[None]
            base_clean_latent = ae.encode(base_t)  # (1, 128, h/16, w/16)
            mask_np = np.asarray(mask_pil, dtype=np.float32) / 255.0
            mask_t = torch.from_numpy(mask_np)[None, None].cuda()
            mask_latent = F.interpolate(
                mask_t, size=(height // 16, width // 16), mode="bilinear", align_corners=False
            ).clamp(0, 1)

        ctx_b, ctx_ids = batched_prc_txt(ctx_inpaint)
        x_final = denoise_blended(
            model, ctx_b, ctx_ids, info,
            base_latent=base_clean_latent,
            mask_latent=mask_latent,
            width=width, height=height,
            seed=seed, num_steps=num_steps, guidance=guidance,
        )
        blended_pil = decode_latent_to_pil(ae, x_final)
        # Pixel-space re-composite for crisp boundary.
        blended_pil = Image.composite(blended_pil, base_pil, mask_pil)
        blended_pil.save(out_dir / "blended_latent.png")
        results["blended_latent"] = blended_pil

    # ---- Method C: pixel_composite ----
    if "pixel_composite" in methods:
        print("[4/5] method: pixel_composite (T2I + alpha blend)...", flush=True)
        latent = run_denoise_with_ctx(
            model, ae, info, ctx_inpaint, width, height, seed,
            num_steps, guidance, ref_images=None,
        )
        fresh_pil = decode_latent_to_pil(ae, latent)
        composite_pil = Image.composite(fresh_pil, base_pil, mask_pil)
        composite_pil.save(out_dir / "pixel_composite.png")
        # Save the unblended fresh T2I too for reference.
        fresh_pil.save(out_dir / "pixel_composite_fresh.png")
        results["pixel_composite"] = composite_pil

    # Free heavy models before contact sheet.
    del model, ae
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Stage 5: contact sheet ----
    print("[5/5] writing comparison + params.json...", flush=True)
    panels = [
        label(base_pil, "base"),
        label(overlay, "bbox"),
        label(masked_input_pil, "masked input"),
    ]
    for name in ["kontext", "blended_latent", "pixel_composite"]:
        if name in results:
            panels.append(label(results[name], name))
    cols = 3
    sheet = grid(panels, cols=cols)
    sheet.save(out_dir / "comparison.png")

    params = {
        "base_prompt": base_prompt,
        "inpaint_prompt": inpaint_prompt,
        "bbox": list(bbox_t),
        "method": method,
        "model_name": model_name,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "guidance": guidance,
        "seed": seed,
        "feather": feather,
        "base_image": base_image,
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
    print(f"done -> {out_dir}", flush=True)


import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../conf", config_name="inpaint")
def main(cfg: DictConfig) -> None:
    import inspect
    raw = OmegaConf.to_container(cfg, resolve=True)
    accepted = set(inspect.signature(_run).parameters)
    _run(**{k: v for k, v in raw.items() if k in accepted})


if __name__ == "__main__":
    main()
