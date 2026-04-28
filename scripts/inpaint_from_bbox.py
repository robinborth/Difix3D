"""
Inpaint a bbox-annotated folder produced by `scripts/bbox_annotator.py`.

Input folder layout (default written under `data/bbox_<ts>/`):
    image.png          original at full resolution
    mask.png           L-mode, 255 inside the bbox, 0 outside
    metadata.json      {bbox, prompt, source_image, width, height, saved_at}

Pipeline:
  1. resize image+mask to a model-friendly working resolution (long side
     = --max_side, rounded to /16)
  2. encode prompt; drop text encoder
  3. run inpainting via FLUX.2 (default method: kontext = native image-edit
     pathway with the greyed-out image as a reference)
  4. recomposite outside the mask at working res
  5. upsample back to original resolution and recomposite once more so
     the unmasked region is byte-identical to the input

Output is written to `output/inpaint_<input_stem>_<ts>/` and includes a
labeled `comparison.png` contact sheet plus `params.json`.

Usage:
    /rhome/rborth/miniconda3/envs/flux2-depth/bin/python \
        scripts/inpaint_from_bbox.py \
        --input_dir data/bbox_20260428_144217 \
        --method all --max_side 1024 --seed 7
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter

# scripts/ is not a package — add it to sys.path so we can import inpaint helpers.
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
from flux2.util import (  # noqa: E402
    FLUX2_MODEL_INFO,
    load_ae,
    load_flow_model,
    load_text_encoder,
)


METHODS = ("kontext", "blended_latent", "pixel_composite")


def resize_for_model(img: Image.Image, mask: Image.Image, max_side: int):
    """Long side = max_side, both dims rounded down to multiples of 16."""
    w0, h0 = img.size
    s = max_side / max(w0, h0)
    w = round_to_multiple(int(round(w0 * s)))
    h = round_to_multiple(int(round(h0 * s)))
    img_r = img.resize((w, h), Image.LANCZOS)
    mask_r = mask.resize((w, h), Image.NEAREST)
    return img_r, mask_r, (w, h)


def feathered(mask: Image.Image, feather: int) -> Image.Image:
    if feather <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(radius=feather))


def thumbnail(img: Image.Image, max_side: int = 512) -> Image.Image:
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.LANCZOS)
    return out


def _run(
    input_dir: str,
    output_dir: Optional[str] = None,
    method: str = "kontext",
    model_name: str = "flux.2-klein-4b",
    max_side: int = 1024,
    feather: int = 8,
    num_steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: int = 42,
    prompt: Optional[str] = None,
):
    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"input_dir not found: {in_dir}")
    img_path = in_dir / "image.png"
    mask_path = in_dir / "mask.png"
    meta_path = in_dir / "metadata.json"
    for p in (img_path, mask_path, meta_path):
        if not p.exists():
            raise SystemExit(f"missing {p.name} in {in_dir}")
    metadata = json.loads(meta_path.read_text())

    model_name = model_name.lower()
    assert model_name in FLUX2_MODEL_INFO, f"{model_name} not in {list(FLUX2_MODEL_INFO)}"
    info = FLUX2_MODEL_INFO[model_name]
    defaults = info.get("defaults", {})
    if num_steps is None:
        num_steps = defaults["num_steps"]
    if guidance is None:
        guidance = defaults["guidance"]

    methods = list(METHODS) if method == "all" else [method]
    for m in methods:
        assert m in METHODS, f"unknown method {m}; choose from {METHODS} or 'all'"

    inpaint_prompt = prompt if prompt is not None else metadata.get("prompt", "")
    if not inpaint_prompt:
        print("WARN: empty prompt — model will see ''", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) if output_dir else Path("output") / f"inpaint_{in_dir.name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)

    base_full = Image.open(img_path).convert("RGB")
    mask_full = Image.open(mask_path).convert("L")
    if base_full.size != mask_full.size:
        raise SystemExit(f"image/mask size mismatch: {base_full.size} vs {mask_full.size}")
    W0, H0 = base_full.size

    # Visualizations at full res.
    base_full.save(out_dir / "input.png")
    mask_full.save(out_dir / "mask.png")
    bbox_overlay = base_full.copy()
    if "bbox" in metadata:
        bx = tuple(int(v) for v in metadata["bbox"])
        ImageDraw.Draw(bbox_overlay).rectangle(bx, outline=(255, 0, 0), width=max(2, W0 // 400))
    bbox_overlay.save(out_dir / "bbox_overlay.png")
    grey_out(base_full, mask_full, grey=128).save(out_dir / "masked_input.png")

    # Working-resolution copies.
    base_w, mask_w, (W, H) = resize_for_model(base_full, mask_full, max_side)
    mask_w_feather = feathered(mask_w, feather)
    print(f"working res: {W}x{H}  (orig {W0}x{H0})", flush=True)

    device = torch.device("cuda")

    # ---- Stage 1: text encoder ----
    print("[1/3] encoding prompt...", flush=True)
    text_encoder = load_text_encoder(model_name, device=device)
    text_encoder.eval()
    with torch.no_grad():
        if info["guidance_distilled"]:
            ctx = text_encoder([inpaint_prompt]).to(torch.bfloat16)
        else:
            ctx_empty = text_encoder([""]).to(torch.bfloat16)
            ctx_p = text_encoder([inpaint_prompt]).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_p], dim=0)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Stage 2: flow + AE ----
    print("[2/3] loading flow + AE...", flush=True)
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name)
    model.eval()
    ae.eval()

    results: dict[str, Image.Image] = {}
    working_res_results: dict[str, Image.Image] = {}

    def _free():
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Method A: kontext ----
    if "kontext" in methods:
        print("[3/3] method: kontext...", flush=True)
        ref = grey_out(base_w, mask_w, grey=128)
        latent = run_denoise_with_ctx(
            model, ae, info, ctx, W, H, seed, num_steps, guidance,
            ref_images=[ref],
        )
        raw = decode_latent_to_pil(ae, latent)
        composited = Image.composite(raw, base_w, mask_w_feather)
        working_res_results["kontext"] = composited
        composited.save(out_dir / "kontext_working_res.png")
        del latent, raw
        _free()

    # ---- Method B: blended_latent ----
    if "blended_latent" in methods:
        print("[3/3] method: blended_latent...", flush=True)
        with torch.no_grad():
            base_t = (2 * T.ToTensor()(base_w) - 1).cuda()[None]  # fp32 for AE encoder
            base_clean_latent = ae.encode(base_t).to(torch.bfloat16)
            del base_t
            mask_t = (T.ToTensor()(mask_w))[None].to(torch.float32).cuda()
            mask_latent = F.interpolate(
                mask_t, size=(H // 16, W // 16), mode="bilinear", align_corners=False,
            ).clamp(0, 1)
            del mask_t
        _free()
        ctx_b, ctx_ids = batched_prc_txt(ctx)
        x_final = denoise_blended(
            model, ctx_b, ctx_ids, info,
            base_latent=base_clean_latent,
            mask_latent=mask_latent,
            width=W, height=H,
            seed=seed, num_steps=num_steps, guidance=guidance,
        )
        del base_clean_latent, mask_latent, ctx_b, ctx_ids
        _free()
        raw = decode_latent_to_pil(ae, x_final)
        del x_final
        _free()
        composited = Image.composite(raw, base_w, mask_w_feather)
        working_res_results["blended_latent"] = composited
        composited.save(out_dir / "blended_latent_working_res.png")

    # ---- Method C: pixel_composite ----
    if "pixel_composite" in methods:
        print("[3/3] method: pixel_composite...", flush=True)
        latent = run_denoise_with_ctx(
            model, ae, info, ctx, W, H, seed, num_steps, guidance,
            ref_images=None,
        )
        raw = decode_latent_to_pil(ae, latent)
        composited = Image.composite(raw, base_w, mask_w_feather)
        working_res_results["pixel_composite"] = composited
        composited.save(out_dir / "pixel_composite_working_res.png")
        del latent, raw
        _free()

    del model, ae
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Upsample back to original res, recomposite with original mask ----
    mask_full_feather = feathered(mask_full, feather)
    for name, im_w in working_res_results.items():
        upsampled = im_w.resize((W0, H0), Image.LANCZOS)
        final = Image.composite(upsampled, base_full, mask_full_feather)
        final.save(out_dir / f"{name}.png")
        results[name] = final

    # ---- Contact sheet ----
    print("writing comparison...", flush=True)
    panels = [
        label(thumbnail(base_full), "input"),
        label(thumbnail(grey_out(base_full, mask_full, grey=128)), "masked input"),
        label(thumbnail(bbox_overlay), "bbox"),
    ]
    for name in METHODS:
        if name in results:
            panels.append(label(thumbnail(results[name]), name))
    grid(panels, cols=3).save(out_dir / "comparison.png")

    params = {
        "input_dir": str(in_dir.resolve()),
        "metadata": metadata,
        "prompt_used": inpaint_prompt,
        "method": method,
        "model_name": model_name,
        "max_side": max_side,
        "working_res": [W, H],
        "original_res": [W0, H0],
        "feather": feather,
        "num_steps": num_steps,
        "guidance": guidance,
        "seed": seed,
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
    print(f"done -> {out_dir}", flush=True)


import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../conf", config_name="inpaint_from_bbox")
def main(cfg: DictConfig) -> None:
    import inspect
    raw = OmegaConf.to_container(cfg, resolve=True)
    accepted = set(inspect.signature(_run).parameters)
    _run(**{k: v for k, v in raw.items() if k in accepted})


if __name__ == "__main__":
    main()
