"""
FLUX.2 depth-conditioned inference.

Given an input image and a prompt, estimate a depth map with Depth Anything V2
and feed the depth map as a reference image to FLUX.2's editing pipeline.
Outputs (depth visualization + final image) are saved to ./output/.

Note: FLUX.2 has no official depth-ControlNet. This uses FLUX.2's standard
image-editing path with the depth map acting as a geometric reference. Results
follow the depth layout to the extent the base model respects reference geometry.

Usage:
    PYTHONPATH=src python scripts/depth_inference.py \
        --input_image path/to/img.jpg \
        --prompt "an oil painting in the same composition" \
        --model_name flux.2-klein-4b
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from transformers import pipeline

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cached,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder


def estimate_depth(
    img: Image.Image,
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: str = "cuda",
) -> Image.Image:
    """Return a 3-channel RGB depth visualization (near=bright) matching input size."""
    depth_pipe = pipeline(
        task="depth-estimation",
        model=depth_model_id,
        device=device,
        torch_dtype=torch.float32,
    )
    result = depth_pipe(img)
    depth = result["predicted_depth"]  # torch.Tensor [1,H,W] or similar
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().detach().cpu().numpy()
    else:
        depth = np.array(depth, dtype=np.float32)

    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    else:
        depth_norm = (depth - dmin) / (dmax - dmin)
    depth_u8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_u8).convert("RGB").resize(img.size, Image.BILINEAR)

    # Free the depth pipeline eagerly — we only need it once.
    del depth_pipe
    torch.cuda.empty_cache()
    return depth_img


def run_flux2_with_reference(
    model_name: str,
    prompt: str,
    ref_images: list[Image.Image],
    width: int,
    height: int,
    num_steps: Optional[int],
    guidance: Optional[float],
    seed: int,
    device: torch.device,
) -> Image.Image:
    model_info = FLUX2_MODEL_INFO[model_name]
    defaults = model_info.get("defaults", {})
    if num_steps is None:
        num_steps = defaults["num_steps"]
    if guidance is None:
        guidance = defaults["guidance"]

    text_encoder = load_text_encoder(model_name, device=device)
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name)
    model.eval()
    ae.eval()
    text_encoder.eval()

    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(ae, ref_images)

        if model_info["guidance_distilled"]:
            ctx = text_encoder([prompt]).to(torch.bfloat16)
        else:
            ctx_empty = text_encoder([""]).to(torch.bfloat16)
            ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)

        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(num_steps, x.shape[1])
        if model_info["guidance_distilled"]:
            denoise_fn = (
                denoise_cached
                if (model_info.get("use_kv_cache") and ref_tokens is not None)
                else denoise
            )
            x = denoise_fn(
                model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=guidance,
                img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )
        else:
            x = denoise_cfg(
                model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=guidance,
                img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())


def round_to_multiple(x: int, m: int = 16) -> int:
    return max(m, (x // m) * m)


def _run(
    input_image: str,
    prompt: str = "a cinematic photograph with the same composition, dramatic lighting",
    model_name: str = "flux.2-klein-4b",
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: Optional[int] = None,
    output_dir: str = "output",
):
    assert model_name.lower() in FLUX2_MODEL_INFO, (
        f"{model_name} not available. Choose from: {list(FLUX2_MODEL_INFO.keys())}"
    )
    model_name = model_name.lower()

    device = torch.device("cuda")
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    src = Image.open(input_image).convert("RGB")
    if width is None or height is None:
        width = round_to_multiple(src.size[0])
        height = round_to_multiple(src.size[1])
        src = src.resize((width, height), Image.LANCZOS)
    else:
        width = round_to_multiple(width)
        height = round_to_multiple(height)
        src = src.resize((width, height), Image.LANCZOS)

    print(f"[1/3] Estimating depth with {depth_model} ...")
    depth_img = estimate_depth(src, depth_model_id=depth_model, device="cuda")

    stem = Path(input_image).stem
    depth_path = out_dir / f"{stem}_depth.png"
    depth_img.save(depth_path)
    print(f"      saved depth map -> {depth_path}")

    if seed is None:
        seed = random.randrange(2**31)
    print(f"[2/3] Running FLUX.2 '{model_name}' conditioned on depth. seed={seed}")
    result = run_flux2_with_reference(
        model_name=model_name,
        prompt=prompt,
        ref_images=[depth_img],
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        device=device,
    )

    idx = len(list(out_dir.glob(f"{stem}_gen_*.png")))
    out_path = out_dir / f"{stem}_gen_{idx:03d}_seed{seed}.png"
    result.save(out_path, quality=95, subsampling=0)
    print(f"[3/3] saved -> {out_path}")


import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../conf", config_name="depth_inference")
def main(cfg: DictConfig) -> None:
    import inspect
    raw = OmegaConf.to_container(cfg, resolve=True)
    accepted = set(inspect.signature(_run).parameters)
    _run(**{k: v for k, v in raw.items() if k in accepted})


if __name__ == "__main__":
    main()
