"""
Small overfitting experiment: depth-conditioned FLUX.2.

Takes a single (image, depth) pair and overfits a subset of FLUX.2's transformer
on the rectified-flow loss with the depth latent prepended as a reference token
sequence (the same conditioning path used by scripts/depth_inference.py at
inference time). Periodically samples to disk so you can watch the model
collapse onto the target.

Memory plan (A6000, 48GB):
  1. Encode prompt (and empty prompt for CFG models) with Qwen → free Qwen.
  2. Encode target image + depth ref with the AE → keep AE on GPU for sampling.
  3. Train transformer with a *subset* of params trainable (default:
     `img_in` + `final_layer` + first double block).

Usage:
    PYTHONPATH=src python scripts/depth_overfit.py \\
        --input_image path/to/img.jpg \\
        --prompt "a cinematic photograph, dramatic lighting" \\
        --steps 200 \\
        --sample_every 50
"""
from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

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
from scripts.depth_inference import estimate_depth, round_to_multiple


TRAINABLE_PRESETS = {
    # name -> list of attribute paths (dot-separated) on the Flux2 module to unfreeze
    "minimal": ["img_in", "final_layer"],
    "small": ["img_in", "final_layer", "double_blocks.0"],
    "double_blocks": ["img_in", "final_layer", "double_blocks"],
    "all": [""],  # everything
}


def _set_trainable(model: torch.nn.Module, preset: str) -> list[torch.nn.Parameter]:
    paths = TRAINABLE_PRESETS[preset]
    for p in model.parameters():
        p.requires_grad_(False)

    trainable: list[torch.nn.Parameter] = []
    for path in paths:
        mod: torch.nn.Module = model
        if path:
            for part in path.split("."):
                mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]  # type: ignore
        for p in mod.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    # Deduplicate by id (in case of overlap)
    seen, unique = set(), []
    for p in trainable:
        if id(p) not in seen:
            seen.add(id(p))
            unique.append(p)
    n = sum(p.numel() for p in unique)
    print(f"[trainable] preset={preset!r}, {len(unique)} tensors, {n/1e6:.2f}M params")
    return unique


@torch.no_grad()
def _sample(
    model,
    ae,
    model_info,
    ctx_for_sampling,
    ctx_ids_for_sampling,
    ref_tokens,
    ref_ids,
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
    seed: int,
) -> Image.Image:
    model.eval()
    shape = (1, 128, height // 16, width // 16)
    g = torch.Generator(device="cuda").manual_seed(seed)
    randn = torch.randn(shape, generator=g, dtype=torch.bfloat16, device="cuda")
    x, x_ids = batched_prc_img(randn)
    timesteps = get_schedule(num_steps, x.shape[1])

    if model_info["guidance_distilled"]:
        denoise_fn = (
            denoise_cached
            if (model_info.get("use_kv_cache") and ref_tokens is not None)
            else denoise
        )
        x = denoise_fn(
            model, x, x_ids, ctx_for_sampling, ctx_ids_for_sampling,
            timesteps=timesteps, guidance=guidance,
            img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
        )
    else:
        x = denoise_cfg(
            model, x, x_ids, ctx_for_sampling, ctx_ids_for_sampling,
            timesteps=timesteps, guidance=guidance,
            img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
        )
    x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
    img = ae.decode(x).float().clamp(-1, 1)
    img = rearrange(img[0], "c h w -> h w c")
    model.train()
    return Image.fromarray((127.5 * (img + 1.0)).cpu().byte().numpy())


def _run(
    input_image: str,
    prompt: str = "a cinematic photograph with the same composition, dramatic lighting",
    model_name: str = "flux.2-klein-4b",
    steps: int = 200,
    lr: float = 1e-4,
    batch_size: int = 1,
    trainable: str = "small",
    sample_every: int = 50,
    sample_steps: Optional[int] = None,
    sample_guidance: Optional[float] = None,
    sample_seed: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: int = 1234,
    output_dir: str = "output/depth_overfit",
):
    assert model_name.lower() in FLUX2_MODEL_INFO, (
        f"{model_name} not in {list(FLUX2_MODEL_INFO)}"
    )
    assert trainable in TRAINABLE_PRESETS, (
        f"--trainable must be one of {list(TRAINABLE_PRESETS)}"
    )
    model_name = model_name.lower()
    model_info = FLUX2_MODEL_INFO[model_name]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(seed)

    # --- 1. load image, estimate depth ---
    src = Image.open(input_image).convert("RGB")
    if width is None or height is None:
        width = round_to_multiple(src.size[0])
        height = round_to_multiple(src.size[1])
    else:
        width = round_to_multiple(width)
        height = round_to_multiple(height)
    src = src.resize((width, height), Image.LANCZOS)
    print(f"[1/4] image={width}x{height}")

    depth_img = estimate_depth(src, device="cuda")
    src.save(out_dir / "target.png")
    depth_img.save(out_dir / "depth.png")

    # --- 2. encode text once, free Qwen ---
    print("[2/4] encoding text")
    text_encoder = load_text_encoder(model_name, device=device)
    with torch.no_grad():
        if model_info["guidance_distilled"]:
            ctx_train = text_encoder([prompt]).to(torch.bfloat16)
            ctx_sample = ctx_train.clone()
        else:
            ctx_empty = text_encoder([""]).to(torch.bfloat16)
            ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
            ctx_train = ctx_prompt.clone()
            ctx_sample = torch.cat([ctx_empty, ctx_prompt], dim=0)
    ctx_train, ctx_train_ids = batched_prc_txt(ctx_train)
    ctx_sample, ctx_sample_ids = batched_prc_txt(ctx_sample)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # --- 3. encode target + depth ref with AE (kept on GPU for sampling) ---
    print("[3/4] encoding target + depth latents")
    ae = load_ae(model_name, device=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(ae, [depth_img])
        # encode target as a "reference" too, then unflatten to a (1, 128, h/16, w/16) latent
        from flux2.sampling import default_prep
        tgt_tensor = default_prep(src, limit_pixels=None)[None].to(device)
        tgt_latent = ae.encode(tgt_tensor).to(torch.bfloat16)  # (1, 128, h/16, w/16)
    print(f"      target latent {tuple(tgt_latent.shape)}, ref tokens {tuple(ref_tokens.shape)}")

    # Tokenize the target latent the same way batched_prc_img does, so it matches
    # what the transformer expects to predict.
    x0_tokens, x0_ids = batched_prc_img(tgt_latent)  # (1, L_img, 128), ids (1, L_img, 4)

    # --- 4. load transformer, set trainable subset, train ---
    print("[4/4] loading transformer")
    model = load_flow_model(model_name, device=device)
    model.train()
    trainable_params = _set_trainable(model, trainable)
    optim = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    # Klein models are guidance-distilled; pass guidance=1.0 during training when used.
    use_guidance = bool(getattr(model, "use_guidance_embed", False))
    g_train = (
        torch.full((batch_size,), 1.0, device=device, dtype=torch.bfloat16)
        if use_guidance else None
    )

    if sample_steps is None:
        sample_steps = model_info["defaults"]["num_steps"]
    if sample_guidance is None:
        sample_guidance = float(model_info["defaults"]["guidance"])

    losses: list[float] = []
    for step in range(1, steps + 1):
        # Rectified-flow target: with x_t = (1-t) x0 + t * noise,
        # dx_t/dt = noise - x0, and the denoise loop applies
        # x <- x + (t_prev - t_curr) * pred. So pred should match (noise - x0).
        t = torch.rand(batch_size, device=device, dtype=torch.bfloat16)
        noise = torch.randn_like(x0_tokens)
        t_b = t.view(batch_size, 1, 1)
        x_t = (1 - t_b) * x0_tokens + t_b * noise
        target = noise - x0_tokens

        # Prepend depth ref tokens (batch=1 → tile if needed).
        ref = ref_tokens.expand(batch_size, -1, -1)
        ref_id = ref_ids.expand(batch_size, -1, -1)
        x_in = torch.cat([x_t, ref], dim=1)
        x_in_ids = torch.cat([x0_ids, ref_id], dim=1)

        pred = model(
            x=x_in, x_ids=x_in_ids,
            timesteps=t.float(),
            ctx=ctx_train.expand(batch_size, -1, -1),
            ctx_ids=ctx_train_ids.expand(batch_size, -1, -1),
            guidance=g_train,
        )
        pred = pred[:, : x_t.shape[1]]
        loss = F.mse_loss(pred.float(), target.float())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if step % 10 == 0 or step == 1:
            mean_recent = sum(losses[-10:]) / min(10, len(losses))
            print(f"step {step:4d} | loss {loss.item():.4f} | mean10 {mean_recent:.4f}")

        if sample_every > 0 and (step % sample_every == 0 or step == steps):
            print(f"      sampling @ step {step}")
            img = _sample(
                model, ae, model_info,
                ctx_sample, ctx_sample_ids,
                ref_tokens, ref_ids,
                height=height, width=width,
                num_steps=sample_steps, guidance=sample_guidance,
                seed=sample_seed,
            )
            img.save(out_dir / f"sample_step{step:04d}.png")

    # final loss curve
    (out_dir / "loss.txt").write_text("\n".join(f"{i+1}\t{l}" for i, l in enumerate(losses)))
    print(f"done. outputs in {out_dir}/")


import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../conf", config_name="depth_overfit")
def main(cfg: DictConfig) -> None:
    import inspect
    raw = OmegaConf.to_container(cfg, resolve=True)
    accepted = set(inspect.signature(_run).parameters)
    _run(**{k: v for k, v in raw.items() if k in accepted})


if __name__ == "__main__":
    main()
