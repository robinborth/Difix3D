"""
FLUX.2 text-to-image ablation across the Klein family.

Runs a single fixed prompt and seed through each Klein model variant registered
in FLUX2_MODEL_INFO and writes one PNG per model to ./output/ablation_<ts>/.
Reuses run_flux2_with_reference() from scripts/depth_inference.py with an empty
ref_images list (pure T2I).

Usage:
    PYTHONPATH=src python scripts/ablation_t2i.py
    PYTHONPATH=src python scripts/ablation_t2i.py --only flux.2-klein-4b
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from flux2.sampling import (  # noqa: E402
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cached,
    denoise_cfg,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder  # noqa: E402


def generate_t2i(model_name: str, prompt: str, width: int, height: int, seed: int) -> Image.Image:
    """Memory-efficient pure T2I: encode text, free text encoder, then load flow model + AE."""
    info = FLUX2_MODEL_INFO[model_name]
    defaults = info.get("defaults", {})
    num_steps = defaults["num_steps"]
    guidance = defaults["guidance"]
    device = torch.device("cuda")

    # Stage 1: encode text, then drop the encoder.
    text_encoder = load_text_encoder(model_name, device=device)
    text_encoder.eval()
    with torch.no_grad():
        if info["guidance_distilled"]:
            ctx = text_encoder([prompt]).to(torch.bfloat16)
        else:
            ctx_empty = text_encoder([""]).to(torch.bfloat16)
            ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2: load flow model + AE and run denoising.
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name)
    model.eval()
    ae.eval()

    with torch.no_grad():
        ctx, ctx_ids = batched_prc_txt(ctx)
        shape = (1, 128, height // 16, width // 16)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(shape, generator=gen, dtype=torch.bfloat16, device="cuda")
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(num_steps, x.shape[1])
        if info["guidance_distilled"]:
            # No img_cond_seq for pure T2I, so denoise_cached path is not applicable here.
            _ = denoise_cached  # keep import used for symmetry
            x = denoise(
                model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=guidance,
                img_cond_seq=None, img_cond_seq_ids=None,
            )
        else:
            x = denoise_cfg(
                model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=guidance,
                img_cond_seq=None, img_cond_seq_ids=None,
            )

        # Free flow model before running AE decode to keep peak memory small.
        del model
        gc.collect()
        torch.cuda.empty_cache()

        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()
        del ae

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())


PROMPT = (
    "A very serious capybara CEO in a tiny pinstripe suit giving a TED talk to "
    "an audience of confused ducks wearing 3D glasses, dramatic stage lighting, "
    "wide shot, photorealistic, cinematic"
)

WIDTH = 768
HEIGHT = 768
SEED = 42

KLEIN_MODELS = [
    "flux.2-klein-4b",
    "flux.2-klein-9b",
    "flux.2-klein-9b-kv",
    "flux.2-klein-base-4b",
    "flux.2-klein-base-9b",
]


def run_one(model_name: str, out_dir: Path) -> dict:
    info = FLUX2_MODEL_INFO[model_name]
    defaults = info.get("defaults", {})
    steps = defaults.get("num_steps")
    guidance = defaults.get("guidance")

    print(f"\n=== {model_name}  (steps={steps}, guidance={guidance}) ===", flush=True)
    t0 = time.time()
    try:
        img = generate_t2i(model_name, PROMPT, WIDTH, HEIGHT, SEED)
        out_path = out_dir / f"{model_name}.png"
        img.save(out_path, quality=95, subsampling=0)
        dt = time.time() - t0
        print(f"    ok  {dt:.1f}s -> {out_path}", flush=True)
        return {
            "model": model_name,
            "status": "ok",
            "path": str(out_path),
            "steps": steps,
            "guidance": guidance,
            "seed": SEED,
            "seconds": round(dt, 2),
        }
    except (Exception, SystemExit) as e:
        dt = time.time() - t0
        err = f"{type(e).__name__}: {e}"
        print(f"    FAIL  {dt:.1f}s  {err}", flush=True)
        traceback.print_exc()
        return {
            "model": model_name,
            "status": "error",
            "error": err,
            "steps": steps,
            "guidance": guidance,
            "seed": SEED,
            "seconds": round(dt, 2),
        }
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _run(only: Optional[str] = None, output_root: str = "output"):
    models = [only] if only else KLEIN_MODELS
    for m in models:
        assert m in FLUX2_MODEL_INFO, f"unknown model {m}"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root) / f"ablation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)

    (out_dir / "prompt.txt").write_text(PROMPT + "\n")

    results = []
    for m in models:
        results.append(run_one(m, out_dir))
        with (out_dir / "results.json").open("w") as f:
            json.dump(
                {
                    "prompt": PROMPT,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "seed": SEED,
                    "runs": results,
                },
                f,
                indent=2,
            )

    print("\n=== Summary ===")
    for r in results:
        tag = "OK " if r["status"] == "ok" else "ERR"
        extra = r.get("path") or r.get("error", "")
        print(f"  [{tag}] {r['model']:24s} {r['seconds']:>6.1f}s  {extra}")


import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../conf", config_name="ablation_t2i")
def main(cfg: DictConfig) -> None:
    import inspect
    raw = OmegaConf.to_container(cfg, resolve=True)
    accepted = set(inspect.signature(_run).parameters)
    _run(**{k: v for k, v in raw.items() if k in accepted})


if __name__ == "__main__":
    main()
