# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Data directory

Store all large/auxiliary data for this repo (datasets, downloaded scenes, checkpoints, training outputs, etc.) under:

```
/cluster/angmar/rborth/difix3d
```

Do not commit large data into the repo or write it to the working tree. When a script wants a `data_dir`, `--ckpt`, `result_dir`, or similar, point it at a path inside `/cluster/angmar/rborth/difix3d` (creating subfolders as needed).

## Repository purpose

Difix3D+ (CVPR 2025) — a single-step diffusion model ("Difix") that removes artifacts from renders of 3D reconstructions, plus integrations that use it to progressively refine NeRF (nerfstudio) and Gaussian Splatting (gsplat) models.

## Setup

```bash
pip install -r requirements.txt
# nerfstudio integration:
cd examples/nerfstudio && pip install -e . && cd ../..
# gsplat: install per upstream instructions, then use examples/gsplat/.
```

`Difix` downloads `stabilityai/sd-turbo` weights from Hugging Face on first use. Pretrained Difix checkpoints are expected as `checkpoints/model*.pkl` (loaded via `--model_path`).

## Common commands

Training (single GPU; multi-GPU adds `--multi_gpu --num_processes N`):
```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
  --output_dir=./outputs/difix/train --dataset_path=data/data.json \
  --max_train_steps 10000 --resolution=512 --learning_rate 2e-5 \
  --train_batch_size=1 --enable_xformers_memory_efficient_attention \
  --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
  --timestep 199
```

Inference (single image, directory, or video frames):
```bash
python src/inference_difix.py --model_path checkpoints/model.pkl \
  --input_image PATH --prompt "remove degradation" \
  --output_dir outputs/difix --timestep 199
# Add --ref_image PATH to use the reference-conditioned multi-view variant.
# Add --video to encode the outputs into output.mp4.
```

Quickstart via diffusers (no local checkpoint needed):
```python
from pipeline_difix import DifixPipeline
pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True).to("cuda")
# or "nvidia/difix_ref" for the reference-image variant.
pipe(prompt, image=img, num_inference_steps=1, timesteps=[199], guidance_scale=0.0)
```

3D refinement:
```bash
# nerfacto (registered as method `difix3d` via the entry point in examples/nerfstudio/pyproject.toml)
ns-train difix3d --data DATA --load-checkpoint CKPT ... nerfstudio-data ...

# gsplat
python examples/gsplat/simple_trainer_difix3d.py default \
  --data_dir DATA --result_dir OUT --no-normalize-world-space --test_every 1 --ckpt CKPT
```

There is no test suite, linter config, or CI in this repo.

## Architecture

The project has two layers worth understanding separately.

### 1. Difix — the 2D artifact-removal model (`src/`)

`Difix` (`src/model.py`) wraps `stabilityai/sd-turbo` into a single-step image-to-image fixer:

- **One-step DDPM scheduler.** `make_1step_sched()` builds a `DDPMScheduler` with `set_timesteps(1)`. Forward calls the UNet once at a fixed `self.timesteps` (default 199), then `sched.step` to recover `prev_sample` directly — no iterative denoising.
- **VAE with skip connections.** `my_vae_encoder_fwd` / `my_vae_decoder_fwd` are monkey-patched onto the SD-turbo VAE. The encoder stashes intermediate down-block activations in `current_down_blocks`; the decoder consumes them via `incoming_skip_acts` through four learned `skip_conv_*` 1×1 convs scaled by `decoder.gamma`. This is what lets a generative VAE preserve fine input detail.
- **LoRA-only training.** UNet and VAE-decoder are adapted with PEFT LoRA (`vae_skip` adapter). `set_train()` only unlocks `lora` params plus the four `skip_conv_*` layers. `save_ckpt` / `save_model` persist only the LoRA + skip weights (plus optimizer), keeping checkpoints small. `load_ckpt_from_state_dict` overlays them onto fresh SD-turbo state.
- **Multi-view input convention.** Inputs are `(B, V, C, H, W)`; `forward` rearranges to `(B*V) C H W` for the VAE/UNet and rearranges back. With a reference image `V=2`, otherwise `V=1`. `mv_unet=True` swaps `diffusers.UNet2DConditionModel` for the local `src/mv_unet.py` (a multi-view-aware UNet used by the `difix_ref` model).
- **Sampling.** `Difix.sample()` resizes input to multiples of 8, runs forward, then resizes back to the original resolution.

`src/train_difix.py` is the Accelerate-based training loop. Total loss = L2 + LPIPS + Gram (style); the Gram term ramps in over `--gram_loss_warmup_steps`. `src/loss.py` implements that Gram/style loss over VGG16 layers `relu1_2..relu5_3` with fixed weights. `src/dataset.py` (`PairedDataset`) reads a JSON manifest with `train`/`test` splits where each entry is `{image, target_image, ref_image?, prompt}`.

`src/pipeline_difix.py` is a self-contained `diffusers`-style `DifixPipeline` published to HF as `nvidia/difix` and `nvidia/difix_ref`. It re-implements the same VAE-with-skips + 1-step UNet flow so users can `from_pretrained(..., trust_remote_code=True)` without the rest of the repo — keep it in sync if you change the inference path in `src/model.py`.

### 2. Difix3D — progressive 3D refinement (`examples/`)

The 3D side is a thin layer that calls Difix on rendered novel views during training so artifact-corrected pseudo-targets feed back into the 3D model.

- **nerfstudio plugin** (`examples/nerfstudio/`): a normal nerfstudio method package whose `pyproject.toml` registers `difix3d` via `[project.entry-points.'nerfstudio.method_configs']`. Components live under `examples/nerfstudio/difix3d/` (config, datamanager, field, pipeline, trainer, model). After `pip install -e .`, `ns-train difix3d` is available.
- **gsplat trainer** (`examples/gsplat/simple_trainer_difix3d.py`): a fork of the upstream gsplat `simple_trainer` that calls Difix on rendered views during training.

Both expect data laid out as `DATA_DIR/{SCENE_ID}/{colmap, images, images_2, images_4, images_8}` and a pretrained NeRF / 3DGS checkpoint passed via `--load-checkpoint` / `--ckpt`.

## Things to know when editing

- `src/model.py` adds `src/` to `sys.path` so intra-`src` imports use bare names (`from model import ...`, `from mv_unet import UNet2DConditionModel`). Keep that pattern when adding modules under `src/`.
- The two UNet sources (`diffusers` vs local `src/mv_unet.py`) are selected by the `mv_unet` flag on `Difix` and chosen automatically in `inference_difix.py` based on whether `--ref_image` is given. If you change the multi-view layout (the `V` axis), update both `forward` in `src/model.py` and the corresponding paths in `src/pipeline_difix.py`.
- `src/dataset.py` currently has a bug: lines 37-49 reference `img_t` / `output_t` / `ref_t` before assignment (the first call on each should be on `input_img` / `output_img` / `ref_img`). Fix when touching dataset code.
- Checkpoints saved by `Difix.save_model` contain only LoRA + skip params; loading requires the same SD-turbo base, so don't change which base model is referenced without bumping the checkpoint format.
