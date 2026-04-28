# scripts/ overview

One-line summary of every script + whether it's "official" (FLUX.2 inference path)
or experimental (added by you on top of the upstream repo). Mark `[keep]` /
`[delete]` next to each entry, then I'll clean up based on this file.

## Official (upstream / part of inference UX)

- [ ] **cli.py** — interactive REPL for FLUX.2 t2i + single/multi-ref editing.
      Has its own `Config` dataclass; not wired through hydra (REPL doesn't fit).
      Entrypoint: `PYTHONPATH=src python scripts/cli.py`.

## Experimental — FLUX.2 image / editing

- [ ] **ablation_t2i.py** — runs one fixed prompt+seed across every Klein variant,
      writes a PNG per model under `output/ablation_<ts>/`. Pure T2I.
      Hydra config: `conf/ablation_t2i.yaml`.
- [ ] **compose_ablation_figure.py** — composes the per-model PNGs from an
      `ablation_*` dir into a labeled single-row contact sheet. Pure pillow,
      no model load. Takes `<ablation_dir>` as positional arg (not hydra'd).

## Experimental — depth-conditioned FLUX.2

- [ ] **depth_inference.py** — input image → Depth-Anything-V2 depth map →
      FLUX.2 image-edit pathway uses depth as geometric reference. Inference
      only. Hydra config: `conf/depth_inference.yaml`.
- [ ] **depth_overfit.py** — overfits a *subset* of the FLUX.2 transformer
      (default: img_in + final_layer + double_blocks.0) on a single
      (image, depth) pair via rectified-flow loss. Periodic sampling.
      Hydra config: `conf/depth_overfit.yaml`. Imports helpers from
      `depth_inference.py`, so requires `PYTHONPATH=src:.`.

## Experimental — inpainting

- [ ] **bbox_annotator.py** — tiny browser tool: open an image, draw a bbox,
      type a prompt, hit save → writes `image.png` / `mask.png` / `metadata.json`
      to `data/bbox_<ts>/`. Hydra config: `conf/bbox_annotator.yaml`.
- [ ] **inpaint.py** — full inpainting demo. Generates a base image from
      `--base_prompt`, builds a bbox mask, then runs three methods
      (kontext / blended_latent / pixel_composite). Side-by-side comparison.
      Hydra config: `conf/inpaint.yaml`.
- [ ] **inpaint_from_bbox.py** — consumes a `data/bbox_<ts>/` folder produced by
      `bbox_annotator.py` and inpaints it (defaults to `kontext` method).
      Reuses helpers from `inpaint.py`. Hydra config: `conf/inpaint_from_bbox.yaml`.

## Experimental — SAM3 segmentation (no FLUX dep)

- [ ] **sam3_segment.py** — one-shot CLI: SAM3 text-prompted instance
      segmentation on a single image, writes mask + overlay.
      Hydra config: `conf/sam3_segment.yaml`.
- [ ] **sam3_app.py** — Flask webapp: pick an image from `data/`, click
      objects (each click → exemplar bbox), optionally add a text prompt,
      live overlay updates, save mask+overlay.
      Hydra config: `conf/sam3_app.yaml`.

## Misc

- [ ] **link_data.sh** — shell helper (not a python script).

---

After you mark items `[delete]`, I'll remove the script + its
`conf/<name>.yaml` and any imports it leaves behind.
