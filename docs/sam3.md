# `src/sam3` — shared SAM3 segmenter

A small wrapper around `transformers.Sam3Model` / `Sam3Processor` that the
SAM3-flavoured scripts in this repo all reuse. Originally inlined three
times (segment CLI, click-app, sam3-conditioned inpaint); pulled out so
there's one place that owns model loading, click-to-bbox conversion, and
post-processing.

## API

```python
from sam3 import Sam3Segmenter, run_sam3, overlay_mask, clicks_to_boxes

seg = Sam3Segmenter.from_pretrained("facebook/sam3")  # bf16, cuda if available
union, masks, boxes, scores = seg.segment(
    image,
    text="vase",                 # optional
    clicks=[(x, y), ...],        # optional; each becomes a 48x48 exemplar bbox
    score_threshold=0.3,
    mask_threshold=0.5,
)
```

`run_sam3(processor, model, device, image, …)` is the same forward pass as a
free function, for callers that want to manage model lifetime themselves.

`overlay_mask(image, union, color, alpha)` returns a tinted PIL preview.

`clicks_to_boxes(clicks, image_size, half=EXEMPLAR_HALF)` converts click
points into clipped exemplar bboxes — used by the webapp's click handler.

## Callers

- `scripts/sam3_segment.py` — one-shot text-prompted CLI.
- `scripts/sam3_app.py` — Flask click-to-segment UI; keeps the segmenter
  in a process-global dict and reuses it across requests.
- `scripts/inpaint_sam3.py` — does *not* construct a `Sam3Segmenter`
  itself; it only consumes a pre-computed mask file. The SAM3 dep enters
  the inpaint pipeline via the upstream sam3_app annotation folder.

## Dtype / device

The model is loaded in bf16. The wrapper casts every floating-point input
tensor to bf16 before the forward pass, otherwise `pixel_values` defaults
to fp32 and the model errors. Device defaults to `cuda` when available;
override with `Sam3Segmenter.from_pretrained(..., device="cpu")`.

## What's not in here

- No batching across multiple images — every script feeds one image at a
  time.
- No mask post-processing (dilation, feathering). That belongs to the
  caller; `inpaint_sam3.py` does it via `PIL.ImageFilter`.
- No mask serialization helpers — callers save `mask.png` directly.
