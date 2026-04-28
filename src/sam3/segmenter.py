"""SAM3 text- and box-prompted segmentation.

Shared by:
  - scripts/sam3_segment.py (one-shot CLI)
  - scripts/sam3_app.py     (Flask click-to-segment webapp)
  - scripts/inpaint_sam3.py (inpainting conditioned on SAM3 masks)

This module owns:
  - model + processor loading (with bf16 + device handling)
  - the click-exemplar → bbox conversion used by the webapp
  - a single forward pass returning per-instance masks/boxes/scores plus
    the union mask
  - a small overlay helper for visualisation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# Half-side of the exemplar bbox built around each click (in pixels).
EXEMPLAR_HALF = 24


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.55,
) -> Image.Image:
    """Tint `mask` regions of `image` with `color` at the given alpha."""
    img = np.asarray(image.convert("RGB")).astype(np.float32)
    m = mask.astype(bool)
    tint = np.array(color, dtype=np.float32)
    img[m] = (1 - alpha) * img[m] + alpha * tint
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def clicks_to_boxes(
    clicks: Iterable[tuple[float, float]],
    image_size: tuple[int, int],
    half: int = EXEMPLAR_HALF,
) -> list[list[float]]:
    """Convert (x, y) clicks into [x0, y0, x1, y1] exemplar boxes clipped to the image."""
    w, h = image_size
    boxes: list[list[float]] = []
    for x, y in clicks:
        boxes.append([
            max(0, x - half),
            max(0, y - half),
            min(w, x + half),
            min(h, y + half),
        ])
    return boxes


def load_sam3(
    model_id: str = "facebook/sam3",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Sam3Processor, Sam3Model, str]:
    """Load processor + model. Returns (processor, model, resolved_device)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return processor, model, device


@dataclass
class Sam3Segmenter:
    """Stateful wrapper that keeps a SAM3 model loaded for repeated calls."""

    processor: Sam3Processor
    model: Sam3Model
    device: str

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Sam3Segmenter":
        processor, model, dev = load_sam3(model_id, device=device, dtype=dtype)
        return cls(processor=processor, model=model, device=dev)

    def segment(
        self,
        image: Image.Image,
        text: str = "",
        clicks: Optional[Iterable[tuple[float, float]]] = None,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run SAM3. Returns (union_mask HxW bool, per_instance_masks NxHxW bool,
        boxes Nx4 float, scores N float)."""
        return run_sam3(
            self.processor,
            self.model,
            self.device,
            image,
            text=text,
            clicks=clicks,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )


def run_sam3(
    processor: Sam3Processor,
    model: Sam3Model,
    device: str,
    image: Image.Image,
    text: str = "",
    clicks: Optional[Iterable[tuple[float, float]]] = None,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a single SAM3 forward pass with text and/or click-exemplar prompts.

    Returns:
        union: HxW bool array — OR of all kept instance masks.
        masks: NxHxW bool array of per-instance masks.
        boxes: Nx4 float array of per-instance bboxes (xyxy).
        scores: N float array of per-instance scores.
    """
    proc_kwargs: dict = {"images": image, "return_tensors": "pt"}
    if text and text.strip():
        proc_kwargs["text"] = text.strip()
    clicks_list = list(clicks) if clicks is not None else []
    if clicks_list:
        boxes = clicks_to_boxes(clicks_list, (image.width, image.height))
        proc_kwargs["input_boxes"] = [boxes]
        proc_kwargs["input_boxes_labels"] = [[1] * len(boxes)]

    inputs = processor(**proc_kwargs).to(device)
    # Match model dtype for float-valued tensors (pixel_values, input_boxes).
    for k, v in list(inputs.items()):
        if torch.is_tensor(v) and v.is_floating_point():
            inputs[k] = v.to(torch.bfloat16)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=mask_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    scores = results["scores"].float().cpu().numpy()
    boxes_arr = results["boxes"].float().cpu().numpy()
    masks = results["masks"].cpu().numpy().astype(bool)

    if len(masks) == 0:
        empty = np.zeros((image.height, image.width), dtype=bool)
        return empty, masks, boxes_arr, scores

    union = np.any(masks, axis=0)
    return union, masks, boxes_arr, scores
