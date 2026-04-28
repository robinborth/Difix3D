"""Run SAM3 text-prompted segmentation on a single image and save mask + overlay.

Usage:
    PYTHONPATH=src python scripts/sam3_segment.py \
        image=data/DSC07956.JPG \
        prompt=vase \
        out=output/sam3
"""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from sam3 import Sam3Segmenter, overlay_mask


@hydra.main(version_base=None, config_path="../conf", config_name="sam3_segment")
def main(cfg: DictConfig) -> None:
    image_path = Path(cfg.image)
    out_path = Path(cfg.out)
    out_path.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")

    print(f"Loading {cfg.model_id}")
    seg = Sam3Segmenter.from_pretrained(cfg.model_id, device=cfg.device)
    print(f"  -> running on {seg.device}")

    union, masks, boxes, scores = seg.segment(
        image,
        text=cfg.prompt,
        score_threshold=cfg.score_threshold,
        mask_threshold=cfg.mask_threshold,
    )

    print(f"Found {len(scores)} instances for prompt '{cfg.prompt}'")
    for i, s in enumerate(scores):
        print(f"  [{i}] score={float(s):.3f} bbox={boxes[i].round(1).tolist()}")

    if len(masks) == 0:
        print("No instances above threshold — nothing to save.")
        return

    stem = f"{image_path.stem}__{cfg.prompt.replace(' ', '_')}"

    Image.fromarray((union * 255).astype(np.uint8)).save(out_path / f"{stem}_mask.png")
    overlay_mask(image, union).save(out_path / f"{stem}_overlay.jpg", quality=92)

    for i, m in enumerate(masks):
        Image.fromarray((m * 255).astype(np.uint8)).save(
            out_path / f"{stem}_mask_{i:02d}_score{scores[i]:.2f}.png"
        )

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
