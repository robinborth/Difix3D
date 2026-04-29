"""4-panel comparison sheet for the 3D inpainting pipeline.

Re-implements the same `label()` / `grid()` helpers as `scripts/inpaint.py`
so this package has no upstream-script import. Plus a thin
`compose_four_panel(...)` convenience that arranges the four stages of
the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 22) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(cand, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def label(img: Image.Image, text: str, pad: int = 6) -> Image.Image:
    """Return a copy of `img` with a small text label at the top-left."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _font()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def grid(
    images: Sequence[Image.Image],
    cols: int,
    pad: int = 8,
    bg: tuple = (20, 20, 20),
) -> Image.Image:
    """Arrange `images` in a `cols`-wide grid. Cells are sized to the max
    width/height across the inputs and padded with `bg`."""
    if not images:
        raise ValueError("grid() needs at least one image")
    cw = max(im.width for im in images)
    ch = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    W = cols * cw + (cols + 1) * pad
    H = rows * ch + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (cw + pad) + (cw - im.width) // 2
        y = pad + r * (ch + pad) + (ch - im.height) // 2
        sheet.paste(im, (x, y))
    return sheet


def compose_four_panel(
    original: Image.Image,
    after_removal: Image.Image,
    inpainted_gt: Image.Image,
    after_finetune: Image.Image,
    out_path: Path,
    cols: int = 2,
) -> Image.Image:
    """Build and save the 2x2 (or 1x4) sheet for one view."""
    panels = [
        label(original, "01 original render"),
        label(after_removal, "02 after gaussian removal"),
        label(inpainted_gt, "03 inpainted GT"),
        label(after_finetune, "04 after finetune"),
    ]
    sheet = grid(panels, cols=cols)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return sheet
