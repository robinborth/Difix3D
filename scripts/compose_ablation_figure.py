"""
Compose the 5 ablation outputs into a single-row figure with labels below.

Usage:
    PYTHONPATH=src python scripts/compose_ablation_figure.py <ablation_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(ablation_dir: Path) -> Path:
    results = json.loads((ablation_dir / "results.json").read_text())
    runs = [r for r in results["runs"] if r["status"] == "ok"]
    if not runs:
        raise SystemExit(f"no ok runs in {ablation_dir}/results.json")

    images = [Image.open(ablation_dir / Path(r["path"]).name).convert("RGB") for r in runs]

    pad = 32
    label_h = 220
    w, h = images[0].size
    images = [img.resize((w, h), Image.LANCZOS) if img.size != (w, h) else img for img in images]

    n = len(images)
    canvas_w = pad + n * (w + pad)
    canvas_h = pad + h + label_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(64)
    sub_font = _load_font(48)

    for i, (img, r) in enumerate(zip(images, runs)):
        x = pad + i * (w + pad)
        y = pad
        canvas.paste(img, (x, y))
        name = r["model"]
        sub = f"{r['steps']} steps  g={r['guidance']}  {r['seconds']:.1f}s"
        tw = draw.textlength(name, font=title_font)
        sw = draw.textlength(sub, font=sub_font)
        draw.text((x + (w - tw) // 2, y + h + 30), name, fill="black", font=title_font)
        draw.text((x + (w - sw) // 2, y + h + 30 + 90), sub, fill="#555555", font=sub_font)

    out_path = ablation_dir / "grid.png"
    canvas.save(out_path, quality=95, subsampling=0)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <ablation_dir>")
    print(compose(Path(sys.argv[1])))
