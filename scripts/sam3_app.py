"""Tiny Flask webapp: click on an image to segment it with SAM3.

Workflow:
    - Pick an image from `data/` (or --image).
    - Left-click on objects → box exemplars handed to SAM3.
    - Optionally also enter a text prompt.
    - Live overlay updates after each click.
    - When satisfied, hit "Save mask" → writes mask + overlay to
      `output/sam3_clicks/<image_stem>__<timestamp>/`.

Usage:
    /rhome/rborth/miniconda3/envs/flux2-depth/bin/python scripts/sam3_app.py \
        --port 7860

Then open http://<host>:7860 .
"""

from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

from sam3 import Sam3Segmenter, overlay_mask as overlay

DATA_DIR = Path("data").resolve()
OUT_DIR_DEFAULT = Path("data").resolve()  # outputs go under data/sam3_<ts>/, mirroring scripts/bbox_annotator.py

app = Flask(__name__)
state: dict = {}  # filled in main()


def pil_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 88) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def list_images() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        p.name
        for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def load_image(name: str) -> Image.Image:
    path = DATA_DIR / name
    return Image.open(path).convert("RGB")


def run_sam3(
    image: Image.Image,
    clicks: list[tuple[float, float]],
    text: str,
    score_threshold: float,
    mask_threshold: float,
) -> tuple[np.ndarray, list[dict]]:
    """Returns (binary union mask HxW, per-instance info)."""
    seg: Sam3Segmenter = state["segmenter"]
    union, _masks, boxes, scores = seg.segment(
        image,
        text=text,
        clicks=clicks if clicks else None,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
    )
    info = [
        {"score": float(s), "bbox": [float(v) for v in b]}
        for s, b in zip(scores, boxes)
    ]
    return union, info


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>SAM3 click-to-segment</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; background:#111; color:#eee; }
  .row { display:flex; gap:16px; align-items:flex-start; }
  .panel { background:#1c1c1c; padding:12px; border-radius:8px; }
  #stage { position:relative; display:inline-block; cursor:crosshair; }
  #stage img { display:block; max-width: 80vw; max-height: 80vh; }
  #overlay { position:absolute; left:0; top:0; pointer-events:none; }
  .dot { position:absolute; width:12px; height:12px; border-radius:50%;
         background:#ff4040; border:2px solid white; transform:translate(-50%,-50%);
         pointer-events:none; box-shadow:0 0 4px black; }
  button, select, input[type=text], input[type=number] {
    background:#2a2a2a; color:#eee; border:1px solid #444; padding:6px 10px; border-radius:4px;
  }
  button:hover { background:#3a3a3a; cursor:pointer; }
  label { display:block; margin: 6px 0 2px; font-size: 12px; color:#aaa; }
  #info { font-size: 12px; color:#aaa; max-width: 320px; }
  .status { font-size: 12px; color:#8c8; margin-top:6px; min-height:16px; }
</style></head>
<body>
<h2>SAM3 click-to-segment</h2>

<div class="row">
  <div class="panel">
    <div id="stage">
      <img id="img" src="" />
      <img id="overlay" src="" />
      <div id="dots"></div>
    </div>
    <div class="status" id="status">Pick an image →</div>
  </div>

  <div class="panel" id="info">
    <label>Image</label>
    <select id="imgSelect"></select>

    <label>Text prompt (optional)</label>
    <input type="text" id="text" placeholder="e.g. vase" style="width:100%;" />

    <label>Score threshold</label>
    <input type="number" id="scoreThr" value="0.3" min="0" max="1" step="0.05" />

    <label>Mask threshold</label>
    <input type="number" id="maskThr" value="0.5" min="0" max="1" step="0.05" />

    <div style="margin-top:10px;">
      <button onclick="runSegment()">Segment</button>
      <button onclick="clearClicks()">Clear clicks</button>
      <button onclick="saveMask()">Save mask</button>
    </div>

    <div style="margin-top:12px; font-size:12px; color:#888;">
      Left-click adds a positive exemplar (small box around click). Hit Segment after
      a click, or any click also auto-runs Segment.
    </div>

    <pre id="results" style="font-size:11px; color:#ccc; white-space:pre-wrap;"></pre>
  </div>
</div>

<script>
const stage = document.getElementById('stage');
const imgEl = document.getElementById('img');
const overlayEl = document.getElementById('overlay');
const dotsEl = document.getElementById('dots');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const imgSelect = document.getElementById('imgSelect');
const textEl = document.getElementById('text');

let clicks = [];        // list of {x,y} in *image* (natural) pixels
let nat = {w:0, h:0};   // natural image size
let busy = false;

async function loadImageList() {
  const r = await fetch('/images');
  const items = await r.json();
  imgSelect.innerHTML = '';
  for (const name of items) {
    const o = document.createElement('option');
    o.value = name; o.textContent = name;
    imgSelect.appendChild(o);
  }
  if (items.length) selectImage(items[0]);
}

function selectImage(name) {
  clicks = [];
  overlayEl.src = '';
  resultsEl.textContent = '';
  imgEl.onload = () => {
    nat.w = imgEl.naturalWidth; nat.h = imgEl.naturalHeight;
    redrawDots();
    statusEl.textContent = `Loaded ${name} (${nat.w}×${nat.h})`;
  };
  imgEl.src = `/image/${encodeURIComponent(name)}?t=${Date.now()}`;
}

imgSelect.addEventListener('change', e => selectImage(e.target.value));

stage.addEventListener('click', e => {
  if (busy) return;
  const rect = imgEl.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (nat.w / rect.width);
  const py = (e.clientY - rect.top)  * (nat.h / rect.height);
  clicks.push({x: px, y: py});
  redrawDots();
  runSegment();
});

function redrawDots() {
  dotsEl.innerHTML = '';
  const rect = imgEl.getBoundingClientRect();
  const stageRect = stage.getBoundingClientRect();
  const sx = rect.width / nat.w, sy = rect.height / nat.h;
  const offX = rect.left - stageRect.left, offY = rect.top - stageRect.top;
  for (const c of clicks) {
    const d = document.createElement('div');
    d.className = 'dot';
    d.style.left = (offX + c.x * sx) + 'px';
    d.style.top  = (offY + c.y * sy) + 'px';
    dotsEl.appendChild(d);
  }
}
window.addEventListener('resize', redrawDots);

function clearClicks() {
  clicks = [];
  overlayEl.src = '';
  resultsEl.textContent = '';
  redrawDots();
  statusEl.textContent = 'Clicks cleared.';
}

async function runSegment() {
  if (busy) return;
  busy = true;
  statusEl.textContent = 'Segmenting…';
  const t0 = performance.now();
  try {
    const r = await fetch('/segment', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        image: imgSelect.value,
        clicks: clicks,
        text: textEl.value,
        score_threshold: parseFloat(document.getElementById('scoreThr').value),
        mask_threshold: parseFloat(document.getElementById('maskThr').value),
      })
    });
    const j = await r.json();
    if (j.error) { statusEl.textContent = 'Error: ' + j.error; return; }
    overlayEl.src = 'data:image/jpeg;base64,' + j.overlay_b64;
    overlayEl.style.width = imgEl.clientWidth + 'px';
    overlayEl.style.height = imgEl.clientHeight + 'px';
    const ms = (performance.now() - t0).toFixed(0);
    statusEl.textContent = `${j.instances.length} instance(s) in ${ms} ms`;
    resultsEl.textContent = j.instances.map((it,i) =>
      `[${i}] score=${it.score.toFixed(3)} bbox=${it.bbox.map(v=>v.toFixed(0)).join(',')}`
    ).join('\\n');
  } finally {
    busy = false;
  }
}

async function saveMask() {
  const r = await fetch('/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      image: imgSelect.value,
      clicks: clicks,
      text: textEl.value,
      score_threshold: parseFloat(document.getElementById('scoreThr').value),
      mask_threshold: parseFloat(document.getElementById('maskThr').value),
    })
  });
  const j = await r.json();
  if (j.error) { statusEl.textContent = 'Save error: ' + j.error; return; }
  statusEl.textContent = 'Saved → ' + j.dir;
}

loadImageList();
</script>
</body></html>
"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/images")
def images():
    return jsonify(list_images())


@app.route("/image/<path:name>")
def image_file(name: str):
    p = DATA_DIR / name
    if not p.exists():
        return ("not found", 404)
    return send_file(p)


@app.route("/segment", methods=["POST"])
def segment():
    payload = request.get_json(force=True)
    name = payload["image"]
    clicks = [(float(c["x"]), float(c["y"])) for c in payload.get("clicks", [])]
    text = payload.get("text", "") or ""
    s_thr = float(payload.get("score_threshold", 0.3))
    m_thr = float(payload.get("mask_threshold", 0.5))
    if not clicks and not text.strip():
        return jsonify({"error": "Need at least one click or a text prompt."})
    try:
        image = load_image(name)
        t0 = time.time()
        union, info = run_sam3(image, clicks, text, s_thr, m_thr)
        dt = time.time() - t0
        ov = overlay(image, union)
        return jsonify(
            {"overlay_b64": pil_to_b64(ov), "instances": info, "elapsed_s": dt}
        )
    except Exception as e:
        return jsonify({"error": repr(e)})


@app.route("/save", methods=["POST"])
def save():
    payload = request.get_json(force=True)
    name = payload["image"]
    clicks = [(float(c["x"]), float(c["y"])) for c in payload.get("clicks", [])]
    text = payload.get("text", "") or ""
    s_thr = float(payload.get("score_threshold", 0.3))
    m_thr = float(payload.get("mask_threshold", 0.5))
    if not clicks and not text.strip():
        return jsonify({"error": "Need at least one click or a text prompt."})
    try:
        image = load_image(name)
        union, info = run_sam3(image, clicks, text, s_thr, m_thr)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(name).stem
        d = state["out_dir"] / f"sam3_{stem}_{ts}"
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray((union * 255).astype(np.uint8)).save(d / "mask.png")
        overlay(image, union).save(d / "overlay.jpg", quality=92)
        image.save(d / f"image{Path(name).suffix.lower()}")
        meta = {
            "image": name,
            "clicks": [{"x": x, "y": y} for x, y in clicks],
            "text": text,
            "score_threshold": s_thr,
            "mask_threshold": m_thr,
            "instances": info,
        }
        (d / "meta.json").write_text(__import__("json").dumps(meta, indent=2))
        return jsonify({"dir": str(d)})
    except Exception as e:
        return jsonify({"error": repr(e)})


import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402


@hydra.main(version_base=None, config_path="../conf", config_name="sam3_app")
def main(cfg: DictConfig) -> None:
    print(f"Loading {cfg.model_id}", flush=True)
    seg = Sam3Segmenter.from_pretrained(cfg.model_id)
    print(f"  -> running on {seg.device}", flush=True)
    state["segmenter"] = seg
    out_dir = Path(cfg.out_dir).resolve() if cfg.out_dir else OUT_DIR_DEFAULT
    state["out_dir"] = out_dir
    state["out_dir"].mkdir(parents=True, exist_ok=True)

    print(f"Serving at http://{cfg.host}:{cfg.port}", flush=True)
    print(f"Saving annotations under {state['out_dir']}/sam3_<stem>_<ts>/", flush=True)
    app.run(host=cfg.host, port=cfg.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
