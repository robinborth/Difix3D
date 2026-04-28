"""
Tiny web UI to annotate a bounding box (and prompt) on an image.

Usage:
    python scripts/bbox_annotator.py --image path/to/img.jpg
    # then open http://localhost:8765 in a browser

On Save, writes to --out_dir (default data/bbox_<ts>/):
  - image.png       copy of the input (original resolution)
  - mask.png        L-mode, 255 inside the bbox, 0 outside
  - metadata.json   {bbox: [x1,y1,x2,y2], prompt, source_image, width, height}

Stdlib http.server only — no Flask/Gradio dependency.
"""

from __future__ import annotations

import io
import json
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image, ImageDraw

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>bbox annotator</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }
  #wrap { position: relative; display: inline-block; user-select: none; }
  #img { display: block; max-width: 95vw; max-height: 80vh; }
  #ov { position: absolute; left: 0; top: 0; cursor: crosshair; }
  .row { margin: 8px 0; }
  input[type=text] { width: 60ch; padding: 6px; background: #222; color: #eee; border: 1px solid #444; }
  button { padding: 8px 14px; background: #2a6; color: white; border: 0; cursor: pointer; }
  button:disabled { background: #555; cursor: not-allowed; }
  #status { margin-left: 12px; color: #aaa; }
  code { background: #222; padding: 2px 6px; }
</style></head><body>
<h2>bbox annotator</h2>
<div class="row">drag on the image to draw a bounding box. click again to redraw.</div>
<div id="wrap">
  <img id="img" src="/image" />
  <canvas id="ov"></canvas>
</div>
<div class="row">
  bbox: <code id="bbox">none</code>
</div>
<div class="row">
  prompt: <input id="prompt" type="text" placeholder="what to inpaint inside the bbox" />
</div>
<div class="row">
  <button id="save" disabled>save</button>
  <span id="status"></span>
</div>
<script>
const img = document.getElementById('img');
const ov = document.getElementById('ov');
const ctx = ov.getContext('2d');
const bboxEl = document.getElementById('bbox');
const saveBtn = document.getElementById('save');
const statusEl = document.getElementById('status');
const promptEl = document.getElementById('prompt');

let natW = 0, natH = 0;       // intrinsic image size
let scale = 1;                 // displayed / natural
let drag = null;               // {x0,y0,x1,y1} in displayed px
let bbox = null;               // [x1,y1,x2,y2] in NATURAL px

function fitOverlay() {
  ov.width = img.clientWidth;
  ov.height = img.clientHeight;
  ov.style.width = img.clientWidth + 'px';
  ov.style.height = img.clientHeight + 'px';
  scale = img.clientWidth / natW;
  redraw();
}
function redraw() {
  ctx.clearRect(0, 0, ov.width, ov.height);
  const r = drag || (bbox ? {x0: bbox[0]*scale, y0: bbox[1]*scale,
                             x1: bbox[2]*scale, y1: bbox[3]*scale} : null);
  if (!r) return;
  const x = Math.min(r.x0,r.x1), y = Math.min(r.y0,r.y1);
  const w = Math.abs(r.x1-r.x0), h = Math.abs(r.y1-r.y0);
  ctx.fillStyle = 'rgba(255,80,80,0.25)';
  ctx.fillRect(x,y,w,h);
  ctx.strokeStyle = '#ff5050'; ctx.lineWidth = 2;
  ctx.strokeRect(x,y,w,h);
}
img.addEventListener('load', () => {
  natW = img.naturalWidth; natH = img.naturalHeight;
  fitOverlay();
});
window.addEventListener('resize', fitOverlay);

function pos(e) {
  const r = ov.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}
ov.addEventListener('mousedown', e => {
  const [x,y] = pos(e); drag = {x0:x,y0:y,x1:x,y1:y}; redraw();
});
ov.addEventListener('mousemove', e => {
  if (!drag) return;
  const [x,y] = pos(e); drag.x1 = x; drag.y1 = y; redraw();
});
function endDrag() {
  if (!drag) return;
  const x1 = Math.round(Math.min(drag.x0,drag.x1)/scale);
  const y1 = Math.round(Math.min(drag.y0,drag.y1)/scale);
  const x2 = Math.round(Math.max(drag.x0,drag.x1)/scale);
  const y2 = Math.round(Math.max(drag.y0,drag.y1)/scale);
  drag = null;
  if (x2-x1 >= 2 && y2-y1 >= 2) {
    bbox = [Math.max(0,x1), Math.max(0,y1),
            Math.min(natW,x2), Math.min(natH,y2)];
    bboxEl.textContent = bbox.join(', ') + `  (${bbox[2]-bbox[0]} x ${bbox[3]-bbox[1]})`;
    saveBtn.disabled = false;
  }
  redraw();
}
ov.addEventListener('mouseup', endDrag);
ov.addEventListener('mouseleave', endDrag);

saveBtn.addEventListener('click', async () => {
  saveBtn.disabled = true; statusEl.textContent = 'saving...';
  try {
    const r = await fetch('/save', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({bbox, prompt: promptEl.value}),
    });
    const j = await r.json();
    if (j.ok) statusEl.textContent = 'saved -> ' + j.out_dir;
    else statusEl.textContent = 'error: ' + j.error;
  } catch (e) { statusEl.textContent = 'error: ' + e; }
  saveBtn.disabled = false;
});
</script></body></html>
"""


def make_handler(image_bytes: bytes, image_mime: str, src_path: Path,
                 width: int, height: int, out_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, PAGE)
            elif self.path == "/image":
                self._send(200, image_bytes, ctype=image_mime)
            else:
                self._send(404, "not found", ctype="text/plain")

        def do_POST(self):
            if self.path != "/save":
                self._send(404, "not found", ctype="text/plain"); return
            n = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(n))
                bbox = [int(v) for v in payload["bbox"]]
                prompt = str(payload.get("prompt", ""))
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(width, x1)); x2 = max(0, min(width, x2))
                y1 = max(0, min(height, y1)); y2 = max(0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"empty bbox {bbox}")

                out_dir.mkdir(parents=True, exist_ok=True)
                Image.open(src_path).convert("RGB").save(out_dir / "image.png")
                mask = Image.new("L", (width, height), 0)
                ImageDraw.Draw(mask).rectangle((x1, y1, x2, y2), fill=255)
                mask.save(out_dir / "mask.png")
                meta = {
                    "source_image": str(src_path.resolve()),
                    "width": width, "height": height,
                    "bbox": [x1, y1, x2, y2],
                    "prompt": prompt,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                }
                (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
                self._send(200, json.dumps({"ok": True, "out_dir": str(out_dir)}),
                           ctype="application/json")
                print(f"saved -> {out_dir}", flush=True)
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           ctype="application/json")

    return Handler


import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="bbox_annotator")
def main(cfg: DictConfig) -> None:
    src = Path(cfg.image).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"image not found: {src}")

    img = Image.open(src).convert("RGB")
    w, h = img.size
    buf = io.BytesIO(); img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) if cfg.out_dir else Path("data") / f"bbox_{ts}"

    handler = make_handler(image_bytes, "image/png", src, w, h, out_dir)
    httpd = HTTPServer((cfg.host, cfg.port), handler)
    url = f"http://{cfg.host}:{cfg.port}"
    print(f"serving {src.name} ({w}x{h}) at {url}")
    print(f"output dir: {out_dir}")
    print("Ctrl-C to stop")

    if not cfg.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
