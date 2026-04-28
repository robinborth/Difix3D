"""Smoke test: Difix Quickstart inference on assets/example_input.png."""
import os
import shutil
from PIL import Image
from diffusers.utils import load_image
from pipeline_difix import DifixPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs", "quickstart")
os.makedirs(OUT, exist_ok=True)

input_path = os.path.join(ROOT, "assets", "example_input.png")
ref_path = os.path.join(ROOT, "assets", "example_ref.png")

# 1. Single-image (no reference) Difix
pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")
input_image = load_image(input_path)
out = pipe("remove degradation", image=input_image, num_inference_steps=1,
           timesteps=[199], guidance_scale=0.0).images[0]
out.save(os.path.join(OUT, "output_difix.png"))
shutil.copy(input_path, os.path.join(OUT, "input.png"))
print("Saved:", os.path.join(OUT, "output_difix.png"))

# 2. Reference-guided Difix
pipe_ref = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
pipe_ref.to("cuda")
ref_image = load_image(ref_path)
out_ref = pipe_ref("remove degradation", image=input_image, ref_image=ref_image,
                   num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
out_ref.save(os.path.join(OUT, "output_difix_ref.png"))
shutil.copy(ref_path, os.path.join(OUT, "ref.png"))
print("Saved:", os.path.join(OUT, "output_difix_ref.png"))
