"""Smoke test: Difix Quickstart inference on assets/example_input.png."""
import os
import shutil

import hydra
from diffusers.utils import load_image
from omegaconf import DictConfig

from pipeline_difix import DifixPipeline


@hydra.main(config_path="../conf", config_name="run_quickstart", version_base=None)
def main(cfg: DictConfig):
    out = cfg.output_dir
    os.makedirs(out, exist_ok=True)

    input_image = load_image(cfg.input_path)

    # 1. Single-image (no reference) Difix
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")
    out_img = pipe(cfg.prompt, image=input_image, num_inference_steps=1,
                   timesteps=[cfg.timestep], guidance_scale=0.0).images[0]
    out_img.save(os.path.join(out, "output_difix.png"))
    shutil.copy(cfg.input_path, os.path.join(out, "input.png"))
    print("Saved:", os.path.join(out, "output_difix.png"))

    # 2. Reference-guided Difix
    pipe_ref = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe_ref.to("cuda")
    ref_image = load_image(cfg.ref_path)
    out_ref = pipe_ref(cfg.prompt, image=input_image, ref_image=ref_image,
                       num_inference_steps=1, timesteps=[cfg.timestep], guidance_scale=0.0).images[0]
    out_ref.save(os.path.join(out, "output_difix_ref.png"))
    shutil.copy(cfg.ref_path, os.path.join(out, "ref.png"))
    print("Saved:", os.path.join(out, "output_difix_ref.png"))


if __name__ == "__main__":
    main()
