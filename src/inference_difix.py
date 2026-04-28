import os
import imageio
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image
from glob import glob
from tqdm import tqdm
from model import Difix


@hydra.main(config_path="../conf", config_name="inference_difix", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = Difix(
        pretrained_name=cfg.model_name,
        pretrained_path=cfg.model_path,
        timestep=cfg.timestep,
        mv_unet=cfg.ref_image is not None,
    )
    model.set_eval()

    if os.path.isdir(cfg.input_image):
        input_images = sorted(glob(os.path.join(cfg.input_image, "*.png")))
    else:
        input_images = [cfg.input_image]

    ref_images = None
    if cfg.ref_image is not None:
        if os.path.isdir(cfg.ref_image):
            ref_images = sorted(glob(os.path.join(cfg.ref_image, "*")))
        else:
            ref_images = [cfg.ref_image]
        assert len(input_images) == len(ref_images), "Number of input images and reference images should be the same"

    output_images = []
    for i, input_image in enumerate(tqdm(input_images, desc="Processing images")):
        image = Image.open(input_image).convert('RGB')
        ref_image = Image.open(ref_images[i]).convert('RGB') if ref_images is not None else None
        output_image = model.sample(
            image,
            height=cfg.height,
            width=cfg.width,
            ref_image=ref_image,
            prompt=cfg.prompt,
        )
        output_images.append(output_image)

    if cfg.video:
        video_path = os.path.join(cfg.output_dir, "output.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        for output_image in tqdm(output_images, desc="Saving video"):
            writer.append_data(np.array(output_image))
        writer.close()
    else:
        for i, output_image in enumerate(tqdm(output_images, desc="Saving images")):
            output_image.save(os.path.join(cfg.output_dir, os.path.basename(input_images[i])))


if __name__ == "__main__":
    main()