"""DA3 inference on a COLMAP-posed scene → 3DGS PLY for splatfacto initialization.

Reads COLMAP poses from data/<scene>/sparse/0, picks an evenly-spaced subset of
views, runs DA3-GIANT-1.1 with the GS head, and writes a single 3DGS PLY at
<out>/gs_ply/0000.ply suitable for `splatfacto-da3 --pipeline.model.init-ply-path`.
"""
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image as PILImage

from depth_anything_3.api import DepthAnything3
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)


def load_colmap_poses(colmap_dir: Path):
    cams = read_cameras_binary(colmap_dir / "cameras.bin")
    imgs = read_images_binary(colmap_dir / "images.bin")
    # Sort by image name for deterministic ordering
    img_list = sorted(imgs.values(), key=lambda im: im.name)
    extrinsics, intrinsics, names = [], [], []
    for im in img_list:
        cam = cams[im.camera_id]
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params
            fx = fy = f
        elif cam.model == "SIMPLE_RADIAL":
            f, cx, cy, _k = cam.params
            fx = fy = f
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        R = qvec2rotmat(im.qvec)
        t = np.asarray(im.tvec, dtype=np.float64)
        E = np.eye(4, dtype=np.float64)
        E[:3, :3] = R
        E[:3, 3] = t  # world->cam
        extrinsics.append(E)
        intrinsics.append(K)
        names.append(im.name)
    return np.stack(extrinsics), np.stack(intrinsics), names, (cam.width, cam.height)


@hydra.main(config_path="../conf", config_name="run_da3_init", version_base=None)
def main(cfg: DictConfig):
    data_dir = Path(cfg.data)
    extrinsics, intrinsics, names, (full_w, full_h) = load_colmap_poses(data_dir / cfg.colmap_subdir)

    N = len(names)
    idx = np.linspace(0, N - 1, num=min(cfg.n_views, N)).round().astype(int)
    idx = np.unique(idx)
    extr_sub = extrinsics[idx]
    intr_sub = intrinsics[idx].copy()
    names_sub = [names[i] for i in idx]

    img_dir = data_dir / cfg.images_subdir
    paths = [str(img_dir / n) for n in names_sub]
    sample = PILImage.open(paths[0])
    in_w, in_h = sample.size
    sx = in_w / full_w
    sy = in_h / full_h
    intr_sub[:, 0, 0] *= sx
    intr_sub[:, 0, 2] *= sx
    intr_sub[:, 1, 1] *= sy
    intr_sub[:, 1, 2] *= sy

    print(f"Selected {len(paths)} / {N} views from {img_dir} ({in_w}x{in_h}); full COLMAP res {full_w}x{full_h}")

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    print(f"Loading {cfg.model_id} ...")
    model = DepthAnything3.from_pretrained(cfg.model_id).to(device).eval()

    print("Running DA3 inference (infer_gs=True) ...")
    pred = model.inference(
        image=paths,
        extrinsics=extr_sub.astype(np.float32),
        intrinsics=intr_sub.astype(np.float32),
        infer_gs=True,
        align_to_input_ext_scale=True,
        process_res=cfg.process_res,
        export_dir=str(out_dir),
        export_format="gs_ply",
    )

    n_g = pred.gaussians.means.shape[1]
    print(f"DA3 produced {n_g} gaussians; PLY written to {out_dir}/gs_ply/0000.ply")


if __name__ == "__main__":
    main()
