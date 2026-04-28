"""Run Depth Anything 3 on garden images: depth viz + GS-head renders of input views."""
import os, glob
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode


@hydra.main(config_path="../conf", config_name="run_da3_garden", version_base=None)
def main(cfg: DictConfig):
    data_dir = cfg.data_dir
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(data_dir, "*.JPG")) + glob.glob(os.path.join(data_dir, "*.jpg")))
    stride = max(len(paths) // cfg.n_views, 1)
    paths = paths[::stride][:cfg.n_views]
    print(f"Selected {len(paths)} images (stride={stride}) from {data_dir}")

    device = torch.device("cuda")
    print(f"Loading {cfg.model_id} ...")
    model = DepthAnything3.from_pretrained(cfg.model_id).to(device).eval()

    # 1) Inference with GS branch on (also returns depth + poses)
    print("Running inference with GS head ...")
    prediction = model.inference(
        paths,
        infer_gs=True,
        process_res=cfg.process_res,
        process_res_method="upper_bound_resize",
    )

    # Save processed source RGBs
    proc = prediction.processed_images  # (N, H, W, 3), float in [0,1] or uint8
    proc_arr = np.asarray(proc)
    if proc_arr.dtype != np.uint8:
        proc_np = (np.clip(proc_arr, 0, 1) * 255).astype(np.uint8)
    else:
        proc_np = proc_arr

    src_dir = os.path.join(out_dir, "source")
    depth_dir = os.path.join(out_dir, "depth")
    render_dir = os.path.join(out_dir, "renders_input_views")
    render_depth_dir = os.path.join(out_dir, "renders_input_views_depth")
    for d in [src_dir, depth_dir, render_dir, render_depth_dir]:
        os.makedirs(d, exist_ok=True)

    # 2) Save depth visualizations
    depth_np = prediction.depth  # (V, H, W) numpy
    print(f"depth shape: {depth_np.shape}, RGB shape: {proc_np.shape}")

    valid = depth_np > 0
    inv = np.where(valid, 1.0 / np.maximum(depth_np, 1e-6), 0.0)
    gmin = float(np.percentile(inv[valid], 2))
    gmax = float(np.percentile(inv[valid], 98))

    for i, p in enumerate(paths):
        name = Path(p).stem
        Image.fromarray(proc_np[i]).save(os.path.join(src_dir, f"{name}.png"))
        dvis = visualize_depth(depth_np[i], depth_min=gmin, depth_max=gmax)
        Image.fromarray(dvis).save(os.path.join(depth_dir, f"{name}.png"))

    print(f"Saved {len(paths)} source RGBs to {src_dir}")
    print(f"Saved {len(paths)} depth visualizations to {depth_dir}")

    # 3) Render input views from the predicted Gaussians ("original" trajectory)
    gs = prediction.gaussians
    H, W = depth_np.shape[-2:]
    extrinsics = torch.from_numpy(prediction.extrinsics).unsqueeze(0).to(gs.means)
    intrinsics = torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(gs.means)

    print("Rendering Gaussians at input views ...")
    color, depth_r = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_shape=(H, W),
        chunk_size=4,
        trj_mode="original",
        use_sh=True,
        color_mode="RGB+ED",
        enable_tqdm=True,
    )
    col_np = (color[0].clamp(0, 1).float().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    dep_r_np = depth_r[0].float().cpu().numpy()

    for i, p in enumerate(paths):
        name = Path(p).stem
        Image.fromarray(col_np[i]).save(os.path.join(render_dir, f"{name}.png"))
        dvis = visualize_depth(dep_r_np[i], depth_min=gmin, depth_max=gmax)
        Image.fromarray(dvis).save(os.path.join(render_depth_dir, f"{name}.png"))

    print(f"Saved renders to {render_dir}")
    print(f"Saved render-depths to {render_depth_dir}")

    # 4) Side-by-side composite for quick inspection
    comp_dir = os.path.join(out_dir, "compare")
    os.makedirs(comp_dir, exist_ok=True)
    for i, p in enumerate(paths):
        name = Path(p).stem
        src = proc_np[i]
        d = visualize_depth(depth_np[i], depth_min=gmin, depth_max=gmax)
        r = col_np[i]
        h, w = src.shape[:2]
        def to_size(a):
            return np.asarray(Image.fromarray(a).resize((w, h)))
        panel = np.concatenate([src, to_size(d), to_size(r)], axis=1)
        Image.fromarray(panel).save(os.path.join(comp_dir, f"{name}.png"))
    print(f"Saved side-by-side comparisons to {comp_dir}")

    print("DONE")


if __name__ == "__main__":
    main()
