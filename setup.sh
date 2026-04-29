#!/usr/bin/env bash
# Bootstrap the unified `wm2sim` (Sim-Ready World Models) conda env.
# Combines flux2 + sam3 + gsplat into a single environment so the 3D
# inpainting pipeline can be a single Python process per stage.
#
# Run from anywhere:
#   bash setup.sh
#
# Prereqs (HPC cluster):
#   - conda
#   - environment-modules (`module load cuda/12.8` works)

set -euo pipefail

ENV=wm2sim

# 1) CUDA toolchain for any CUDA-extension builds (gsplat JIT, fused-ssim).
module load cuda/12.8

# 2) Fresh env on python 3.12 (matches flux2 pyproject upper bound).
conda create -y -n "$ENV" python=3.12 -c conda-forge
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# 3) PyTorch matching the flux2 stack (torch 2.8.0 + cu129).
pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
    --extra-index-url https://download.pytorch.org/whl/cu129

# 4) flux2 deps (mirrors the upstream flux2 pyproject).
pip install --no-cache-dir \
    "einops==0.8.1" \
    "transformers==5.6.2" \
    "safetensors==0.4.5" \
    "fire==0.7.1" \
    "openai==2.8.1" \
    "accelerate==1.12.0" \
    "hydra-core==1.3.2" \
    "hydra-submitit-launcher==1.2.0"

# 5) Difix-side deps (the Difix3D base requirements minus things bundled above).
pip install --no-cache-dir \
    diffusers \
    peft \
    lpips \
    pillow \
    "numpy<2.0.0" \
    flask

# 6) gsplat + its training-recipe deps (matches examples/gsplat/requirements.txt).
pip install --no-cache-dir gsplat==1.5.3
pip install --no-cache-dir \
    viser \
    "nerfview==0.0.2" \
    "imageio[ffmpeg]" \
    scikit-learn \
    tqdm \
    "torchmetrics[image]" \
    opencv-python \
    "tyro>=0.8.8" \
    tensorboard \
    tensorly \
    pyyaml \
    matplotlib

# Custom CUDA-built deps (need `module load cuda/12.8` already done).
pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"
pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"

# 7) Make `src/` and `examples/` importable as top-level packages.
pip install --no-cache-dir -e .

# 8) Pin numpy<2 LAST so it survives every dep upgrade above. pycolmap chokes
#    on numpy 2's stricter uint cast (np.uint64(-1) raises OverflowError).
pip install --no-cache-dir "numpy<2.0.0" --force-reinstall

echo
echo "wm2sim env ready. Activate with: conda activate wm2sim"
echo "Remember to 'module load cuda/12.8' before any gsplat / fused-ssim use."
