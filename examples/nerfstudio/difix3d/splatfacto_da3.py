"""Splatfacto variant that initializes its gaussians from a DA3-produced 3DGS PLY.

The PLY layout matches splatfacto's own export (and DA3's `export_to_gs_ply`):
- positions x,y,z
- DC SH features f_dc_{0..2}
- (optional) higher-order SH features f_rest_*  (DA3 with save_sh_dc_only=True omits these)
- opacity (pre-sigmoid)
- scale_{0..2} (log)
- rot_{0..3} (wxyz)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch
from plyfile import PlyData

from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
    num_sh_bases,
)


@dataclass
class SplatfactoDA3ModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoDA3Model)
    init_ply_path: Optional[Path] = None
    """Path to a 3DGS PLY (e.g. produced by DA3) to use as initialization. Replaces
    the SfM-seeded gaussians with the full means/scales/rotations/opacities/SH from the PLY."""


def _load_3dgs_ply(ply_path: Path, sh_degree: int) -> dict[str, torch.Tensor]:
    data = PlyData.read(str(ply_path))
    v = data["vertex"]
    n = len(v)

    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    opacities = np.asarray(v["opacity"], dtype=np.float32).reshape(n, 1)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)

    dim_sh = num_sh_bases(sh_degree)
    rest_props = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    expected_rest = (dim_sh - 1) * 3
    if len(rest_props) == 0:
        f_rest = np.zeros((n, dim_sh - 1, 3), dtype=np.float32)
    else:
        assert len(rest_props) == expected_rest, (
            f"PLY has {len(rest_props)} f_rest_* properties, expected {expected_rest} for sh_degree={sh_degree}"
        )
        # Splatfacto/Inria PLY convention: f_rest stored as (3, dim_sh-1) flattened, i.e.
        # channel-major. We store back to (n, dim_sh-1, 3).
        rest = np.stack([np.asarray(v[p], dtype=np.float32) for p in rest_props], axis=1)  # (n, 3*(dim_sh-1))
        f_rest = rest.reshape(n, 3, dim_sh - 1).transpose(0, 2, 1)

    return {
        "means": torch.from_numpy(means),
        "scales": torch.from_numpy(scales),
        "quats": torch.from_numpy(quats),
        "features_dc": torch.from_numpy(f_dc),
        "features_rest": torch.from_numpy(f_rest),
        "opacities": torch.from_numpy(opacities),
    }


class SplatfactoDA3Model(SplatfactoModel):
    config: SplatfactoDA3ModelConfig

    def populate_modules(self):
        super().populate_modules()
        if self.config.init_ply_path is None:
            return

        ply_path = Path(self.config.init_ply_path)
        params = _load_3dgs_ply(ply_path, self.config.sh_degree)

        # Reuse the device of the existing parameters
        device = self.gauss_params["means"].device
        new_params = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.to(device)) for k, v in params.items()}
        )
        self.gauss_params = new_params

        n = params["means"].shape[0]
        print(f"[splatfacto-da3] loaded {n} gaussians from {ply_path}")
