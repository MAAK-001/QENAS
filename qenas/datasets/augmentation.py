"""Segmentation-preserving paired augmentation (training split only).

Geometric transforms are sampled once per sample and applied identically to the
image (bilinear) and the mask (nearest-neighbour). Photometric transforms touch
the image only. All randomness comes from a ``numpy.random.Generator`` derived
from (seed, epoch, sample index), so augmentation is reproducible and independent
of the number of data-loader workers and of interruptions/resumes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def augment_pair(img: torch.Tensor, mask: torch.Tensor, cfg: Dict[str, Any],
                 rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """``img``: float [3,H,W] in [0,1]; ``mask``: float [1,H,W] in {0,1}."""
    if rng.random() < cfg.get("hflip", 0.0):
        img, mask = img.flip(-1), mask.flip(-1)
    if rng.random() < cfg.get("vflip", 0.0):
        img, mask = img.flip(-2), mask.flip(-2)

    if rng.random() < cfg.get("affine_p", 0.0):
        h, w = img.shape[-2:]
        ang = math.radians(rng.uniform(-cfg.get("rotate_deg", 0.0), cfg.get("rotate_deg", 0.0)))
        lo, hi = cfg.get("scale", [1.0, 1.0])
        sc = rng.uniform(lo, hi)
        t = cfg.get("translate", 0.0)
        tx, ty = rng.uniform(-t, t) * 2.0, rng.uniform(-t, t) * 2.0   # normalised coords span [-1, 1]
        cos, sin = math.cos(ang) / sc, math.sin(ang) / sc
        # inverse mapping output->input in normalised coordinates, aspect-ratio corrected
        ar = w / h
        theta = torch.tensor([[cos, -sin / ar, tx],
                              [sin * ar, cos, ty]], dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(theta, size=(1, 1, h, w), align_corners=False)
        img = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False).squeeze(0)
        mask = F.grid_sample(mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros",
                             align_corners=False).squeeze(0)

    b = cfg.get("brightness", 0.0)
    c = cfg.get("contrast", 0.0)
    if b > 0 or c > 0:
        cf = rng.uniform(1.0 - c, 1.0 + c) if c > 0 else 1.0
        bf = rng.uniform(-b, b) if b > 0 else 0.0
        mean = img.mean()
        img = ((img - mean) * cf + mean + bf).clamp_(0.0, 1.0)
    return img, mask
