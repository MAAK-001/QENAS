"""Image/mask loading and resizing.

* Images: converted to 3-channel RGB (grayscale replicated), resized with
  **bilinear** interpolation (Pillow's bilinear filter is anti-aliased when
  down-sampling, which matters for 4288x2848 IDRiD fundus photographs).
* Masks: union of all mask files, binarised at the *original* resolution, then
  resized with **nearest-neighbour** interpolation (never bilinear) and stored as
  {0, 1} uint8.
* Resized arrays are cached on disk (``.npz``), keyed by file identities and all
  preprocessing parameters, so repeated runs do not re-decode the raw files.
* Normalisation statistics are estimated on the **training split only**.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from ..config import QENAS_ROOT
from ..utils.io import retry_on_lock
from ..utils.logging_utils import ProgressTimer, get_logger
from .discovery import DatasetError, Sample, read_binary_mask_union

log = get_logger()

PREPROCESS_VERSION = 2   # bump if the preprocessing code changes (invalidates the cache)


def load_and_resize(sample: Sample, size_hw: Tuple[int, int], threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = size_hw
    with Image.open(sample.image) as im:
        im = im.convert("RGB")
        orig_wh = im.size
        img = np.asarray(im.resize((w, h), resample=Image.BILINEAR), dtype=np.uint8)
    mask = read_binary_mask_union(sample.masks, threshold, size_hw=(orig_wh[1], orig_wh[0]))
    if mask.shape != (orig_wh[1], orig_wh[0]):
        raise DatasetError(f"Mask/image size mismatch for {sample.sample_id}: mask {mask.shape[::-1]} vs image {orig_wh}")
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_r = np.asarray(mask_img.resize((w, h), resample=Image.NEAREST), dtype=np.uint8)
    mask_r = (mask_r > 127).astype(np.uint8)
    if img.shape != (h, w, 3) or mask_r.shape != (h, w):
        raise DatasetError(f"Unexpected tensor shapes after resizing {sample.sample_id}: {img.shape}, {mask_r.shape}")
    return img, mask_r


def _cache_key(samples: Sequence[Sample], size_hw: Tuple[int, int], threshold: int, dataset: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"v": PREPROCESS_VERSION, "size": list(size_hw), "thr": threshold, "ds": dataset}).encode())
    for s in samples:
        h.update(s.sample_id.encode())
        for p in [s.image] + list(s.masks):
            st = os.stat(p)
            h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()[:24]


def preload_split(samples: Sequence[Sample], size_hw: Tuple[int, int], threshold: int, dataset: str,
                  split_name: str, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (images [N,H,W,3] uint8, masks [N,H,W] uint8 in {0,1}) for a split."""
    cache_file = None
    if use_cache:
        key = _cache_key(samples, size_hw, threshold, dataset)
        cache_dir = QENAS_ROOT / "cache" / "preprocessed"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{dataset}_{split_name}_{size_hw[0]}x{size_hw[1]}_{key}.npz"
        if cache_file.exists():
            try:
                with np.load(cache_file) as z:
                    imgs, masks = z["images"], z["masks"]
                if imgs.shape[0] == len(samples) and set(np.unique(masks).tolist()) <= {0, 1}:
                    log.info("Loaded cached preprocessed %s/%s (%d samples) from %s", dataset, split_name,
                             len(samples), cache_file.name)
                    return imgs, masks
                log.warning("Preprocessing cache %s is inconsistent; rebuilding", cache_file.name)
            except Exception as exc:
                log.warning("Preprocessing cache %s unreadable (%s); rebuilding", cache_file.name, exc)
    n = len(samples)
    imgs = np.zeros((n, size_hw[0], size_hw[1], 3), dtype=np.uint8)
    masks = np.zeros((n, size_hw[0], size_hw[1]), dtype=np.uint8)
    timer = ProgressTimer(n)
    for i, s in enumerate(samples):
        imgs[i], masks[i] = load_and_resize(s, size_hw, threshold)
        timer.step()
        if (i + 1) % max(1, n // 5) == 0 or i + 1 == n:
            log.info("  preprocessing %s/%s: %d/%d | %s", dataset, split_name, i + 1, n, timer.summary())
    if cache_file is not None:
        tmp = cache_file.with_name(cache_file.stem + ".tmp.npz")
        np.savez_compressed(tmp, images=imgs, masks=masks)
        retry_on_lock(os.replace, tmp, cache_file)
    return imgs, masks


def channel_stats(images: np.ndarray) -> Tuple[List[float], List[float]]:
    """Per-channel mean/std in [0, 1] units (computed in float64, streaming)."""
    n = images.shape[0]
    s = np.zeros(3, dtype=np.float64)
    ss = np.zeros(3, dtype=np.float64)
    cnt = 0
    for i in range(n):
        x = images[i].reshape(-1, 3).astype(np.float64) / 255.0
        s += x.sum(0)
        ss += (x * x).sum(0)
        cnt += x.shape[0]
    mean = s / cnt
    std = np.sqrt(np.maximum(ss / cnt - mean**2, 1e-12))
    return mean.tolist(), std.tolist()


def trivial_foreground_dice(masks: np.ndarray) -> float:
    """Mean per-image Dice of the trivial 'everything is foreground' predictor (competence floor)."""
    m = masks.reshape(masks.shape[0], -1).astype(np.float64)
    g = m.sum(1)
    n = m.shape[1]
    return float(np.mean(2.0 * g / (g + n)))
