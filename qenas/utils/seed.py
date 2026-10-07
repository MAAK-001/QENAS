"""Seeding and random-state capture/restore for exact reproducibility and resume."""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs.

    ``deterministic=True`` asks PyTorch for deterministic kernels where available
    (``warn_only`` so that ops without a deterministic implementation still run).
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - very old torch
            pass


def derive_seed(*parts: Any) -> int:
    """Deterministically derive a 31-bit seed from arbitrary parts (stable across runs)."""
    import hashlib

    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def get_rng_states() -> Dict[str, Any]:
    """Capture all global RNG states (stored inside checkpoints)."""
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restore global RNG states captured by :func:`get_rng_states`."""
    if not states:
        return
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"].cpu() if hasattr(states["torch"], "cpu") else states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(states["cuda"])
        except Exception:
            pass
