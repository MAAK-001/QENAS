"""Corruption-resistant checkpointing.

Every save writes atomically (temporary file + ``os.replace``) and keeps the
previous checkpoint as ``<name>.prev``. Loading tries the newest file first and
falls back to the previous one if the newest is unreadable (e.g. the process
was killed or the disk filled up mid-write). Corruption is reported, never
silently ignored.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

from .io import atomic_write_bytes, retry_on_lock

log = logging.getLogger("qenas")


class CheckpointError(RuntimeError):
    """Raised when no readable checkpoint exists although one was expected."""


def save_checkpoint(path: os.PathLike, state: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(state, buf)
    data = buf.getvalue()
    if path.exists():
        # Keep the last good checkpoint as a fallback.
        retry_on_lock(shutil.copy2, path, path.with_name(path.name + ".prev"))
    atomic_write_bytes(path, data)


def _try_load(path: Path, map_location: Any) -> Any:
    # weights_only=False: checkpoints contain RNG states, histories and configs we wrote ourselves.
    return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint(path: os.PathLike, map_location: Any = "cpu", required: bool = False) -> Optional[Any]:
    """Load a checkpoint, falling back to ``<path>.prev`` if the newest file is corrupted."""
    path = Path(path)
    candidates = [path, path.with_name(path.name + ".prev")]
    errors = []
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            state = _try_load(cand, map_location)
            if cand != path:
                log.warning("Checkpoint %s was unreadable; recovered from fallback %s", path, cand)
            return state
        except Exception as exc:  # corrupted / truncated file
            errors.append(f"{cand}: {type(exc).__name__}: {exc}")
            log.error("Corrupted checkpoint detected: %s (%s)", cand, exc)
    if errors:
        raise CheckpointError("All checkpoint files are corrupted:\n  " + "\n  ".join(errors))
    if required:
        raise CheckpointError(f"Required checkpoint not found: {path}")
    return None
