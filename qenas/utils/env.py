"""Environment capture (versions, device) for reproducibility records, and device selection."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict

import numpy as np
import torch


def resolve_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Device '{spec}' requested but CUDA is not available in this PyTorch build "
                           f"(torch {torch.__version__}). Use --device cpu.")
    return torch.device(spec)


def environment_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device": str(device),
        "numpy_version": np.__version__,
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / 1024**3, 2)
    for mod in ("pymoo", "scipy", "PIL", "matplotlib", "yaml"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = None
    return info


def is_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a (GPU or CPU allocator) out-of-memory error."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "not enough memory" in msg or "can't allocate memory" in msg
