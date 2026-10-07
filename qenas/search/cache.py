"""Cache of training-free evaluations.

Key = SHA-256 of every factor that influences the objectives: chromosome, the full
architecture signature (channels, scale, DARTS genotype definition, network
version), the SynFlow configuration (dtype, input shape, initialisation seed,
batch size, objective version) and the PyTorch version. One small JSON file per
key, written atomically -> safe to share between concurrent runs and experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..utils.io import load_json, save_json
from ..utils.logging_utils import get_logger

log = get_logger()


class EvaluationCache:
    def __init__(self, directory: Path, context: Dict[str, Any]):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.context_text = json.dumps(context, sort_keys=True, default=str)
        self.hits = 0
        self.misses = 0

    def key(self, chromosome: Sequence[int]) -> str:
        payload = json.dumps({"chromosome": [int(g) for g in chromosome]}, sort_keys=True) + self.context_text
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, chromosome: Sequence[int]) -> Optional[Dict[str, Any]]:
        k = self.key(chromosome)
        p = self._path(k)
        if not p.exists():
            self.misses += 1
            return None
        try:
            rec = load_json(p)
            res = rec["result"]
            if rec.get("key") != k or [int(g) for g in res["chromosome"]] != [int(g) for g in chromosome]:
                raise ValueError("cache record does not match its key")
            for f in ("f1_params", "f2_synflow", "synflow_raw"):
                v = float(res[f])
                if not math.isfinite(v):
                    raise ValueError(f"non-finite cached value {f}={v}")
        except Exception as exc:
            log.warning("Ignoring corrupted cache entry %s (%s)", p, exc)
            self.misses += 1
            return None
        self.hits += 1
        return dict(res)

    def put(self, chromosome: Sequence[int], result: Dict[str, Any]) -> None:
        k = self.key(chromosome)
        save_json(self._path(k), {"key": k, "context": self.context, "result": result})
