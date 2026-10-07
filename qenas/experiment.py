"""Experiment directories, stage bookkeeping and safe resume.

Layout::

    results/<DATASET>/experiment_YYYYMMDD_HHMMSS/
        state.json            stage completion, fingerprint, test-evaluation flag
        config.json           fully resolved configuration (+ fingerprint)
        environment.json      Python / PyTorch / CUDA / device information
        ...                   stage outputs (CSV / JSON / Markdown)
        checkpoints/          resumable checkpoints of every stage
        figures/              publication figures (PNG + PDF)
        logs/run.log

Resume policy
* default: the most recent *unfinished* experiment of the dataset is resumed if its
  configuration fingerprint equals the current one; if it differs the run stops and
  lists the differing keys (never mixes configurations);
* ``--no-resume``: always start a new experiment directory;
* ``--experiment-dir``: resume exactly that directory;
* completed experiments are never modified — a new directory is created instead.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Optional

from .config import QENAS_ROOT, diff_configs
from .utils.io import load_json, save_json


class ResumeConflict(RuntimeError):
    pass


class Experiment:
    def __init__(self, directory: Path, cfg: Dict[str, Any], fingerprint: str):
        self.dir = Path(directory)
        self.cfg = cfg
        self.fingerprint = fingerprint
        self.figures = self.dir / "figures"
        self.checkpoints = self.dir / "checkpoints"
        self.logs = self.dir / "logs"
        for d in (self.figures, self.checkpoints, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        if self.state_path.exists():
            self.state = load_json(self.state_path)
        else:
            self.state = {"fingerprint": fingerprint, "created": _now(), "status": "running", "stages": {},
                          "test_evaluated": False}
            self.save_state()

    # ------------------------------------------------------------------ state
    def save_state(self) -> None:
        self.state["updated"] = _now()
        save_json(self.state_path, self.state)

    def is_done(self, stage: str) -> bool:
        return self.state["stages"].get(stage, {}).get("status") == "done"

    def mark_done(self, stage: str, info: Optional[Dict[str, Any]] = None) -> None:
        self.state["stages"][stage] = {"status": "done", "finished": _now(), **(info or {})}
        self.save_state()

    def mark_completed(self) -> None:
        self.state["status"] = "completed"
        self.save_state()

    @property
    def completed(self) -> bool:
        return self.state.get("status") == "completed"

    # ------------------------------------------------------------------ discovery
    @staticmethod
    def dataset_dir(cfg: Dict[str, Any]) -> Path:
        return QENAS_ROOT / "results" / cfg["data"]["dataset"]

    @staticmethod
    def _prefix(cfg: Dict[str, Any]) -> str:
        return "smoke_" if cfg.get("smoke_test") else "experiment_"

    @classmethod
    def create(cls, cfg: Dict[str, Any], fingerprint: str) -> "Experiment":
        base = cls.dataset_dir(cfg)
        base.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = base / f"{cls._prefix(cfg)}{stamp}"
        k = 1
        while d.exists():
            d = base / f"{cls._prefix(cfg)}{stamp}_{k}"
            k += 1
        d.mkdir(parents=True)
        return cls(d, cfg, fingerprint)

    @classmethod
    def open_existing(cls, directory: Path, cfg: Dict[str, Any], fingerprint: str) -> "Experiment":
        directory = Path(directory)
        if not (directory / "state.json").exists():
            raise ResumeConflict(f"{directory} is not a QENAS experiment directory (no state.json)")
        st = load_json(directory / "state.json")
        if st.get("fingerprint") != fingerprint:
            old_cfg = load_json(directory / "config.json")["config"] if (directory / "config.json").exists() else {}
            diffs = diff_configs(old_cfg, cfg)
            lines = "\n".join(f"    {k}: experiment={a!r}  current={b!r}" for k, (a, b) in diffs.items())
            raise ResumeConflict(f"Configuration differs from experiment {directory.name}; refusing to resume.\n"
                                 f"{lines}\n  Re-run with the original options, or use --no-resume to start "
                                 f"a new experiment.")
        return cls(directory, cfg, fingerprint)

    @classmethod
    def find_resumable(cls, cfg: Dict[str, Any]) -> Optional[Path]:
        base = cls.dataset_dir(cfg)
        if not base.exists():
            return None
        cands = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(cls._prefix(cfg))
                        and (p / "state.json").exists()], key=lambda p: p.name)
        for p in reversed(cands):
            try:
                st = load_json(p / "state.json")
            except Exception:
                continue
            if st.get("status") != "completed":
                return p
        return None


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")
