"""Experiment directories, stage bookkeeping and safe resume.

Layout::

    results/<DATASET>/experiment_YYYYMMDD_HHMMSS/
        state.json            stage completion, fingerprint, test-evaluation flag
        config.json           fully resolved configuration (+ fingerprint)
        environment.json      Python / PyTorch / CUDA / device information
        ...                   stage outputs (CSV / JSON / Markdown)
        checkpoints/          resumable checkpoints of every stage
        figures/              publication figures (PNG)
        logs/run.log

Resume policy
* default: the most recent experiment of the dataset with the identical configuration
  fingerprint is opened - an unfinished one is resumed, a finished one is reported and never
  modified. Otherwise the most recent *unfinished* experiment is opened, which stops with the
  list of differing keys (configurations are never mixed);
* ``--no-resume``: always start a new experiment directory.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Optional

from .config import diff_configs
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
        return Path(cfg["paths"]["results"]) / cfg["data"]["dataset"]

    @classmethod
    def create(cls, cfg: Dict[str, Any], fingerprint: str) -> "Experiment":
        base = cls.dataset_dir(cfg)
        base.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = base / f"experiment_{stamp}"
        k = 1
        while d.exists():
            d = base / f"experiment_{stamp}_{k}"
            k += 1
        d.mkdir(parents=True)
        return cls(d, cfg, fingerprint)

    @classmethod
    def open_existing(cls, directory: Path, cfg: Dict[str, Any], fingerprint: str) -> "Experiment":
        directory = Path(directory)
        st = load_json(directory / "state.json")
        if st.get("fingerprint") != fingerprint:
            old_cfg = load_json(directory / "config.json")["config"] if (directory / "config.json").exists() else {}
            diffs = diff_configs(old_cfg, cfg)
            lines = "\n".join(f"    {k}: experiment={a!r}  current={b!r}" for k, (a, b) in diffs.items())
            raise ResumeConflict(f"Configuration differs from the unfinished experiment {directory.name}; refusing to "
                                 f"resume.\n{lines}\n  Use --no-resume to start a new experiment.")
        return cls(directory, cfg, fingerprint)

    @classmethod
    def find_resumable(cls, cfg: Dict[str, Any], fingerprint: str) -> Optional[Path]:
        """Most recent experiment with the identical fingerprint, finished or not. Otherwise the most recent
        *unfinished* experiment (opening it reports the configuration conflict), else None."""
        base = cls.dataset_dir(cfg)
        if not base.exists():
            return None
        cands = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("experiment_")
                        and (p / "state.json").exists()], key=lambda p: p.name)
        states = []
        for p in reversed(cands):
            try:
                states.append((p, load_json(p / "state.json")))
            except Exception:
                continue
        for p, st in states:
            if st.get("fingerprint") == fingerprint:
                return p
        for p, st in states:
            if st.get("status") != "completed":
                return p
        return None


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")
