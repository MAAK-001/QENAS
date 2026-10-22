"""Experiment directories, stage bookkeeping and safe resume.

An experiment is identified by (dataset, method, seed) and lives in a fixed directory::

    results/<DATASET>/<method>/seed_<seed>/
        state.json            stage completion, fingerprint, test-evaluation flag
        config.json           fully resolved configuration (+ fingerprint)
        environment.json      Python / PyTorch / CUDA / device information
        ...                   stage outputs (CSV / JSON / Markdown)
        checkpoints/          resumable checkpoints of every stage
        figures/              publication figures (PNG)
        logs/run.log

Resume policy: running the same (dataset, method, seed) again opens the same directory. With the identical
configuration fingerprint an unfinished experiment is resumed and a finished one is reported and never
modified. A different configuration stops with the list of differing keys (configurations are never mixed);
delete the directory to start that experiment again.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Optional

from .config import diff_configs
from .utils.io import load_json, save_json


class ResumeConflict(RuntimeError):
    pass


def dataset_dirname(cfg: Dict[str, Any]) -> str:
    return cfg["data"]["dataset"]


def experiment_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["results"]) / dataset_dirname(cfg) / cfg["method"] / f"seed_{int(cfg['seed'])}"


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
                          "test_evaluated": False, "method": cfg["method"], "dataset": cfg["data"]["dataset"],
                          "seed": int(cfg["seed"])}
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

    # ------------------------------------------------------------------ opening
    @classmethod
    def open(cls, cfg: Dict[str, Any], fingerprint: str) -> "Experiment":
        """The experiment of (dataset, method, seed): resumed if it exists with the same configuration."""
        d = experiment_dir(cfg)
        if (d / "state.json").exists():
            st = load_json(d / "state.json")
            if st.get("fingerprint") != fingerprint:
                old_cfg = load_json(d / "config.json")["config"] if (d / "config.json").exists() else {}
                diffs = diff_configs(old_cfg, cfg)
                lines = "\n".join(f"    {k}: experiment={a!r}  current={b!r}" for k, (a, b) in diffs.items())
                raise ResumeConflict(f"{d} was created with a different configuration; refusing to resume.\n"
                                     f"{lines}\n  Delete that directory to run this experiment again.")
        return cls(d, cfg, fingerprint)


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")
