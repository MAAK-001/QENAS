"""Run one experiment (dataset, method, seed) from start to end — shared by main.py (QENAS) and study.py."""

from __future__ import annotations

import sys
import time
import traceback
from typing import Any, Dict, Optional

from .config import config_fingerprint
from .datasets.discovery import locate_data_root
from .experiment import Experiment, ResumeConflict
from .methods import method_spec
from .pipeline import Pipeline
from .utils.env import environment_info, resolve_device
from .utils.io import save_json
from .utils.logging_utils import format_seconds, setup_logging
from .utils.seed import seed_everything


def run_experiment(cfg: Dict[str, Any], stop_after: Optional[str] = None) -> int:
    """Returns 0 (finished or stopped as requested), 1 (error), 2 (configuration conflict), 130 (interrupted)."""
    t_start = time.time()
    setup_logging()
    # e.g. on Kaggle the datasets live under the read-only /kaggle/input/<dataset-slug>/...
    cfg["data"]["root"] = locate_data_root(cfg["data"]["root"], cfg["data"]["dataset"])
    fp = config_fingerprint(cfg)   # data.root, device and paths are not part of the fingerprint
    try:
        exp = Experiment.open(cfg, fp)
    except ResumeConflict as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    log = setup_logging(exp.logs / "run.log")
    if exp.completed:
        log.info("Experiment %s is already completed; it is not modified.", exp.dir)
        s = exp.dir / "summary.txt"
        if s.exists():
            print(s.read_text(encoding="utf-8"))
        return 0
    device = resolve_device(cfg["device"])
    seed_everything(int(cfg["seed"]))
    env = environment_info(device)
    save_json(exp.dir / "config.json", {"fingerprint": fp, "config": cfg})
    save_json(exp.dir / "environment.json", env)
    log.info("QENAS | dataset %s | method %s | seed %d | experiment %s | device %s | torch %s | python %s | "
             "fingerprint %s", cfg["data"]["dataset"], cfg["method"], cfg["seed"], exp.dir, device,
             env["torch_version"], env["python_version"], fp)
    if device.type == "cpu":
        log.warning("Running on CPU. The training-free search takes hours and the supervised stages days on a CPU; "
                    "a CUDA GPU is strongly recommended.")
    pipeline_class = Pipeline
    if method_spec(cfg["method"])["kind"] == "nnunet":
        from .study.nnunet import NNUNetPipeline as pipeline_class
    try:
        pipeline_class(cfg, exp, device).run(stop_after=stop_after)
    except KeyboardInterrupt:
        log.warning("Interrupted. All completed epochs/generations are checkpointed; re-run the same command to resume.")
        return 130
    except Exception:
        log.error("QENAS stopped with an error:\n%s", traceback.format_exc())
        log.error("Progress up to the last checkpoint is preserved; fix the problem and re-run the same command.")
        return 1
    # Time the experiment took from start to end: the sum of all stage times (counts earlier sessions of a
    # resumed run too) and the wall-clock time of this session.
    total = sum(float(s.get("seconds", 0.0)) for s in exp.state["stages"].values())
    exp.state["total_seconds"] = round(total, 1)
    exp.save_state()
    line = (f"QENAS time on {cfg['data']['dataset']} ({cfg['method']}, seed {cfg['seed']}): {format_seconds(total)} "
            f"from start to end (all stages) | this session: {format_seconds(time.time() - t_start)}")
    log.info(line)
    if exp.completed and (exp.dir / "summary.txt").exists():
        with open(exp.dir / "summary.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return 0
