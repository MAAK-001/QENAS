"""QENAS — Quick Evolutionary NAS.  Command-line entry point (one pipeline, one protocol).

    python main.py --dataset BUSI
    python main.py --dataset CVC
    python main.py --dataset IDRID

An interrupted run is resumed automatically by re-running the same command. The only options are run settings
(where the data is, which device, data-loader workers, seed) and experiment control; the method itself is fixed in
qenas/config.py.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from qenas.config import SUPPORTED_DATASETS, build_config, config_fingerprint
from qenas.datasets.discovery import locate_data_root
from qenas.experiment import Experiment, ResumeConflict
from qenas.pipeline import STAGES, Pipeline
from qenas.utils.env import environment_info, resolve_device
from qenas.utils.io import save_json
from qenas.utils.logging_utils import setup_logging
from qenas.utils.seed import seed_everything


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QENAS: training-free NSGA-II (parameters, SynFlow, ZiCo) + "
                                            "learning-capability selection for medical image segmentation")
    p.add_argument("--dataset", required=True, type=str.upper, choices=SUPPORTED_DATASETS)
    p.add_argument("--data-root", default=None, help="folder containing BUSI*/CVC*/IDRID* (default: ../Datasets)")
    p.add_argument("--device", default=None, help="auto (default) | cpu | cuda | cuda:N")
    p.add_argument("--num-workers", type=int, default=None, help="data-loader worker processes (default 0)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default 42); repeat runs use other seeds")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="start a new experiment instead of resuming / reporting the latest one")
    p.add_argument("--stop-after", default=None, choices=STAGES,
                   help="stop cleanly after this stage; the same command without it continues from there")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = build_config(a.dataset, overrides={"data.root": a.data_root, "device": a.device,
                                             "data.num_workers": a.num_workers, "seed": a.seed})
    setup_logging()
    # e.g. on Kaggle the datasets live under the read-only /kaggle/input/<dataset-slug>/...
    cfg["data"]["root"] = locate_data_root(cfg["data"]["root"], cfg["data"]["dataset"])
    fp = config_fingerprint(cfg)   # data.root, device and paths are not part of the fingerprint
    try:
        found = Experiment.find_resumable(cfg, fp) if a.resume else None
        exp = Experiment.open_existing(found, cfg, fp) if found else Experiment.create(cfg, fp)
    except ResumeConflict as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    log = setup_logging(exp.logs / "run.log")
    if exp.completed:
        log.info("Experiment %s is already completed; it will not be modified (use --no-resume for a new one).",
                 exp.dir)
        s = exp.dir / "summary.txt"
        if s.exists():
            print(s.read_text(encoding="utf-8"))
        return 0
    device = resolve_device(cfg["device"])
    seed_everything(int(cfg["seed"]))
    env = environment_info(device)
    save_json(exp.dir / "config.json", {"fingerprint": fp, "config": cfg})
    save_json(exp.dir / "environment.json", env)
    log.info("QENAS | dataset %s | experiment %s | device %s | torch %s | python %s | seed %d | fingerprint %s",
             cfg["data"]["dataset"], exp.dir, device, env["torch_version"], env["python_version"], cfg["seed"], fp)
    if device.type == "cpu":
        log.warning("Running on CPU. The training-free search takes hours and the supervised stages days on a CPU; "
                    "a CUDA GPU is strongly recommended.")
    try:
        Pipeline(cfg, exp, device).run(stop_after=a.stop_after)
    except KeyboardInterrupt:
        log.warning("Interrupted. All completed epochs/generations are checkpointed; re-run the same command to resume.")
        return 130
    except Exception:
        log.error("QENAS stopped with an error:\n%s", traceback.format_exc())
        log.error("Progress up to the last checkpoint is preserved; fix the problem and re-run the same command.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
