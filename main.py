"""QENAS — Quick Evolutionary NAS.  Command-line entry point (one pipeline, one protocol).

    python main.py --dataset BUSI
    python main.py --dataset CVC
    python main.py --dataset IDRID
    python main.py --dataset KVASIR

Results: results/<DATASET>/qenas/seed_<seed>/. An interrupted run is resumed automatically by re-running the same
command; a finished one is reported and never modified. The only options are run settings (where the data is,
which device, data-loader workers, seed) and --stop-after; the method itself is fixed in qenas/config.py.
The evaluation study of the paper (baselines, ablations, repeated seeds) is run with study.py.
"""

from __future__ import annotations

import argparse
import sys

from qenas.config import SUPPORTED_DATASETS, build_config
from qenas.pipeline import STAGES
from qenas.runner import run_experiment


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QENAS: training-free NSGA-II (parameters, SynFlow, ZiCo) + "
                                            "learning-capability selection for medical image segmentation")
    p.add_argument("--dataset", required=True, type=str.upper, choices=SUPPORTED_DATASETS)
    p.add_argument("--data-root", default=None, help="folder containing BUSI*/CVC*/IDRID*/Kvasir* (default: ../Datasets)")
    p.add_argument("--device", default=None, help="auto (default) | cpu | cuda | cuda:N")
    p.add_argument("--num-workers", type=int, default=None, help="data-loader worker processes (default 0)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default 42); the split is always the same")
    p.add_argument("--stop-after", default=None, choices=STAGES,
                   help="stop cleanly after this stage; the same command without it continues from there")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = build_config(a.dataset, overrides={"data.root": a.data_root, "device": a.device,
                                             "data.num_workers": a.num_workers, "seed": a.seed})
    return run_experiment(cfg, stop_after=a.stop_after)


if __name__ == "__main__":
    sys.exit(main())
