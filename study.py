"""The QENAS evaluation study (STUDY.md): QENAS vs baselines, ablations, random search, external validation and the
CVC leakage study, 3 seeds each, identical splits. QENAS itself is run exactly as by main.py.

    python study.py run --gpus 0 1            # run / resume the whole plan (Kaggle: both T4 GPUs)
    python study.py run --gpus cpu            # same on the CPU (slow)
    python study.py status                    # which jobs are done
    python study.py analyse                   # paper tables, statistics and figures -> results/analysis/
    python study.py figures                   # regenerate every experiment's figures from its saved results
    python study.py archive results.zip       # results without model checkpoints, as one zip

Internal (started by "run" for every job):
    python study.py job --dataset BUSI --method unet --seed 42
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from qenas.config import DEFAULT_CONFIG, SPLIT_UNITS, SUPPORTED_DATASETS
from qenas.methods import METHODS


def cmd_job(a) -> int:
    from qenas.pipeline import study_config
    from qenas.runner import run_experiment
    from qenas.study.plan import Job
    from qenas.utils.env import resolve_device
    cfg = study_config(a.dataset, a.method, a.seed, split_unit=a.split_unit,
                       overrides={"data.root": a.data_root, "device": a.device, "data.num_workers": a.num_workers,
                                  "paths.results": a.results})
    rc = run_experiment(cfg)
    if rc == 0 and Job(a.dataset, a.method, a.seed, a.split_unit).external:
        from qenas.study.external import external_validation
        external_validation(cfg, resolve_device(cfg["device"]))
    return rc


def cmd_run(a) -> int:
    from qenas.study.scheduler import run_study
    gpus = [None] if a.gpus == ["cpu"] else [int(g) for g in a.gpus]
    run_study(Path(a.results), gpus, a.hours, data_root=a.data_root, num_workers=a.num_workers,
              threads_per_job=a.threads)
    return 0


def cmd_status(a) -> int:
    from qenas.study.plan import plan
    from qenas.study.scheduler import job_done
    jobs = plan()
    done = [j for j in jobs if job_done(Path(a.results), j)]
    for j in jobs:
        print(f"{'done   ' if j in done else 'pending'}  {j.id}")
    print(f"\n{len(done)}/{len(jobs)} jobs done")
    return 0


def cmd_analyse(a) -> int:
    from qenas.study.analysis import analyse
    analyse(Path(a.results), data_root=a.data_root)
    return 0


def cmd_figures(a) -> int:
    from qenas.study.figures import regenerate
    regenerate(Path(a.results))
    return 0


def cmd_archive(a) -> int:
    root = Path(a.results)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and "checkpoints" not in p.relative_to(root).parts:
                z.write(p, Path("results") / p.relative_to(root))
    print(f"wrote {a.out}")
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results", default=DEFAULT_CONFIG["paths"]["results"], help="results folder")
    common.add_argument("--data-root", default=None, help="folder containing the dataset folders")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", parents=[common], help="run / resume the study plan")
    r.add_argument("--gpus", nargs="+", default=["0"], help="GPU indices, e.g. 0 1; or 'cpu'")
    r.add_argument("--hours", type=float, default=11.3, help="start no job that would not finish within this time")
    r.add_argument("--num-workers", type=int, default=1, help="data-loader workers per job")
    r.add_argument("--threads", type=int, default=2, help="CPU threads per job")
    j = sub.add_parser("job", parents=[common], help="run one experiment (used by 'run')")
    j.add_argument("--dataset", required=True, type=str.upper, choices=SUPPORTED_DATASETS)
    j.add_argument("--method", required=True, choices=sorted(METHODS))
    j.add_argument("--seed", required=True, type=int)
    j.add_argument("--split-unit", default="dataset", choices=SPLIT_UNITS)
    j.add_argument("--device", default="auto")
    j.add_argument("--num-workers", type=int, default=0)
    sub.add_parser("status", parents=[common], help="list done / pending jobs")
    sub.add_parser("analyse", parents=[common], help="write the paper tables, statistics and figures")
    sub.add_parser("figures", parents=[common], help="regenerate every experiment's figures from its saved results")
    ar = sub.add_parser("archive", parents=[common], help="zip the results without checkpoints")
    ar.add_argument("out")
    a = p.parse_args(argv)
    return {"job": cmd_job, "run": cmd_run, "status": cmd_status, "analyse": cmd_analyse, "figures": cmd_figures,
            "archive": cmd_archive}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
