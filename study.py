"""The QENAS evaluation study v2 (STUDY.md): QENAS vs U-Net, Attention U-Net and nnU-Net, the loss-coefficient and
selection ablations, random search and external validation, 3 seeds each, identical splits. QENAS itself is run
exactly as by main.py.

    python study.py run --gpus 0 1            # run / resume the whole plan (Kaggle: both T4 GPUs)
    python study.py run --gpus cpu            # same on the CPU (slow)
    python study.py status                    # which jobs are done
    python study.py run --plan ablation --worker A --gpus 0 1   # component ablation study, Kaggle account A (or B)
    python study.py status --plan ablation    # progress of the whole ablation study (36 experiments)
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

from qenas.config import DEFAULT_CONFIG, SUPPORTED_DATASETS
from qenas.methods import METHODS


def cmd_job(a) -> int:
    from qenas.pipeline import study_config
    from qenas.runner import run_experiment
    from qenas.study.plan import Job
    from qenas.utils.env import resolve_device
    cfg = study_config(a.dataset, a.method, a.seed,
                       overrides={"data.root": a.data_root, "device": a.device, "data.num_workers": a.num_workers,
                                  "paths.results": a.results})
    rc = run_experiment(cfg)
    if rc == 0 and Job(a.dataset, a.method, a.seed).external:
        from qenas.study.external import ExternalUnavailable, external_validation, unavailable_path
        from qenas.utils.io import save_json
        try:
            external_validation(cfg, resolve_device(cfg["device"]))
        except ExternalUnavailable as exc:   # recorded, so that the job does not fail in every later session
            print(f"[external] not available: {exc}", file=sys.stderr)
            save_json(unavailable_path(cfg), {"reason": str(exc)})
    if rc == 0 and METHODS[a.method]["kind"] == "nnunet":
        # nnU-Net's working folder (exported images, preprocessed data, weights) is no longer needed once the
        # experiment and its external validation are finished; its plans and logs are kept in <experiment>/nnunet.
        import shutil
        from qenas.experiment import experiment_dir
        shutil.rmtree(experiment_dir(cfg) / "nnunet_work", ignore_errors=True)
    return rc


def _status_name(a) -> str:
    return "study_status.json" if a.plan == "v2" else f"ablation_status{'_' + a.worker.upper() if a.worker else ''}.json"


def cmd_run(a) -> int:
    import time
    from qenas.study.plan import select_plan
    from qenas.study.scheduler import run_study
    gpus = [None] if a.gpus == ["cpu"] else [int(g) for g in a.gpus]

    def log(text):
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + text, flush=True)
    run_study(Path(a.results), gpus, a.hours, data_root=a.data_root, num_workers=a.num_workers,
              threads_per_job=a.threads, jobs=select_plan(a.plan, a.worker), status_name=_status_name(a), log=log)
    return 0


def cmd_status(a) -> int:
    from qenas.study.plan import select_plan
    from qenas.study.scheduler import job_done, stage_progress
    res = Path(a.results)
    jobs = select_plan(a.plan, a.worker)
    done = [j for j in jobs if job_done(res, j)]
    for j in jobs:
        print(f"{'done   ' if j in done else 'pending'}  {j.id}" + ("" if j in done else f"  ({stage_progress(res, j)})"))
    print(f"\n{len(done)}/{len(jobs)} jobs done ({100 * len(done) / max(1, len(jobs)):.0f} %), "
          f"{len(jobs) - len(done)} left")
    return 0 if not a.check_complete or len(done) == len(jobs) else 3


def cmd_analyse(a) -> int:
    from qenas.study.analysis import analyse
    analyse(Path(a.results), data_root=a.data_root, old_results=Path(a.old_results))
    return 0


def cmd_figures(a) -> int:
    from qenas.study.figures import regenerate
    regenerate(Path(a.results))
    return 0


def cmd_archive(a) -> int:
    root = Path(a.results)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            parts = p.relative_to(root).parts
            if p.is_file() and "checkpoints" not in parts and "nnunet_work" not in parts:
                z.write(p, Path("results") / p.relative_to(root))
    print(f"wrote {a.out}")
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results", default=DEFAULT_CONFIG["paths"]["results"], help="results folder")
    common.add_argument("--data-root", default=None, help="folder containing the dataset folders")
    common.add_argument("--plan", default="v2", choices=["v2", "ablation"],
                        help="v2: the evaluation study; ablation: the component ablation study (IDRiD, Kvasir-SEG)")
    common.add_argument("--worker", default=None, type=str.upper, choices=["A", "B"],
                        help="ablation plan only: the share of one Kaggle account (default: all jobs)")
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
    j.add_argument("--device", default="auto")
    j.add_argument("--num-workers", type=int, default=0)
    stp = sub.add_parser("status", parents=[common], help="list done / pending jobs")
    stp.add_argument("--check-complete", action="store_true", help="exit code 3 unless every job is done")
    an = sub.add_parser("analyse", parents=[common], help="write the paper tables, statistics and figures")
    an.add_argument("--old-results", default=str(Path(__file__).resolve().parent / "Old Results" / "study_v1"),
                    help="results of the preliminary study v1 (adds the v1 -> v2 comparison, T14)")
    sub.add_parser("figures", parents=[common], help="regenerate every experiment's figures from its saved results")
    ar = sub.add_parser("archive", parents=[common], help="zip the results without checkpoints")
    ar.add_argument("out")
    a = p.parse_args(argv)
    return {"job": cmd_job, "run": cmd_run, "status": cmd_status, "analyse": cmd_analyse, "figures": cmd_figures,
            "archive": cmd_archive}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
