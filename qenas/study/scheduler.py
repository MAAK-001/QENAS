"""Runs the study plan: one job per GPU at a time, dependency-aware, resumable and time-budget aware.

* A job is one experiment (dataset, method, seed) run in its own process (``study.py job ...``) with
  ``CUDA_VISIBLE_DEVICES`` set to one GPU, so two GPUs run two jobs in parallel. Finished jobs are skipped,
  unfinished ones resume from their last checkpoint (the experiment directory is fixed).
* A job is started only if its dependencies are finished and its expected duration fits into the remaining time
  budget (Kaggle sessions end after 12 h). Expected durations are the median measured total time of finished jobs
  of the same (method, dataset), else the conservative estimates of ``plan.py``.
* At the hard deadline still-running jobs are terminated; checkpoints are written after every epoch/generation,
  so they continue in the next session.
* A failed job is retried once in the same session; its dependants wait. Every job writes its own log file.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from ..utils.io import load_json, save_json
from .plan import Job, plan

HARD_MARGIN_H = 0.25        # running jobs are terminated this long after the start budget ends
SAFETY = 1.15               # expected duration multiplier


def job_dir(results: Path, job: Job) -> Path:
    return results / job.id


def job_done(results: Path, job: Job) -> bool:
    st = job_dir(results, job) / "state.json"
    if not st.exists():
        return False
    try:
        if load_json(st).get("status") != "completed":
            return False
    except Exception:
        return False
    ext = job_dir(results, job) / "external_CVC"
    return not job.external or (ext / "external_metrics.json").exists() or (ext / "external_unavailable.json").exists()


def measured_minutes(results: Path, jobs: Sequence[Job]) -> Dict[tuple, float]:
    by_key: Dict[tuple, List[float]] = {}
    for j in jobs:
        st = job_dir(results, j) / "state.json"
        if st.exists():
            try:
                s = load_json(st)
                if s.get("status") == "completed" and s.get("total_seconds"):
                    by_key.setdefault((j.method, j.dataset), []).append(s["total_seconds"] / 60)
            except Exception:
                pass
    return {k: statistics.median(v) for k, v in by_key.items()}


def default_command(job: Job, device: str, data_root: Optional[str], num_workers: int, results: Path) -> List[str]:
    cmd = [sys.executable, str(Path(__file__).resolve().parents[2] / "study.py"), "job", "--dataset", job.dataset,
           "--method", job.method, "--seed", str(job.seed), "--device", device,
           "--num-workers", str(num_workers), "--results", str(results)]
    if data_root:
        cmd += ["--data-root", data_root]
    return cmd


def stage_progress(results: Path, job: Job) -> str:
    """'stage k/n (name)' of an unfinished experiment, from its state.json."""
    from ..methods import stages_for
    stages = stages_for(job.method)
    st = job_dir(results, job) / "state.json"
    done = 0
    if st.exists():
        try:
            s = load_json(st).get("stages", {})
            done = sum(1 for k in stages if s.get(k, {}).get("status") == "done")
        except Exception:
            pass
    nxt = stages[done] if done < len(stages) else "finishing"
    return f"stage {min(done + 1, len(stages))}/{len(stages)} ({nxt})"


def progress_line(results: Path, jobs: Sequence[Job], running: Sequence[Job], n_gpus: int,
                  measured: Dict[tuple, float]) -> str:
    done = [j for j in jobs if job_done(results, j)]
    left = [j for j in jobs if j not in done]
    pct = 100.0 * len(done) / max(1, len(jobs))
    est_h = sum(measured.get((j.method, j.dataset), j.estimate_minutes()) for j in left) / 60 / max(1, n_gpus)
    run = "; ".join(f"{j.id} {stage_progress(results, j)}" for j in running) or "none"
    return (f"[progress] {len(done)}/{len(jobs)} jobs done ({pct:.0f} %), {len(left)} left "
            f"(~{est_h:.1f} h on {n_gpus} GPU{'s' if n_gpus > 1 else ''}) | running: {run}")


def run_study(results: Path, gpus: Sequence[Optional[int]], budget_hours: float, data_root: Optional[str] = None,
              num_workers: int = 1, threads_per_job: int = 2, poll_seconds: float = 15.0,
              jobs: Optional[List[Job]] = None,
              command: Optional[Callable[[Job, str], List[str]]] = None, log=print,
              status_name: str = "study_status.json", progress_every: float = 600.0) -> Dict[str, Dict]:
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    logs = results / "study_logs"
    logs.mkdir(exist_ok=True)
    jobs = jobs if jobs is not None else plan()
    by_id = {j.id: j for j in jobs}
    status_file = results / status_name
    last_progress = 0.0
    status: Dict[str, Dict] = load_json(status_file) if status_file.exists() else {}
    attempts: Dict[str, int] = {}
    t0 = time.time()
    running: Dict[Optional[int], tuple] = {}

    def elapsed_h() -> float:
        return (time.time() - t0) / 3600.0

    def save() -> None:
        for j in jobs:
            s = status.setdefault(j.id, {})
            if job_done(results, j):
                s["status"] = "done"
            elif s.get("status") not in ("running", "failed"):
                s["status"] = "pending"
        save_json(status_file, status)

    def expected(j: Job, measured: Dict[tuple, float]) -> float:
        return measured.get((j.method, j.dataset), j.estimate_minutes()) * SAFETY

    log(f"[study] {len(jobs)} jobs | done {sum(job_done(results, j) for j in jobs)} | GPUs {list(gpus)} | "
        f"start budget {budget_hours:.2f} h")
    while True:
        for g, (j, proc, ts, fh) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            minutes = (time.time() - ts) / 60
            ok = rc == 0 and job_done(results, j)
            status[j.id] = {**status.get(j.id, {}), "status": "done" if ok else "failed", "exit_code": rc,
                            "session_minutes": round(minutes, 1), "gpu": g}
            log(f"[study] {'finished' if ok else 'FAILED  '} {j.id} (exit {rc}, {minutes:.0f} min, GPU {g})")
            del running[g]
            save()
            last_progress = 0.0                   # report progress after every finished job
        if elapsed_h() > budget_hours + HARD_MARGIN_H:
            for g, (j, proc, ts, fh) in running.items():
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                fh.close()
                status[j.id] = {**status.get(j.id, {}), "status": "pending", "note": "stopped at session deadline"}
                log(f"[study] deadline: stopped {j.id} (resumes from its last checkpoint next session)")
            running.clear()
            break
        measured = measured_minutes(results, jobs)
        busy = {r[0].id for r in running.values()}
        startable = []
        for j in jobs:
            if j.id in busy or job_done(results, j) or attempts.get(j.id, 0) >= 2:
                continue
            if not all(job_done(results, by_id[d]) for d in j.deps if d in by_id):
                continue
            startable.append(j)
        fits = [j for j in startable if elapsed_h() + expected(j, measured) / 60 <= budget_hours]
        for g in gpus:
            if g in running or not fits:
                continue
            j = fits.pop(0)
            attempts[j.id] = attempts.get(j.id, 0) + 1
            device = "cpu" if g is None else "cuda"
            cmd = command(j, device) if command else default_command(j, device, data_root, num_workers, results)
            env = dict(os.environ, OMP_NUM_THREADS=str(threads_per_job), MKL_NUM_THREADS=str(threads_per_job))
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            fh = open(logs / (j.id.replace("/", "__") + ".log"), "a", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[g] = (j, proc, time.time(), fh)
            status[j.id] = {**status.get(j.id, {}), "status": "running", "gpu": g, "attempt": attempts[j.id]}
            log(f"[study] started  {j.id} on GPU {g} (expected ~{expected(j, measured):.0f} min, "
                f"elapsed {elapsed_h():.2f} h)")
            save()
        if time.time() - last_progress >= progress_every:
            log(progress_line(results, jobs, [r[0] for r in running.values()], len(gpus), measured))
            last_progress = time.time()
        if not running:
            break
        time.sleep(poll_seconds)
    save()
    done = sum(job_done(results, j) for j in jobs)
    log(f"[study] session end after {elapsed_h():.2f} h | {done}/{len(jobs)} jobs done")
    log(progress_line(results, jobs, [], len(gpus), measured_minutes(results, jobs)))
    return status
