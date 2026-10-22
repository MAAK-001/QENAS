# QENAS component ablation study (IDRiD, Kvasir-SEG; seeds 45-47) — one Kaggle notebook cell.
#
# Account 1: WORKER = "A"   (random-search population, max-ZiCo pick on IDRiD)
# Account 2: WORKER = "B"   (parameters-only search, random architecture)
#
# Every session: paste the cell, Save Version -> Save & Run All (Commit). Repeat until the log shows 100 %.
# * code: GitHub branch "ablation"; results: "kaggle-ablation-a" / "kaggle-ablation-b" (one per account);
# * finished experiments are skipped, unfinished ones resume from their last checkpoint;
# * progress (jobs done, %, estimated hours left, current stage) is printed every 10 minutes;
# * results are pushed every 30 minutes and at the end;
# * when BOTH accounts are finished, the cell that finishes last writes all 36 experiments to "kaggle-ablation".
import os, subprocess, threading, time

# ---------------- settings ----------------
WORKER      = "A"                                   # "A" on the first Kaggle account, "B" on the second
REPO_PATH   = "github.com/MAAK-001/QENAS.git"
CODE_BRANCH = "ablation"                            # code of the ablation study
BASE_BRANCH = "kaggle-ablation"                     # start: the 15 existing experiments; end: all 36
RUN_BRANCH  = f"kaggle-ablation-{WORKER.lower()}"   # this account's results
OTHER       = f"kaggle-ablation-{'b' if WORKER == 'A' else 'a'}"
GPUS        = ["0", "1"]                            # Kaggle "GPU T4 x2"
SESSION_H   = 11.0                                  # no job is started that would not finish within this time
SYNC_EVERY  = 30 * 60
WORK        = "/kaggle/working"
CODE        = f"{WORK}/QENAS"
RES         = f"{WORK}/QENAS_results"
# ------------------------------------------
assert WORKER in ("A", "B")
T0 = time.time()
os.environ["GIT_TERMINAL_PROMPT"] = "0"
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN").strip()
url = f"https://x-access-token:{token}@{REPO_PATH}"


def say(text):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + text, flush=True)


def sh(cmd, cwd=None, check=True, timeout=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"") + (e.stderr or b"")).decode(errors="replace").replace(token, "***")
        raise RuntimeError(f"{' '.join(cmd[:4])} did not finish within {timeout} s. Last output:\n{out[-2000:]}")
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])} failed:\n{(r.stdout + r.stderr).replace(token, '***')[-2000:]}")
    return r


def log(text):
    say(text)
    with open(f"{RES}/kaggle_notebook_{WORKER}.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")


def status(worker=True):
    """Progress of this account's jobs (worker=True) or of the whole ablation study."""
    cmd = ["python", "study.py", "status", "--plan", "ablation", "--results", f"{RES}/results"]
    r = sh(cmd + (["--worker", WORKER] if worker else ["--check-complete"]), CODE, check=False)
    lines = r.stdout.strip().splitlines()
    return (lines[-1] if lines else r.stderr[-500:]), r.returncode


# 1) Code and results (this account's branch; on its first session it starts from the base branch)
say("1/5 downloading code")
if not os.path.exists(CODE):
    sh(["git", "clone", "--depth", "1", "--branch", CODE_BRANCH, url, CODE], timeout=600)
say("1/5 downloading results")
if not os.path.exists(f"{RES}/.git"):
    if RUN_BRANCH in sh(["git", "ls-remote", "--heads", url, RUN_BRANCH], timeout=300).stdout:
        sh(["git", "clone", "--depth", "1", "--single-branch", "--branch", RUN_BRANCH, url, RES], timeout=1800)
    else:
        sh(["git", "clone", "--depth", "1", "--single-branch", "--branch", BASE_BRANCH, url, RES], timeout=1800)
        sh(["git", "checkout", "-q", "-b", RUN_BRANCH], RES)
sh(["git", "config", "user.email", "kaggle@run"], RES)
sh(["git", "config", "user.name", f"Kaggle ablation {WORKER}"], RES)
with open(f"{RES}/.git/info/exclude", "w") as f:            # never pushed: images, checkpoint backups, candidates
    f.write("*.png\n*.pdf\n*.prev\nresults/*/*/*/checkpoints/candidates/\n")

# 2) Packages (each step limited to 20 min)
say("2/5 installing packages")
torch_before = sh(["python", "-c", "import torch; print(torch.__version__)"]).stdout.strip()
sh(["pip", "install", "--progress-bar", "off", "--default-timeout", "60", "pymoo>=0.6.2,<0.7", "tifffile",
    "imagecodecs"], timeout=1200)
torch_after = sh(["python", "-c", "import torch, pymoo; print(torch.__version__)"]).stdout.strip()
if torch_after != torch_before:
    raise RuntimeError(f"the package installation changed torch ({torch_before} -> {torch_after}); stopping")
log(f"worker {WORKER} | torch {torch_after} | pymoo installed")

# 3) Push (one amended commit on top of the base branch, so the branch keeps only the latest state)
lock = threading.Lock()
MSG = f"Ablation results, worker {WORKER}"


def push(note=""):
    with lock:
        sh(["git", "add", "-A"], RES, check=False)
        staged = sh(["git", "diff", "--cached", "--name-only"], RES, check=False).stdout.split()
        for p in [p for p in staged if os.path.isfile(f"{RES}/{p}") and os.path.getsize(f"{RES}/{p}") > 95e6]:
            sh(["git", "rm", "-q", "--cached", p], RES, check=False)       # GitHub rejects files > 100 MB
            with open(f"{RES}/.git/info/exclude", "a") as f:
                f.write(p + "\n")
        amend = sh(["git", "log", "-1", "--format=%s"], RES, check=False).stdout.strip() == MSG
        sh(["git", "commit", "-q", "--allow-empty", "-m", MSG] + (["--amend"] if amend else []), RES, check=False)
        try:
            r = sh(["git", "push", "-q", "-f", "origin", f"HEAD:{RUN_BRANCH}"], RES, check=False, timeout=900)
            ok, err = r.returncode == 0, r.stderr
        except RuntimeError as e:
            ok, err = False, str(e)
        log(f"{note} " + ("results pushed" if ok else "PUSH FAILED: " + err.replace(token, "***")[-300:]))
        return ok


say("3/5 push check")
if not push("[start check]"):
    raise RuntimeError("Pushing to GitHub does not work - check the GITHUB_TOKEN secret of this account")
stop = threading.Event()


def periodic():
    while not stop.wait(SYNC_EVERY):
        push("[periodic]")


threading.Thread(target=periodic, daemon=True).start()

# 4) Run this account's share of the ablation plan on both GPUs
data_root = sh(["python", "-c", "from qenas.datasets.discovery import locate_data_root; "
                "print(locate_data_root('/nonexistent', 'IDRID'))"], CODE).stdout.strip()
log(f"progress at start (worker {WORKER}): {status()[0]}")
hours = SESSION_H - (time.time() - T0) / 3600
log(f"4/5 data root {data_root} | budget {hours:.2f} h")
proc = subprocess.run(["python", "study.py", "run", "--plan", "ablation", "--worker", WORKER, "--gpus", *GPUS,
                       "--hours", f"{hours:.2f}", "--results", f"{RES}/results", "--data-root", data_root],
                      cwd=CODE, env=dict(os.environ, PYTHONUNBUFFERED="1"))
log(f"study session finished (exit {proc.returncode})")
mine, _ = status()
log(f"progress at end (worker {WORKER}): {mine}")
stop.set()
push("[session end]")

# 5) If both accounts are finished: all 36 experiments -> the base branch "kaggle-ablation"
if " 0 left" in mine:
    heads = sh(["git", "ls-remote", "--heads", url, OTHER], timeout=300).stdout
    if OTHER in heads:
        sh(["git", "fetch", "-q", "--depth", "1", "origin", OTHER], RES, timeout=1800)
        sh(["git", "checkout", "-q", "FETCH_HEAD", "--", "results"], RES)
        overall, rc = status(worker=False)
        log(f"whole ablation study: {overall}")
        if rc == 0:
            sh(["git", "add", "-A"], RES)
            sh(["git", "commit", "-q", "-m", "Ablation study: all 36 experiments"], RES)
            r = sh(["git", "push", "-q", "-f", "origin", f"HEAD:{BASE_BRANCH}"], RES, check=False, timeout=1800)
            log("5/5 COMPLETE: all 36 experiments are on GitHub branch kaggle-ablation" if r.returncode == 0 else
                "5/5 final push FAILED: " + r.stderr.replace(token, "***")[-300:])
        else:
            sh(["git", "reset", "-q", "--hard", "HEAD"], RES)
            log(f"5/5 Done for worker {WORKER}. Waiting for the other account ({OTHER}); its final session "
                f"will write the complete study to kaggle-ablation.")
    else:
        log(f"5/5 Done for worker {WORKER}. The other account has not started yet ({OTHER} does not exist).")
else:
    log(f"5/5 Session over. Run this cell again to continue (worker {WORKER}: {mine}).")
