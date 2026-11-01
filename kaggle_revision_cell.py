# QENAS journal revision (REVISION.md) - one Kaggle notebook cell, five Kaggle accounts.
#
#   WORKER "1" | "2" | "3": nnU-Net 1000 epochs on IDRiD (seed 45 | 46 | 47, spans two sessions)
#                           + Supervised NSGA-II on BUSI and IDRiD (same seed) on the second GPU
#   WORKER "4":             UNeXt on BUSI, CVC-ClinicDB, IDRiD (seeds 45-47)
#   WORKER "5":             UNeXt on Kvasir-SEG (seeds 45-47) + the external Kvasir-SEG -> CVC-ClinicDB test
#   OPTIONAL = True (workers 4, 5 only): nnU-Net 1000 epochs on BUSI (about three sessions per run)
#
# Every session: paste the cell, Save Version -> Save & Run All (Commit). Repeat until the log shows "0 left".
# * code: GitHub branch CODE_BRANCH at the commit CODE_COMMIT (identical on every account);
# * results: GitHub branch kaggle-revision-<WORKER> (pushed every 30 min and at the end);
# * before any training: code / split / data / software checksums; a mismatch stops the session;
# * every experiment resumes from its last checkpoint; nnU-Net's checkpoint is kept in a private Kaggle dataset of
#   THIS account ("qenas-revision-persist-<WORKER>"), restored at the start of every session and updated every
#   2 hours and at the end. Never move a started nnU-Net run to another account.
#
# One-time setup per account: Kaggle secrets GITHUB_TOKEN (the study token), and for workers 1-3 (and 4, 5 with
# OPTIONAL) KAGGLE_USERNAME (your Kaggle user name) + KAGGLE_API_TOKEN (Settings -> API -> Generate New Token: the
# "KGAT_..." token; the legacy kaggle.json "key" also works, as KAGGLE_KEY), all ticked for this notebook;
# accelerator GPU T4 x2; Internet on; the datasets input with BUSI..., CVC..., IDRID..., Kvasir-SEG (as study v2).
import glob, json, os, shutil, subprocess, threading, time

# ---------------- settings ----------------
WORKER       = "1"                                  # "1" ... "5": one per Kaggle account
OPTIONAL     = False                                # workers 4, 5: also the optional BUSI nnU-Net runs
REPO_PATH    = "github.com/MAAK-001/QENAS.git"
CODE_BRANCH  = "revision"
CODE_COMMIT  = ""                                   # the commit SHA all accounts must use ("" = branch head)
RUN_BRANCH   = f"kaggle-revision-{WORKER}"
GPUS         = ["0", "1"]                           # Kaggle "GPU T4 x2": one job per GPU
SESSION_H    = 11.0                                 # end of the start budget; long nnU-Net runs pause before it
SYNC_EVERY   = 30 * 60                              # push results to GitHub
PERSIST_EVERY = 2 * 3600                            # upload nnU-Net checkpoints to the account's Kaggle dataset
TORCH_REF    = "2.11.0"                             # torch of study v2 (2.11.0+cu128)
TORCH_POLICY = "install"                            # "install" the reference torch if different | "require" | "warn"
WORK         = "/kaggle/working"
CODE         = f"{WORK}/QENAS"
RES          = f"{WORK}/QENAS_results"
PERSIST_SLUG = f"qenas-revision-persist-{WORKER}"
# ------------------------------------------
assert WORKER in ("1", "2", "3", "4", "5")
T0 = time.time()
os.environ["GIT_TERMINAL_PROMPT"] = "0"
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
token = secrets.get_secret("GITHUB_TOKEN").strip()
url = f"https://x-access-token:{token}@{REPO_PATH}"
NEEDS_PERSIST = WORKER in ("1", "2", "3") or OPTIONAL


def secret(label):
    try:
        return secrets.get_secret(label).strip()
    except Exception:
        return ""


if NEEDS_PERSIST:
    # new Kaggle API token ("KGAT_...", KAGGLE_API_TOKEN, Kaggle CLI >= 1.8) or legacy kaggle.json key (KAGGLE_KEY)
    os.environ["KAGGLE_USERNAME"] = secret("KAGGLE_USERNAME")
    for label in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        if secret(label):
            os.environ[label] = secret(label)
    if not os.environ["KAGGLE_USERNAME"] or not (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")):
        raise RuntimeError("Worker 1-3 needs the notebook secrets KAGGLE_USERNAME (your Kaggle user name) and "
                           "KAGGLE_API_TOKEN (the KGAT_... token from Settings -> API), both ticked for this notebook")
    DATASET_ID = f"{os.environ['KAGGLE_USERNAME']}/{PERSIST_SLUG}"


def say(text):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + text, flush=True)


def sh(cmd, cwd=None, check=True, timeout=None, env=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
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


def status():
    cmd = ["python", "study.py", "status", "--plan", "revision", "--worker", WORKER, "--results", f"{RES}/results"]
    r = sh(cmd + (["--optional"] if OPTIONAL else []), CODE, check=False)
    lines = r.stdout.strip().splitlines()
    return lines[-1] if lines else r.stderr[-500:]


# 1) Code (pinned commit) and this account's results branch
say("1/7 downloading code")
if not os.path.exists(CODE):
    sh(["git", "clone", "--branch", CODE_BRANCH, url, CODE], timeout=900)
    if CODE_COMMIT:
        sh(["git", "checkout", "-q", CODE_COMMIT], CODE)
commit = sh(["git", "rev-parse", "HEAD"], CODE).stdout.strip()
say("1/7 downloading results")
if not os.path.exists(f"{RES}/.git"):
    if RUN_BRANCH in sh(["git", "ls-remote", "--heads", url, RUN_BRANCH], timeout=300).stdout:
        sh(["git", "clone", "--depth", "1", "--single-branch", "--branch", RUN_BRANCH, url, RES], timeout=1800)
    else:
        os.makedirs(f"{RES}/results", exist_ok=True)
        sh(["git", "init", "-q"], RES)
        sh(["git", "checkout", "-q", "-b", RUN_BRANCH], RES)
        sh(["git", "remote", "add", "origin", url], RES)
sh(["git", "config", "user.email", "kaggle@run"], RES)
sh(["git", "config", "user.name", f"Kaggle revision {WORKER}"], RES)
with open(f"{RES}/.git/info/exclude", "w") as f:            # never pushed: images, big or transient checkpoints
    f.write("*.png\n*.pdf\n*.prev\n*.tmp\nresults/*/*/*/checkpoints/candidates/\n"
            "results/*/*/*/checkpoints/supervised_search/\nresults/*/*/*/checkpoints/loss_weights/\n"
            "results/*/*/*/nnunet_work/\nresults/*/*/*/nnunet_resume/\n")
log(f"worker {WORKER} | code {CODE_BRANCH} @ {commit}" + ("" if not CODE_COMMIT or commit.startswith(CODE_COMMIT)
                                                         else "  (DIFFERENT FROM CODE_COMMIT)"))

# 2) Packages: the study-v2 versions (pymoo 0.6.x, nnunetv2 2.8.1 without its torch pin, timm for UNeXt)
say("2/7 installing packages")
torch_now = sh(["python", "-c", "import torch; print(torch.__version__)"]).stdout.strip()
if not torch_now.startswith(TORCH_REF):
    if TORCH_POLICY == "require":
        raise RuntimeError(f"torch {torch_now} differs from study v2 ({TORCH_REF}); set TORCH_POLICY='install'")
    if TORCH_POLICY == "install":
        say(f"torch {torch_now} != {TORCH_REF}: installing torch {TORCH_REF} (cu128)")
        sh(["pip", "install", "-q", "--progress-bar", "off", f"torch=={TORCH_REF}", "torchvision==0.26.0",
            "--index-url", "https://download.pytorch.org/whl/cu128"], timeout=2400)
torch_before = sh(["python", "-c", "import torch; print(torch.__version__)"]).stdout.strip()
pip = ["pip", "install", "--progress-bar", "off", "--default-timeout", "60"]
sh(pip + ["pymoo>=0.6.2,<0.7"], timeout=1200)
sh(pip + ["--no-deps", "nnunetv2==2.8.1"], timeout=1200)
sh(pip + ["acvl-utils==0.2.6", "dynamic-network-architectures==0.4.4", "batchgeneratorsv2==0.3.5",
          "batchgenerators==0.25.3", "blosc2>=3.3.2", "SimpleITK>=2.2.1", "nibabel", "imagecodecs", "yacs", "einops",
          "graphviz", "tifffile", "seaborn"], timeout=1200)
if sh(["python", "-c", "import timm"], check=False).returncode != 0:
    sh(pip + ["--no-deps", "timm==1.0.22"], timeout=1200)
torch_after = sh(["python", "-c", "import torch, nnunetv2, pymoo, timm; print(torch.__version__)"]).stdout.strip()
if torch_after != torch_before:
    raise RuntimeError(f"the package installation changed torch ({torch_before} -> {torch_after}); stopping")
log(f"torch {torch_after} | nnunetv2 2.8.1 | pymoo | timm installed")
if NEEDS_PERSIST:                         # the Kaggle CLI (not used by any experiment): >= 1.8 reads KGAT_ tokens
    sh(pip + ["--upgrade", "kaggle>=1.8.0"], timeout=1200)
    if sh(["python", "-c", "import torch; print(torch.__version__)"]).stdout.strip() != torch_after:
        raise RuntimeError("upgrading the Kaggle CLI changed torch; stopping")
    r = sh(["kaggle", "datasets", "list", "--mine"], check=False, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("Kaggle API login failed - check the secrets KAGGLE_USERNAME / KAGGLE_API_TOKEN: "
                           + (r.stdout + r.stderr)[-800:])
    log(f"Kaggle API login OK ({os.environ['KAGGLE_USERNAME']})")

# 3) Identity checks: code, splits, data and software (stops the session on a split or data mismatch)
data_root = sh(["python", "-c", "from qenas.datasets.discovery import locate_data_root; "
                "print(locate_data_root('/nonexistent', 'BUSI'))"], CODE).stdout.strip()
say("3/7 checksums")
os.makedirs(f"{RES}/results", exist_ok=True)
r = sh(["python", "scripts/revision_checksums.py", "--data-root", data_root, "--check",
        "--out", f"{RES}/results/revision_checksums_{WORKER}.json"], CODE, check=False, timeout=3600)
log("checksums: " + (r.stdout.strip().splitlines() or ["?"])[-1])
if r.returncode != 0:
    raise RuntimeError("checksum check failed - the data or split differs from the study:\n" + r.stdout[-3000:])
code_sha = json.load(open(f"{RES}/results/revision_checksums_{WORKER}.json"))["code"]["sha256"]
log(f"code sha256 {code_sha[:16]} (must be the same on all five accounts)")

# 4) GitHub push (one amended commit, so the branch keeps only the latest state)
lock = threading.Lock()
MSG = f"Revision results, worker {WORKER}"


def push(note=""):
    with lock:
        sh(["git", "add", "-A"], RES, check=False)
        staged = sh(["git", "diff", "--cached", "--name-only"], RES, check=False).stdout.split()
        for p in [p for p in staged if os.path.isfile(f"{RES}/{p}") and os.path.getsize(f"{RES}/{p}") > 95e6]:
            sh(["git", "rm", "-q", "--cached", p], RES, check=False)        # GitHub rejects files > 100 MB
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


say("4/7 push check")
if not push("[start check]"):
    raise RuntimeError("Pushing to GitHub does not work - check the GITHUB_TOKEN secret of this account")

# 5) nnU-Net persistent storage: this account's private Kaggle dataset <user>/qenas-revision-persist-<WORKER>
PDL, PUP = f"{WORK}/persist_download", f"{WORK}/persist_upload"
plock = threading.Lock()


def resume_dirs():
    return sorted(glob.glob(f"{RES}/results/*/nnunet_full/seed_*/nnunet_resume"))


def restore_persist():
    """Newest dataset version (or an attached input folder qenas_persist) -> results/.../nnunet_resume/."""
    shutil.rmtree(PDL, ignore_errors=True)
    os.makedirs(PDL)
    r = sh(["kaggle", "datasets", "download", DATASET_ID, "-p", PDL, "--unzip"], check=False, timeout=3600)
    if r.returncode != 0:                                   # older CLI syntax
        r = sh(["kaggle", "datasets", "download", "-d", DATASET_ID, "-p", PDL, "--unzip"], check=False, timeout=3600)
    for z in glob.glob(f"{PDL}/**/*.zip", recursive=True):  # folders uploaded with --dir-mode zip
        shutil.unpack_archive(z, os.path.dirname(z))
    srcs = [PDL] if r.returncode == 0 else []
    srcs += glob.glob("/kaggle/input/*/qenas_persist")
    if not srcs:
        log(f"persist: no stored checkpoints yet ({DATASET_ID} not found) - fine for the first session")
        return
    for src in srcs:
        for man in glob.glob(f"{src}/**/nnunet_resume/manifest.json", recursive=True):
            m = json.load(open(man))
            dst = f"{RES}/results/{m['experiment']}/nnunet_resume"
            cur = f"{dst}/manifest.json"
            if os.path.exists(cur) and json.load(open(cur))["epoch"] >= m["epoch"]:
                continue
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(os.path.dirname(man), dst)
            log(f"persist: restored {m['experiment']} {m['file']} (epoch {m['epoch']}) from {src}")


def upload_persist(note):
    """Consistent snapshot (manifest + the checkpoint it names, sha256 verified) -> new dataset version."""
    if not resume_dirs():
        return
    with plock:
        shutil.rmtree(PUP, ignore_errors=True)
        for d in resume_dirs():
            for _ in range(3):
                m = json.load(open(f"{d}/manifest.json"))
                rel = d[len(f"{RES}/results/"):]
                out = f"{PUP}/qenas_persist/{rel}"
                shutil.rmtree(out, ignore_errors=True)
                os.makedirs(out)
                shutil.copy2(f"{d}/{m['file']}", f"{out}/{m['file']}")
                ok = sh(["sha256sum", f"{out}/{m['file']}"]).stdout.split()[0] == m["sha256"]
                if ok:
                    json.dump(m, open(f"{out}/manifest.json", "w"), indent=1)
                    for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
                        if os.path.exists(f"{d}/{f}"):
                            shutil.copy2(f"{d}/{f}", f"{out}/{f}")
                    break
                time.sleep(20)                              # the job replaced the checkpoint meanwhile
        json.dump({"title": PERSIST_SLUG, "id": DATASET_ID, "licenses": [{"name": "CC0-1.0"}]},
                  open(f"{PUP}/dataset-metadata.json", "w"))
        exists = sh(["kaggle", "datasets", "status", DATASET_ID], check=False, timeout=300).returncode == 0
        cmd = (["kaggle", "datasets", "version", "-p", PUP, "-m", f"{note} {time.strftime('%Y-%m-%d %H:%M')}",
                "--dir-mode", "zip", "-d"] if exists else ["kaggle", "datasets", "create", "-p", PUP, "--dir-mode", "zip"])
        r = sh(cmd, check=False, timeout=3600)
        log(f"{note} persist: " + ("checkpoints uploaded to " + DATASET_ID if r.returncode == 0 else
                                   "UPLOAD FAILED: " + (r.stdout + r.stderr)[-400:]))


if NEEDS_PERSIST:
    say("5/7 restoring nnU-Net checkpoints of this account")
    restore_persist()
stop = threading.Event()


def periodic():
    last = time.time()
    while not stop.wait(SYNC_EVERY):
        push("[periodic]")
        if NEEDS_PERSIST and time.time() - last >= PERSIST_EVERY:
            upload_persist("[periodic]")
            last = time.time()


threading.Thread(target=periodic, daemon=True).start()

# 6) Run this account's jobs on both GPUs (the split stage verifies the saved study-v2 split)
log(f"progress at start (worker {WORKER}): {status()}")
hours = SESSION_H - (time.time() - T0) / 3600
log(f"6/7 data root {data_root} | budget {hours:.2f} h")
proc = subprocess.run(["python", "study.py", "run", "--plan", "revision", "--worker", WORKER, "--gpus", *GPUS,
                       "--hours", f"{hours:.2f}", "--results", f"{RES}/results", "--data-root", data_root]
                      + (["--optional"] if OPTIONAL else []),
                      cwd=CODE, env=dict(os.environ, PYTHONUNBUFFERED="1", QENAS_REFERENCE_SPLITS="1"))
log(f"study session finished (exit {proc.returncode})")

# 7) Final upload of the checkpoints, push, archive
stop.set()
if NEEDS_PERSIST:
    upload_persist("[session end]")
mine = status()
log(f"progress at end (worker {WORKER}): {mine}")
push("[session end]")
subprocess.run(["python", "study.py", "archive", f"{WORK}/QENAS_revision_{WORKER}.zip", "--results", f"{RES}/results"],
               cwd=CODE)
log(f"7/7 {'DONE: all jobs of worker ' + WORKER + ' finished' if ' 0 left' in mine else 'Session over. Run the cell again to continue'}"
    f" ({mine}). Results: GitHub branch {RUN_BRANCH}.")
