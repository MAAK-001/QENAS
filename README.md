# QENAS — Quick Evolutionary NAS for medical image segmentation (version 3.0)

QENAS searches U-shaped segmentation networks built from **five block types**: four manually designed blocks
(Residual, Dense, Inception, ConvNeXt) and a DARTS block. These are the blocks of the Mixed-GGNAS search space,
implemented from the official code.

The search uses **NSGA-II (pymoo)** with **two training-free objectives**. No weight is updated during the search.

| objective | measures | direction |
|---|---|---|
| f1 = number of parameters | model complexity | minimised |
| f2 = 1/(1+ZiCo) | gradient signal-to-noise of the segmentation loss on two training batches (official ZiCo code) | ZiCo maximised |

After the search:

1. **16 candidates** of the final population are trained by **successive halving**: all 16 for 5 epochs, the best
   8 to 10 epochs, the best 4 to 20 epochs (160 epochs in total).
2. The finalist with the strongest **learning capability** is selected: the highest mean validation Dice of its
   last 5 epochs, among the finalists that pass the pathology gates.
3. The loss weights are chosen on the validation set.
4. The selected architecture is trained normally (**120 epochs**) with a validation set; the best-validation
   checkpoint is kept.
5. It is tested **once**. The test reports:
   * the Mixed-GGNAS metrics: **Dice (= DSC)**, **mIoU**, **Params**, **FLOPs**;
   * **HD95** (boundary) and **Sensitivity** (clinical);
   * the inference time.

QENAS is **one pipeline with one protocol** (`main.py`). The **evaluation study** of the paper is run by
`study.py`. Study v1 (QENAS 2.1) is finished; QENAS 3.0 contains the changes it motivated, and study v2 tests them:

* comparison with U-Net, Attention U-Net and **nnU-Net** on identical splits;
* the **loss-coefficient ablation** (α grid vs fixed α = 0.5) and the selection ablations;
* random search;
* 3 new seeds per experiment;
* external validation.

| document | contents |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | the method |
| [STUDY.md](STUDY.md) | the experimental protocol of study v2 (and what v1 already answered) |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | every decision, with reasons and evidence (incl. the v1 results behind every 3.0 change) |
| [PAPER_POSITIONING.md](PAPER_POSITIONING.md) | related work and what QENAS adds |
| [DATASETS.md](DATASETS.md) | dataset structure and splits |
| `Old Results/` | everything of the preliminary study v1 (QENAS 2.1): results, report `QENAS_Study_v1.pdf` (local, not pushed) |

---

## 1. Layout

```
GGNAS/
├── Datasets/          BUSI (Breast Ultrasound Image)/  CVC-ClinicD (Polyp)/  IDRID/  Kvasir-SEG/
└── QENAS/
    ├── main.py        QENAS: one dataset, one seed, the whole pipeline
    ├── study.py       the evaluation study: run / status / analyse / figures / archive
    ├── qenas/
    │   ├── config.py      the QENAS protocol (one fixed configuration) + per-dataset data facts
    │   ├── methods.py     QENAS and the study conditions (baselines, ablations, random search)
    │   ├── runner.py      runs one experiment from start to end (timer included)
    │   ├── experiment.py  experiment folders results/<DATASET>/<method>/seed_<s>/, stage state, safe resume
    │   ├── pipeline.py    the stages (split → … → report), shared by all conditions
    │   ├── datasets/      discovery (per dataset), leakage-free splits, preprocessing, augmentation, loaders
    │   ├── models/        Mixed-GGNAS blocks, DARTS cells, QENASNet, U-Net / Attention U-Net, params & FLOPs
    │   ├── search/        objectives (params, ZiCo), cache, NSGA-II, random search, candidate subset
    │   ├── training/      loss, metrics (Dice, IoU, mIoU, sensitivity, HD95), trainer, early stopping,
    │   │                  successive halving and selection
    │   ├── study/         study plan, GPU scheduler, nnU-Net baseline, external validation, analysis
    │   ├── visualization/ figures
    │   └── utils/         atomic I/O, checkpoints, seeding, environment, logging/ETA
    ├── scripts/inspect_datasets.py   dataset inspection (no model, no training)
    ├── tests/         unit tests + end-to-end tests of QENAS and of the whole study (pytest)
    ├── results/       the final results: study v2 of QENAS 3.0 (not on main; GitHub branch kaggle-study-v2)
    ├── Old Results/   the preliminary study v1 (QENAS 2.1): results and report (not on main)
    └── cache/         zero-cost evaluation cache, preprocessed-image cache (created by runs)
```

## 2. Installation

Python ≥ 3.10 (tested with 3.13.7). Install PyTorch for your platform/CUDA from <https://pytorch.org>, then:

```bash
pip install -r requirements.txt
```

The nnU-Net baseline of the study additionally needs nnunetv2. Every nnunetv2 2.x release excludes torch 2.9.*
(DESIGN_DECISIONS D46), so it is installed without its dependencies and the dependencies separately:

```bash
pip install --no-deps nnunetv2==2.8.1
```

```bash
pip install acvl-utils==0.2.6 dynamic-network-architectures==0.4.4 batchgeneratorsv2==0.3.5 batchgenerators==0.25.3 "blosc2>=3.3.2" "SimpleITK>=2.2.1" nibabel imagecodecs yacs einops graphviz tifffile seaborn
```

The datasets must be in `GGNAS/Datasets/` (default) or another folder given with `--data-root`. The folder names
only need to *start with* `BUSI`, `CVC`, `IDRID` and `Kvasir`.

## 3. Verify the installation

The unit tests and the end-to-end tests run all of QENAS and every study condition on small synthetic datasets.
They take about 10 minutes on a CPU (the nnU-Net test runs only where nnunetv2 is installed, about 5 minutes
more):

```bash
python -m pytest -q tests
```

Inspect the four datasets (structure, pairing, integrity, resulting splits; no model is built):

```bash
python scripts/inspect_datasets.py
```

## 4. Running QENAS

Each command runs the complete QENAS pipeline for one dataset, from the split to the final test and report:

```bash
python main.py --dataset BUSI
```

```bash
python main.py --dataset CVC
```

```bash
python main.py --dataset IDRID
```

```bash
python main.py --dataset KVASIR
```

Results go to `results/<DATASET>/qenas/seed_<seed>/`. The research summary (`summary.txt`) is printed at the
end; its last line is the time QENAS took on the dataset from start to end.

| Stage | What happens | Main outputs |
|---|---|---|
| `split` | dataset discovery, integrity checks, fixed leakage-free split | `dataset_report.json`, `split.json` |
| `search` | training-free NSGA-II (params, ZiCo), population 20, generation 0 + 10 generations | `search_results.*`, `search_population_history.*`, `nsga2_generations.*` (incl. hypervolume), `pareto_front.*`, `search_summary.json` |
| `pareto` | 16 candidates: the final front, filled up in NSGA-II's survival order | `pareto_candidates.*` |
| `candidates` | successive halving: 16 × 5 → 8 × 10 → 4 × 20 epochs | `successive_halving.json`, `candidate_results.json`, `candidate_training.csv`, `candidate_histories/` |
| `selection` | learning-capability selection among the 4 finalists | `selection.json`, `candidate_comparison.*` |
| `loss_weights` | α/β grid on validation (α = 0.5 run reused) | `loss_weights.*` |
| `final` | supervised training with validation (120 epochs), best checkpoint | `final_training.csv`, `final_validation.json` |
| `test` | **single** test evaluation | `final_metrics.json`, `final_result.*`, `test_per_image.csv`, `test_predictions.npz` |
| `report` | model summary, architecture figure, tables, reproducibility record, research summary | `model_summary.*`, `reproducibility.json`, `summary.txt` |

### Run settings

The method is fixed in `qenas/config.py`. The command line only takes run settings:

| option | meaning |
|---|---|
| `--dataset BUSI\|CVC\|IDRID\|KVASIR` | dataset (required) |
| `--data-root PATH` | folder containing the dataset folders (default `../Datasets`; Kaggle's `/kaggle/input` is searched automatically) |
| `--device auto\|cpu\|cuda\|cuda:N` | device (default `auto`) |
| `--num-workers N` | data-loader worker processes (default 0) |
| `--seed N` | run seed (default 42); the split is the same for every seed |
| `--stop-after STAGE` | stop cleanly after a stage; the same command without it continues from there |

For example, QENAS on BUSI with seed 45 on a GPU:

```bash
python main.py --dataset BUSI --seed 45 --device cuda
```

## 5. The evaluation study (`study.py`)

STUDY.md describes the research questions, the conditions and the statistics. The plan of study v2 has 84
experiments, each with the seeds 45, 46 and 47:

* QENAS, U-Net, Attention U-Net and nnU-Net on 4 datasets;
* the loss-coefficient ablation (`qenas_fixed_alpha`) on 4 datasets;
* the selection ablations (`qenas_random_pick`, `qenas_zico_pick`) on BUSI and Kvasir-SEG;
* random search on 4 datasets;
* plus external validation inside the Kvasir-SEG jobs of QENAS, U-Net, Attention U-Net and nnU-Net.

Run or resume the whole plan (one job per listed GPU):

```bash
python study.py run --gpus 0
```

Show which experiments are done:

```bash
python study.py status
```

Write all paper tables, statistical tests and figures to `results/analysis/` (`--data-root` enables the
qualitative figures; the comparison with study v1 is read from `Old Results/study_v1`):

```bash
python study.py analyse --data-root ../Datasets
```

All paper tables (one per metric, with published results of other methods), statistics, ablations, cost analysis
and paper figures, written to `results/paper/` (needs the datasets for the extra metrics):

```bash
python scripts/paper_analysis.py
```

Publication figures (600 dpi, Times New Roman, one font size, no titles; the file name is the title), written to
`results/figures/` (comparisons, QENAS stages, qualitative masks, review) and `results/figures/3 Architecture/`:

```bash
python scripts/paper_figures.py
python scripts/paper_diagrams.py
python scripts/paper_architecture.py
```

Publication tables (one Word file, `results/tables/QENAS Tables.docx`; the renderer needs the npm package `docx`):

```bash
python scripts/paper_tables.py
node scripts/render_tables_docx.js
```

Regenerate every experiment's figures from its saved results (PNG files are not stored on GitHub):

```bash
python study.py figures
```

Zip all results without model checkpoints:

```bash
python study.py archive QENAS_results.zip
```

Rebuild the report of study v1 (needs `pip install reportlab`):

```bash
python "Old Results/build_study_v1_pdf.py" . "Old Results/QENAS_Study_v1.pdf" cache/report_figures
```

On Kaggle, use the cell of §6, which runs the plan on both T4 GPUs and saves everything to GitHub.

## 6. Running study v2 on Kaggle

### One-time setup

1. The code must be on GitHub `main` (done for QENAS 3.0).
2. GitHub token (fine-grained, only the QENAS repository, **Contents: Read and write**) stored as the Kaggle
   secret `GITHUB_TOKEN`, switched on for the notebook — the same secret as for study v1.
3. Notebook settings: **Accelerator GPU T4 x2**, **Internet on**, and the dataset input containing `BUSI…`,
   `CVC…`, `IDRID…` and `Kvasir-SEG` — the same notebook as for study v1 works.

### Every session

Paste the cell (it replaces the v1 cell), then **Save Version → Save & Run All (Commit)**. Repeat with the same
cell until `[study] … 84/84 jobs done` appears in the log (study v2 is complete: 84/84).

* Each session takes the latest code from `main`.
* It continues the results of the branch `kaggle-study-v2`: finished experiments are skipped and unfinished ones
  resume from their last checkpoint. The v1 results on `kaggle-study` are never touched.
* It starts no job that cannot finish before the 12-hour limit.
* Results are pushed every 30 minutes and at the end of the session.

```python
import os, subprocess, threading, time

# ---------------- settings ----------------
REPO_PATH   = "github.com/MAAK-001/QENAS.git"
CODE_BRANCH = "main"               # code: always the latest main
RUN_BRANCH  = "kaggle-study-v2"    # results only; main is never touched
GPUS        = ["0", "1"]           # Kaggle "GPU T4 x2"
SESSION_H   = 11.0                 # no job is started that would not finish within this time of the session
SYNC_EVERY  = 30 * 60              # push results every 30 minutes
WORK        = "/kaggle/working"
CODE        = f"{WORK}/QENAS"
RES         = f"{WORK}/QENAS_results"   # git repository of the results branch
# ------------------------------------------
T0 = time.time()
os.environ["GIT_TERMINAL_PROMPT"] = "0"
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN").strip()
url = f"https://x-access-token:{token}@{REPO_PATH}"

def say(text):                                   # visible at once in the Kaggle log
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

def log(text):                                   # also written to the results branch
    say(text)
    with open(f"{RES}/kaggle_notebook.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")

# 1) Code and results. A NEW results branch is created only if it really does not exist on GitHub.
say("1/5 downloading code")
if not os.path.exists(CODE):
    sh(["git", "clone", "--depth", "1", "--branch", CODE_BRANCH, url, CODE], timeout=600)
say("1/5 downloading results")
if not os.path.exists(f"{RES}/.git"):
    heads = sh(["git", "ls-remote", "--heads", url, RUN_BRANCH], timeout=300).stdout
    if RUN_BRANCH in heads:
        sh(["git", "clone", "--depth", "1", "--single-branch", "--branch", RUN_BRANCH, url, RES], timeout=1800)
    else:
        os.makedirs(RES, exist_ok=True)
        sh(["git", "init", "-q"], RES)
        sh(["git", "checkout", "-q", "-b", RUN_BRANCH], RES)
        sh(["git", "remote", "add", "origin", url], RES)
sh(["git", "config", "user.email", "kaggle@run"], RES)
sh(["git", "config", "user.name", "Kaggle run"], RES)
with open(f"{RES}/.git/info/exclude", "w") as f:   # never pushed: images, checkpoint backups, big weights, nnU-Net work
    f.write("*.png\n*.pdf\n*.prev\nresults/*/unet/*/checkpoints/\nresults/*/attention_unet/*/checkpoints/\n"
            "results/*/*/*/checkpoints/candidates/\nresults/*/nnunet/*/nnunet_work/\n")

# 2) Packages (each step limited to 20 min; on failure the pip output is shown). nnunetv2 excludes torch 2.9.*,
#    which may be preinstalled; its 2D networks do not need that exclusion, so it is installed without dependencies.
say("2/5 installing packages")
torch_before = sh(["python", "-c", "import torch; print(torch.__version__)"]).stdout.strip()
pip = ["pip", "install", "--progress-bar", "off", "--default-timeout", "60"]
sh(pip + ["pymoo>=0.6.2,<0.7"], timeout=1200)
sh(pip + ["--no-deps", "nnunetv2==2.8.1"], timeout=1200)
sh(pip + ["acvl-utils==0.2.6", "dynamic-network-architectures==0.4.4", "batchgeneratorsv2==0.3.5",
          "batchgenerators==0.25.3", "blosc2>=3.3.2", "SimpleITK>=2.2.1", "nibabel", "imagecodecs", "yacs", "einops",
          "graphviz", "tifffile", "seaborn"], timeout=1200)
torch_after = sh(["python", "-c", "import torch, nnunetv2, pymoo; print(torch.__version__)"]).stdout.strip()
if torch_after != torch_before:
    raise RuntimeError(f"the package installation changed torch ({torch_before} -> {torch_after}); stopping")
log(f"torch {torch_after} | nnunetv2 2.8.1 | pymoo installed")

# 3) Push the results (one amended commit, so the branch keeps only the latest state)
lock = threading.Lock()
def push(note=""):
    with lock:
        sh(["git", "add", "-A"], RES, check=False)
        staged = sh(["git", "diff", "--cached", "--name-only"], RES, check=False).stdout.split()
        big = [p for p in staged if os.path.isfile(f"{RES}/{p}") and os.path.getsize(f"{RES}/{p}") > 95e6]
        for p in big:                              # GitHub rejects files > 100 MB
            sh(["git", "rm", "-q", "--cached", p], RES, check=False)
            with open(f"{RES}/.git/info/exclude", "a") as f:
                f.write(p + "\n")
        first = sh(["git", "log", "-1", "--format=%s"], RES, check=False).stdout.strip() != "Study results"
        sh(["git", "commit", "-q", "--allow-empty", "-m", "Study results"] + ([] if first else ["--amend"]), RES,
           check=False)
        try:
            r = sh(["git", "push", "-q", "-f", "origin", RUN_BRANCH], RES, check=False, timeout=900)
            ok, err = r.returncode == 0, r.stderr
        except RuntimeError as e:
            ok, err = False, str(e)
        log(f"{note} " + ("results pushed" if ok else "PUSH FAILED: " + err.replace(token, "***")[-300:]))
        return ok

say("3/5 push check")
if not push("[start check]"):
    raise RuntimeError("Pushing to GitHub does not work - check the GITHUB_TOKEN secret")
stop = threading.Event()
def periodic():
    while not stop.wait(SYNC_EVERY):
        push("[periodic]")
threading.Thread(target=periodic, daemon=True).start()

# 4) Run the study plan on both GPUs
data_root = sh(["python", "-c", "from qenas.datasets.discovery import locate_data_root; "
                "print(locate_data_root('/nonexistent', 'BUSI'))"], CODE).stdout.strip()
hours = SESSION_H - (time.time() - T0) / 3600
log(f"4/5 data root {data_root} | study budget {hours:.2f} h")
proc = subprocess.run(["python", "study.py", "run", "--gpus", *GPUS, "--hours", f"{hours:.2f}",
                       "--results", f"{RES}/results", "--data-root", data_root], cwd=CODE)
log(f"study session finished (exit {proc.returncode})")

# 5) Paper tables + figures, final push, and a zip of everything (with figures, without weights) in Output
subprocess.run(["python", "study.py", "analyse", "--results", f"{RES}/results", "--data-root", data_root], cwd=CODE)
subprocess.run(["python", "study.py", "status", "--results", f"{RES}/results"], cwd=CODE)
stop.set()
push("[session end]")
subprocess.run(["python", "study.py", "archive", f"{WORK}/QENAS_study_v2_results.zip", "--results", f"{RES}/results"],
               cwd=CODE)
log("5/5 Done. Results: GitHub branch kaggle-study-v2, and Output: QENAS_study_v2_results.zip")
```

### Component ablation study (IDRiD, Kvasir-SEG) on two Kaggle accounts

Six conditions × 2 datasets × seeds 45–47 = 36 experiments (`study.py status --plan ablation`): full QENAS,
random-search population (no NSGA-II), parameters only (no ZiCo), random architecture (no search, no ZiCo),
max-ZiCo pick (no successive halving), fixed α = 0.5. The 15 experiments of study v2 are on the results branch
`kaggle-ablation`; the 21 new ones are split between account A and account B.

Setup on **both** accounts: the Kaggle secret `GITHUB_TOKEN` (same token), GPU T4 x2, Internet on, and the
datasets input with `IDRID…` and `Kvasir-SEG`. Paste `kaggle_ablation_cell.py` into a notebook cell with
`WORKER = "A"` on the first account and `WORKER = "B"` on the second, then Save Version → Save & Run All
(Commit). Repeat a session until the log shows `0 left`. Each account pushes to its own branch
(`kaggle-ablation-a` / `-b`); the account that finishes last writes all 36 experiments to `kaggle-ablation`.

```bash
git fetch origin kaggle-ablation
```

```bash
git archive origin/kaggle-ablation results | tar -x
```

### Getting the results onto your PC

Fetch the results branch of study v2:

```bash
git fetch origin kaggle-study-v2
```

Extract its `results/` folder into `results/` (nothing is staged on `main`):

```bash
git archive origin/kaggle-study-v2 results | tar -x
```

Regenerate the figures, and the paper tables including the comparison with study v1:

```bash
python study.py figures
```

```bash
python study.py analyse --data-root ../Datasets
```

The Kaggle version's Output also contains `QENAS_study_v2_results.zip` with all figures.

**What is not on GitHub:**

* PNG figures (regenerate them with the commands above);
* the U-Net / Attention U-Net / nnU-Net weights (larger than GitHub's file limit; their predictions and metrics
  are saved).

## 7. Interruptions and resume

Nothing long runs without a checkpoint:

* the search is checkpointed after every generation;
* every training run is checkpointed after every epoch (optimiser, scheduler, scaler and RNG states included);
* successive halving saves every candidate's history after each run;
* nnU-Net writes a checkpoint every 5 epochs.

Re-run the same command (or the Kaggle cell), and the experiment continues from the last checkpoint.

* A finished experiment is never modified: re-running reports it.
* An experiment whose configuration changed (a different code version that affects results) is refused with the
  list of differing keys. Delete its folder to run it again.
* The device and the data root are not part of the configuration, so an experiment can move between machines.
* Checkpoints a finished stage no longer needs are deleted; a finished experiment keeps only its final weights.
* On Kaggle the candidate checkpoints and nnU-Net's working folder are not pushed. If a session ends inside such
  a stage, the next session repeats that stage (the scheduler normally prevents this by not starting jobs that
  cannot finish).

## 8. Compute time

* **QENAS on a Kaggle T4** (study v1, QENAS 2.1, measured): 33 min (IDRiD) to 106 min (Kvasir-SEG) per run from
  start to end; the search itself 5–10 min. QENAS 3.0 has the same candidate budget and 20 % longer final training.
* **Study v2:** about 90 GPU-hours (nnU-Net estimated conservatively), i.e. about 45 hours with two GPUs, which is
  4–5 Kaggle sessions. Measured durations replace the estimates as jobs finish.
* **On a CPU:** the search alone takes hours and the supervised stages days.

**OneDrive:** this project lives in a OneDrive folder. Sync clients briefly lock new files. QENAS retries locked
writes automatically, but pausing sync during long local runs is recommended.

## 9. Outputs of one experiment

```
results/BUSI/qenas/seed_45/
├── config.json  environment.json  state.json  reproducibility.json  summary.txt
├── dataset_report.json  split.json
├── search_results.*  search_population_history.*  search_summary.json  nsga2_generations.*
├── pareto_front.*  pareto_candidates.*
├── successive_halving.json  candidate_training.csv  candidate_results.json  candidate_histories/
├── selection.json  candidate_comparison.*
├── loss_weights.*  loss_weight_histories/
├── final_training.csv  final_validation.json
├── final_metrics.json  final_result.*  test_per_image.csv  test_predictions.npz
├── model_summary.*
├── checkpoints/final/best.pt
├── figures/   (PNG, 300 dpi)
└── logs/run.log
```

## 10. Scientific safeguards (summary)

* **Search:** training-free; no weight is ever updated.
  * Two objectives (parameters, ZiCo). No TOPSIS, no weighted sum, no proxy combination.
  * ZiCo is computed with the unmodified official code on two training batches. The search never uses
    validation or test images.
* **Isolation:** the test split is loaded only by the final test stage, exactly once. Nothing is ever selected
  on test data.
* **Splits:** leakage-free and identical for all methods, seeds and both studies:
  * CVC is split by colonoscopy sequence;
  * BUSI is stratified, with a conflicting duplicate removed;
  * IDRiD and Kvasir-SEG keep their official test splits.
* **Fair comparisons:** U-Net and Attention U-Net are trained with exactly QENAS' protocol; nnU-Net runs its own
  pipeline on the identical images and split. Ablations differ from QENAS in one component only. Paired tests are
  run on the same test images.
* **Consistency:** the same network class and the same initialisation are used for ZiCo, candidate training and
  final training.
* **Honesty:** results are reported as they are. Every design choice is documented in DESIGN_DECISIONS.md.

## 11. Known limitations

* BUSI, IDRiD and Kvasir-SEG have no patient identifiers, so their splits are image-level (BUSI exact
  duplicates are handled).
* The dataset-specific DARTS genotypes were derived by the Mixed-GGNAS authors on their own (unknown) split
  (DESIGN_DECISIONS D5). Kvasir-SEG uses the CVC-ClinicDB genotype.
* ZiCo is a zero-cost *proxy*; inside the search elite it did not predict trained quality in study v1, which is
  why QENAS selects with supervised successive halving.
* nnU-Net is trained for 100 instead of 1000 epochs (compute; DESIGN_DECISIONS D47).
* Mixed-GGNAS cannot be re-run on these splits (no released architectures/weights; about 26 h of search per
  dataset). Its published numbers come from its own splits.
* Bit-exact reproducibility is verified on CPU. Some CUDA kernels may differ in the last bits; nnU-Net's data
  loading is not seeded.
