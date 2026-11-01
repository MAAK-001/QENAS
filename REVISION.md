# QENAS — journal revision: three additional experiments

The revision adds exactly three comparisons. QENAS itself, its results (study v2) and every existing baseline result
are re-used unchanged; the new conditions run through the same pipeline (`study.py`, `qenas/pipeline.py`), on the
same saved splits (split seed 42), with seeds 45, 46, 47, on Kaggle Tesla T4 GPUs, with the software of study v2.

| # | condition (`qenas/methods.py`) | what differs from the existing study | datasets | runs |
|---|---|---|---|---:|
| 1 | `supervised_nsga2` — **Supervised NSGA-II** | QENAS with f2 = 1 − S1 instead of 1/(1+ZiCo); S1 = mean validation Dice over epochs 1–5 of a 5-epoch training under the candidate protocol (= QENAS' rung-1 score) | BUSI, IDRiD | 6 |
| 2 | `nnunet_full` — **nnU-Net (1000 epochs)** | the default `nnUNetTrainer` instead of `nnUNetTrainer_100epochs`; resumes across sessions | IDRiD (BUSI optional) | 3 (+3) |
| 3 | `unext` — **UNeXt** | the official UNeXt network (1.47 M parameters) in place of U-Net, same protocol | all four (+ external Kvasir-SEG → CVC-ClinicDB) | 12 |

Design decisions: DESIGN_DECISIONS.md D51–D56. Fairness tests: `tests/test_revision.py`, `tests/test_nnunet_full.py`.

---

## 1. What exactly runs

### 1.1 Supervised NSGA-II (`qenas/search/supervised.py`)

* **Identical to QENAS** (tested): search space and encoding, pymoo NSGA-II with uniform crossover (p_c = 0.9) and
  random-resetting mutation (p_m = 1/8), population 20, 10 generations (≤ 220 distinct evaluations), duplicate
  elimination, seed, split, evaluation cache, per-generation checkpoints, hypervolume per generation; then stages 2–4
  unchanged: 16 candidates (front + fill-ups, farthest-point rule), successive halving 16×5 → 8×10 → 4×20 epochs,
  gates G1–G5, α grid {0.2, 0.4, 0.5, 0.6, 0.8}, 120-epoch final training, best-validation checkpoint, one test.
* **Only f2 differs:** `f2 = 1 − S1`. S1 is computed by the successive-halving code itself: `Pipeline.init_model`
  (θ₀(a, s)), `Trainer` with `candidate_training` (AdamW 1e-3, wd 5e-5, constant LR, batch 8 / IDRiD 4, AMP),
  α = 0.5, `fit(until=5)`, `criterion(history, 5)`. ZiCo is still computed (cached, training-free) and recorded, so
  every output has the QENAS format; it is not optimised.
* **Rung 1 is retrained, not re-used** (D53): bit-identity of a re-used checkpoint cannot be guaranteed on CUDA
  (some kernels run with `warn_only` determinism), and re-use would save ≤ 17 min per run. On the CPU the retrained
  rung-1 score equals S1 exactly (test); on the GPU, table R5 reports how many of the 16 candidates match bit-for-bit
  and the largest difference.
* **Outputs:** everything QENAS writes (`search_results.*` = the archive with params, ZiCo, S1, f2, training epochs and
  seconds per architecture; `nsga2_generations.*` with hypervolume and best S1 per generation; `pareto_*`;
  `successive_halving.json`; selection; α grid; final training; `final_metrics.json` with parameters and FLOPs;
  `test_per_image.csv`; `test_predictions.npz`), plus `supervised_fitness/<chromosome>.json` (one record per trained
  architecture), `supervised_search_histories/*.csv`, and in `search_summary.json` the training spent in the search
  (`fitness_trainings`, `fitness_epochs_total`, `fitness_seconds_total`). Stage times are in `state.json` exactly as
  for QENAS (search, candidates = successive halving, loss_weights = α grid, final).
* **Resume:** every finished 5-epoch training is stored immediately; an interrupted training resumes from its last
  epoch; the search resumes from its last generation and re-uses every stored record (tested: an interrupted and
  resumed search ends with exactly the archive of an uninterrupted one).

### 1.2 nnU-Net, 1000 epochs (`qenas/study/nnunet_full.py`)

* nnunetv2 2.8.1, default `nnUNetTrainer` (1000 epochs × 250 iterations), `2d`, default plans, fold 0 = our
  train/validation split (`splits_final.json`), the same export of our preprocessed images as the existing nnU-Net
  runs, final checkpoint, test without TTA (main) and with nnU-Net's mirroring (secondary; per-image file
  `test_per_image_mirroring_tta.csv`), all scored by our metric code.
* **Safe resume (nnU-Net's own `--c`):** the job stops by itself at the end of the last epoch that fits into the
  session (`QENAS_STOP_AT`, set by the scheduler; also on SIGTERM): nnU-Net writes `checkpoint_latest.pth`, the job
  copies it atomically with a manifest (epoch, SHA-256 of checkpoint, plans and split) to `nnunet_resume/` and exits
  with code 75 (*paused*). The Kaggle cell uploads that folder to a **private Kaggle dataset of the same account**
  every 2 h and at the session end, and restores it at the start of the next session. The new session re-creates the
  data export and preprocessing (deterministic; the plans' SHA-256 must match), puts the checkpoint back, and nnU-Net
  continues from its epoch. `state.json` (pushed to GitHub) records the last persisted epoch; a missing or older
  checkpoint stops the job instead of silently restarting (tested end-to-end with the real nnunetv2).
* GPU time = the sum over all sessions (`stages.nnunet_train.seconds`, `sessions`).

### 1.3 UNeXt (`qenas/models/unext.py`)

* The official `archs.py` of github.com/jeya-maria-jose/UNeXt-pytorch (MIT), classes copied unchanged (only unused
  imports dropped, incl. the unused `mmcv` import); `UNext(num_classes=1)` with the official defaults.
* Verified: **1,471,921 parameters (paper 1.47 M)** and **0.573 GFLOPs per 256 × 256 image with our thop counter
  (paper 0.57)**. Input sides must be divisible by 32: 256 × 256 and 320 × 512 work (checked in `validate_config`).
* Trained through exactly the U-Net code path (tested: same stages, identical `Trainer` calls — protocol, α, seed,
  early stopping, checkpoints — only the network differs): α grid (20-epoch runs), 120-epoch final training,
  best-validation checkpoint, batch 8 (IDRiD 4), same augmentation and loss, no TTA, one test; Kvasir-SEG jobs add the
  external test on all 612 CVC-ClinicDB frames.

## 2. Statistics and tables (`scripts/revision_tables.py`)

One script merges the results of the five accounts and regenerates the affected tables in `results/revision/`:

| file | table |
|---|---|
| `R1_accuracy_<metric>` | accuracy (Dice, mIoU, HD95, sensitivity, IoU): QENAS, nnU-Net 100 / 1000 epochs, Attention U-Net, U-Net, UNeXt, Supervised NSGA-II; 4 test sets + external |
| `R2_paired_dice` | QENAS vs nnU-Net, Attention U-Net, U-Net, **UNeXt** — per-image Wilcoxon, **Holm over these four** per test set, bootstrap CI, rank-biserial r |
| `R2_friedman` | Friedman test + mean ranks of the **five** networks, Nemenyi CD (q₀.₀₅ = 2.728 for k = 5) |
| `R3_qenas_vs_nnunet_1000` | paired statistics + **TOST** of QENAS vs nnU-Net 1000 epochs (no TTA = main; mirroring = secondary; 100 epochs for reference), **pre-specified margin ±2 Dice points** (`--margin`) |
| `R4_size_and_gpu_time` | parameters, FLOPs (thop, per training batch and per image), inference ms, GPU hours per run and in total |
| `R5_qenas_vs_supervised_nsga2` | **new table "QENAS vs Supervised NSGA-II"**: Dice and HD95 (paired per-image, Holm over the two), parameters, GPU time per stage and in total, training epochs spent in the search, rung-1 identity check, selected chromosomes |

The paired statistics are those of `scripts/paper_analysis.py` (`paired`, `holm`), imported, not re-implemented.
TOST did not exist in the repository; it is added in `revision_tables.tost`: two one-sided Wilcoxon signed-rank
tests on the per-image differences (H0: mean d ≤ −2 pp, H0: mean d ≥ +2 pp; equivalence if both are rejected at 0.05),
with the one-sided t-test version and the 90 % bootstrap CI as checks (D55).

## 3. Running it on Kaggle (5 accounts)

### One-time setup per account

1. The code on GitHub branch `revision` (this commit); write its SHA into `CODE_COMMIT` of the cell — the same SHA on
   all five accounts.
2. Kaggle secrets: `GITHUB_TOKEN` (the study token). Workers 1–3 (and 4–5 if `OPTIONAL = True`) also `KAGGLE_USERNAME`
   (your Kaggle user name) and `KAGGLE_API_TOKEN` (Settings → API → *Generate New Token*: the `KGAT_…` token; a
   legacy `kaggle.json` key also works as `KAGGLE_KEY`), all **ticked for the notebook** in *Add-ons → Secrets*, for
   the private dataset that keeps the nnU-Net checkpoints. The cell checks the login before any training.
3. Notebook: **GPU T4 x2**, Internet on, the dataset input with `BUSI…`, `CVC…`, `IDRID…`, `Kvasir-SEG` (the same input
   as study v2). Environment: *pin to original environment* if offered; otherwise the cell installs torch 2.11.0
   (cu128) when the image has another version (`TORCH_POLICY`).
4. Paste `kaggle_revision_cell.py` with `WORKER = "1"` … `"5"` → **Save Version → Save & Run All (Commit)**. Repeat
   (same account, same notebook) until the log shows `0 left`.

### Every session (automatic)

clone code at `CODE_COMMIT` → install packages (study-v2 versions) → **checksums** (code, split, data, software;
mismatch of split or data stops the session before any training) → restore nnU-Net checkpoints of this account →
run the account's jobs on both GPUs (`study.py run --plan revision --worker W`; the split stage verifies the saved
study-v2 split, `QENAS_REFERENCE_SPLITS=1`) → push results every 30 min to `kaggle-revision-W` → upload checkpoints
every 2 h and at the end → final push.

**Resume options everywhere:** every job resumes from its last checkpoint (search: per generation and per stored
fitness record; every training: per epoch; nnU-Net: per epoch via `checkpoint_latest.pth`). Locally:
`python study.py run --plan revision --worker W --gpus 0` (re-run = continue); one job:
`python study.py job --dataset IDRID --method nnunet_full --seed 45 --results results` (re-run = continue).

## 4. Time estimate (Tesla T4, from the measured study-v2 runs)

Reference timings (study v2, two jobs per session as now): QENAS 0.47 h (IDRiD) / 1.31 h (BUSI) / 1.47 h (CVC) /
2.49 h (Kvasir); candidate epoch (training + validation incl. HD95) 3.2 s IDRiD, 12.7 s BUSI; nnU-Net 100 epochs
68.5 min (IDRiD, 41 s/epoch) / 180 min (BUSI); Attention U-Net full protocol 13–61 min.

| job | derivation | GPU h per run | runs | GPU h |
|---|---|---:|---:|---:|
| Supervised NSGA-II, BUSI | search 220 × 5 ep × 12.7 s = 3.9 h + ZiCo 0.1 h; SH 0.58 + α grid 0.28 + final 0.36 | **5.2** | 3 | 15.6 |
| Supervised NSGA-II, IDRiD | search 220 × 5 ep × 3.2 s = 1.0 h + ZiCo 0.16 h; SH 0.16 + α 0.06 + final 0.08 | **1.45** | 3 | 4.4 |
| nnU-Net 1000 ep, IDRiD | 10 × 68.5 min + 2 sessions of data export/preprocessing + test | **11.6** | 3 | 35.0 |
| UNeXt BUSI / CVC / IDRiD / Kvasir (+ext.) | = Attention U-Net (data loading and validation dominate) | 0.62 / 0.55 / 0.22 / 1.03 | 12 | 7.3 |
| **required total** | | | **21** | **≈ 62 GPU h** (≈ 72 with the scheduler's 15 % margin) |
| optional: nnU-Net 1000 ep, BUSI | 10 × 180 min + 3–4 sessions of setup | 30.2 | 3 | 90.5 |

Training epochs spent by Supervised NSGA-II's search: ≤ 220 × 5 = **1,100 epochs per run** (QENAS: 0).
Per method: Supervised NSGA-II 20 GPU h, nnU-Net 1000 epochs 35 GPU h (+90.5 optional), UNeXt 7.3 GPU h.

## 5. Assignment to the five Kaggle accounts

Rules: one job per GPU, two GPUs per session, session ≤ 12 h (`SESSION_H = 11`: the start budget; long runs pause
before it, leaving time for upload and push); an nnU-Net run stays on **one** account (its checkpoint lives in that
account's dataset); each account's weekly GPU quota (≈ 30 h/week on Kaggle — check *Settings → Quota*: it is shown
there and can change; whether a *T4 x2* hour counts once or twice is not documented — the plan fits either way).

| account | GPU 0 | GPU 1 | GPU h | sessions (wall) |
|---|---|---|---:|---|
| **1** | nnU-Net IDRiD s45 (sessions 1–2) | Supervised BUSI s45 → Supervised IDRiD s45 | 18.3 | 2 (≈ 11.3 h + 1.2 h) |
| **2** | nnU-Net IDRiD s46 | Supervised BUSI s46 → Supervised IDRiD s46 | 18.3 | 2 (≈ 11.3 h + 1.2 h) |
| **3** | nnU-Net IDRiD s47 | Supervised BUSI s47 → Supervised IDRiD s47 | 18.3 | 2 (≈ 11.3 h + 1.2 h) |
| **4** | UNeXt BUSI, CVC, IDRiD × 3 seeds (9 jobs, both GPUs) | | 4.2 | 1 (≈ 2.5 h) |
| **5** | UNeXt Kvasir-SEG × 3 seeds + external | | 3.1 | 1 (≈ 2.4 h) |
| optional 4 / 5 | nnU-Net BUSI s45 (acct 4); s46 + s47 (acct 5) | | +30 / +60 | 3 each (≈ 31 h) → spans 2 quota weeks |

Each account uses at most ≈ 12.5 session hours per week (≤ 25 even if a T4 x2 hour counted twice), so the required
work finishes in **one week, two sessions on accounts 1–3 and one on 4–5**, about 12.5 h of wall-clock time if all
accounts start together. One seed per account for 1–3 keeps the three repetitions independent of one account.

## 6. Identity check across accounts

* `CODE_COMMIT`: every account checks out the same commit (logged).
* `scripts/revision_checksums.py --check` in every session: SHA-256 of the code (all `qenas/**.py`, `study.py`,
  `main.py`, `splits/*.json`), of each saved split (`splits/<DATASET>.json`, ordered sample ids of study v2, identical
  in all 105 v2 experiments) and of every image and mask byte of each dataset in split order, compared with
  `splits/data_checksums.json` (computed on the reference PC); stops on a mismatch. The result is saved as
  `results/revision_checksums_<W>.json` and pushed.
* At merge time `revision_tables.py` compares the five files: code, data, splits and software (torch, CUDA, cuDNN,
  numpy, pymoo, nnunetv2, timm, GPU) must be identical; differences from the study-v2 software are listed.
* Independently, each experiment's split stage refuses a split that differs in membership or order from the saved one.

## 7. Collecting and merging the results (checklist)

1. All five logs show `0 left`; accounts 1–3 show `finished IDRID/nnunet_full/seed_…` (not `paused`).
2. On the PC (QENAS repository, any branch; nothing is staged):

   ```bash
   python scripts/revision_tables.py --fetch
   ```

   It fetches `kaggle-revision-1` … `-5`, extracts their `results/` into `results/` (new experiment folders only),
   prints the identity comparison (must end with `5 accounts compared: identical`) and `21/21 required revision jobs
   done`, and writes `results/revision/` (R1–R5 + `report.json`).
3. Check in `R5`: rung-1 identity line; in `R3`: the TOST verdict with margin ±2.
4. Weights not on GitHub: nnU-Net final checkpoints are in each account's dataset `qenas-revision-persist-W`
   (download only if needed); UNeXt and Supervised NSGA-II final weights are on GitHub (`checkpoints/final/best.pt`).
5. Optional: `python study.py figures` regenerates per-experiment figures (PNG are not pushed).
6. After acceptance of the numbers, the Kaggle datasets `qenas-revision-persist-1…3` may be deleted.

## 8. No other runs are needed

* **QENAS** is not re-run: its code path is unchanged (the refactors are behaviour-preserving, `DEFAULT_CONFIG` is
  untouched — the 21 component-ablation experiments run with this code base keep their configuration fingerprints —
  and the existing test suite passes), and its study-v2 runs (seeds 45–47, same splits) are the reference of every
  new comparison. (The 84 v2 configurations differ from today's only by the key `data.split_unit = "dataset"`, removed
  after v2 together with the frame-split study; it never changed the dataset-level split, which is verified against
  `splits/*.json` in every new run.)
* **U-Net, Attention U-Net, nnU-Net (100 epochs)**: existing v2 runs, re-used for the Holm family and the five-network
  Friedman test, which only need UNeXt's per-image results in addition.
* **External validation**: only UNeXt is new (inside its Kvasir-SEG jobs); the other four exist.
* **Supervised NSGA-II** needs no extra QENAS run: QENAS v2 used the same seeds, splits, protocol and stages.
* **nnU-Net 1000 epochs** is required on IDRiD only (optional BUSI); CVC-ClinicDB and Kvasir-SEG are not needed for
  the equivalence claim, which the paper makes on BUSI / IDRiD.

So the revision consists of exactly the 21 required runs (+ 3 optional) above.
