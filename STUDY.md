# QENAS evaluation study v2 — experimental protocol

The study tests QENAS 3.0 (the method of `main.py`, METHODOLOGY.md) the way a journal paper must. QENAS itself is
not changed by the study. Every comparison condition is defined in `qenas/methods.py`, shares all other code with
QENAS, and is run by `study.py`.

**Study v1** (QENAS 2.1, seeds 42–44, 78 experiments) is complete: code at git tag `study-v1`, results on the
GitHub branch `kaggle-study` (local copy `results_v1/`), report `reports/QENAS_Study_v1.pdf`. Its findings led to
QENAS 3.0 (DESIGN_DECISIONS, table at the end). **Study v2 is productive, not complete:** it runs only what changed,
what was never tested, or what needs more evidence, with **new seeds 45, 46, 47**, so the changes are not judged
on the runs that motivated them. The split is unchanged, so v1 and v2 are tested on the same images.

## What v2 does not repeat (answered by v1)

| question | v1 answer | where |
|---|---|---|
| Does a random frame split inflate CVC-ClinicDB scores? | yes: +20.29 pp (QENAS), +28.41 pp (U-Net) | v1 T16, D35 |
| How do params, SynFlow and ZiCo relate? | ZiCo ⟂ size; SynFlow agrees with ZiCo and rewards size | v1 T6, T7, D14 |
| Does SynFlow help? | no; removing it was better on Kvasir-SEG | v1 T3, T4, D10 |
| Non-dominated share with 3 objectives | 40–51 % (pop 20); final population fully non-dominated | v1 T8, D16 |
| Do the proxies predict trained quality inside the elite? | no (ρ n.s. everywhere) | v1 T9, D18 |
| Does flip TTA help? | no (±0.5 pp) | v1 T11, D27 |

## Research questions of v2

| RQ | question | experiments | output |
|---|---|---|---|
| RQ1 | QENAS 3.0 vs hand-designed networks and nnU-Net on the same splits | QENAS, U-Net, Attention U-Net (re-run under the v2 protocol), **nnU-Net (new)** on BUSI, CVC-ClinicDB, IDRiD, Kvasir-SEG | T1, T2 |
| RQ2 | **Does optimising the loss coefficients help, or does it only cost time?** (never tested) | QENAS (α grid) vs `qenas_fixed_alpha` (same architecture, α = 0.5), all four datasets | T3, T4 |
| RQ3 | Does the new selection (successive halving + end-of-run Dice) beat a random candidate and the zero-shot rule? | `qenas_random_pick`, `qenas_zico_pick` on BUSI and Kvasir-SEG (as in v1) | T5, T6, T8 |
| RQ4 | Does NSGA-II on (params, ZiCo) still beat random search with the same budget? | random search, all datasets (training-free) | T7 |
| RQ5 | External validation of all four methods | Kvasir-SEG-trained models on all 612 CVC-ClinicDB frames | T12, T13 |
| — | v1 → v2: did the changes help? | QENAS 2.1 (v1) vs QENAS 3.0; baselines under both protocols, same test images | T14 |
| — | Evidence for D20, D21, D25, D29 | from the runs above | T8–T11, T15 |

## Conditions (`qenas/methods.py`)

| name | what differs from QENAS | datasets |
|---|---|---|
| `qenas` | — (the method, QENAS 3.0) | all four |
| `qenas_fixed_alpha` | the identical QENAS architecture (inherited search and selection) trained with α = β = 0.5, no α grid | all four |
| `qenas_random_pick` | final architecture = a uniformly random one of QENAS' 16 candidates (seed includes the dataset, D50) | BUSI, Kvasir-SEG |
| `qenas_zico_pick` | final architecture = the candidate with the highest ZiCo, chosen without training (zero-shot rule) | BUSI, Kvasir-SEG |
| `unet` | U-Net, 31.04 M parameters (the Mixed-GGNAS baseline variant), no search | all four |
| `attention_unet` | Attention U-Net, 8.73 M (the Mixed-GGNAS baseline variant), no search | all four |
| `nnunet` | nnU-Net v2 with its own pipeline (2d, default plans, `nnUNetTrainer_100epochs`), fold 0 = the QENAS train/validation split (D46, D47) | all four |
| `random_search` | uniform random sampling instead of NSGA-II, same number of evaluations (220), no training | all four |

**Fairness.** U-Net and Attention U-Net are trained with exactly the protocol QENAS uses for its own selected
architecture: α grid on validation (20-epoch runs), final training (AdamW, warm-up + poly, **120 epochs**, early
stopping only on overfitting), the best-validation checkpoint, **no TTA**, a single test evaluation. Same loss,
augmentation, resolution and batch size. nnU-Net uses its own pipeline on the identical images and split, and
receives at least the GPU time of the complete QENAS pipeline (D47). The inheriting conditions re-use QENAS' search
and successive-halving runs of the same seed and change only one rule.

## Data and splits

| dataset | images | split (train / validation / test) | leakage unit |
|---|---:|---|---|
| BUSI | 645 | 413 / 103 / 129, stratified benign/malignant | image (exact duplicates handled) |
| CVC-ClinicDB | 612 | 395 / 96 / 121 | colonoscopy sequence (29) |
| IDRiD (optic disc) | 81 | 43 / 11 / 27 (official test split) | image |
| Kvasir-SEG | 1000 | 704 / 176 / 120 (official `val.txt` = test split) | image |

**The split is identical for every condition, every seed and both studies** (`data.split_seed = 42`).

## Repetitions

Seeds **45, 46, 47** for every condition: **84 experiments**, plus 12 external evaluations (inside the Kvasir-SEG
jobs of QENAS, U-Net, Attention U-Net and nnU-Net).

## Metrics

Computed per test image at the network resolution and averaged:

* the Mixed-GGNAS metrics: Dice (= DSC), mIoU, parameters, FLOPs (thop convention per training batch, verified
  against the Mixed-GGNAS tables);
* HD95 (boundary) and sensitivity (clinical);
* foreground IoU;
* inference time per image (median of 50 forward passes, batch 1, on the run's GPU).

Per-image metrics and binary predictions of every test evaluation are saved (`test_per_image.csv`,
`test_predictions.npz`).

## Statistics (`qenas/study/analysis.py`)

* mean ± SD over the three seeds;
* paired comparisons on the same test images: per-image Dice averaged over the seeds, two-sided Wilcoxon
  signed-rank test, mean difference with a 95 % bootstrap CI (10 000 resamples), Holm–Bonferroni correction within
  each table and dataset;
* the loss-coefficient ablation is additionally paired by seed (same architecture), with the validation gain and
  the time of the α grid (T3);
* search quality: hypervolume of the non-dominated set of all evaluated architectures (union-normalised,
  2 objectives, reference point 1.1), NSGA-II vs random search with the same number of evaluations;
* successive halving: Kendall τ between the rungs, the rank of the selected candidate after 5 epochs, and the
  choice ΔDice would have made among the finalists (T8);
* proxy validity: Spearman ρ between params / ZiCo and the 5-epoch criterion of all 16 candidates (T9).

## Running it

Locally, the whole plan:

```bash
python study.py run --gpus 0
```

The progress of every job:

```bash
python study.py status
```

All paper tables and figures, written to `results/analysis/` (with `--v1-results` also the v1 → v2 comparison):

```bash
python study.py analyse --data-root ../Datasets --v1-results results_v1
```

On Kaggle (two T4 GPUs, 12 h sessions), use the notebook cell in README.md §6. Each session continues where the
previous one stopped:

* finished experiments are skipped;
* unfinished ones resume from their last checkpoint;
* jobs that cannot finish within the session are not started;
* progress is pushed to the GitHub branch `kaggle-study-v2` every 30 minutes.

**Estimated compute** (v1 measurements scaled to the v2 protocol; nnU-Net estimated conservatively at 2–2.5 h per
run): about 90 GPU-hours, i.e. about 45 hours with two GPUs in parallel — 4–5 Kaggle sessions. Measured durations
replace the estimates as jobs finish.

## Outputs

* `results/<DATASET>/<method>/seed_<s>/`: one experiment (all stage outputs, figures, logs; only the final weights
  are kept; nnU-Net: its plans and training log in `nnunet/`);
* `results/KVASIR/<method>/seed_<s>/external_CVC/`: external validation;
* `results/study_status.json`, `results/study_logs/`: scheduler state and one log per job;
* `results/analysis/`: tables T1–T15 (Markdown + JSON), figures, and a README mapping every file to its research
  question.
