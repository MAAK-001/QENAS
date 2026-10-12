# QENAS evaluation study — experimental protocol

The study tests QENAS (the method of `main.py`, METHODOLOGY.md) the way a journal paper must. QENAS itself is
not changed by the study. Every comparison condition is defined in `qenas/methods.py`, shares all other code with
QENAS, and is run by `study.py`.

## Research questions

| RQ | question | experiments | output |
|---|---|---|---|
| RQ1 | How does QENAS compare with hand-designed networks trained on the *same* splits? | QENAS, U-Net, Attention U-Net on BUSI, CVC-ClinicDB, IDRiD, Kvasir-SEG | T1, T2 |
| RQ2 | What does each QENAS component contribute? | ablations on BUSI and Kvasir-SEG (below) | T3, T4 |
| RQ3 | Does NSGA-II search better than random sampling with the same budget? | random search, all datasets (training-free) | T5 |
| RQ4 | Do the models generalise to another hospital and device? | Kvasir-SEG-trained models tested on all 612 CVC-ClinicDB frames | T14, T15 |
| RQ5 | How much does a random frame split inflate CVC-ClinicDB scores? | QENAS and U-Net with a frame-level split | T16 |
| — | Evidence for the design decisions (D14, D16, D25, D27, D29) | from the runs above | T6–T13 |

## Conditions (`qenas/methods.py`)

| name | what differs from QENAS | datasets |
|---|---|---|
| `qenas` | — (the method) | all four |
| `unet` | U-Net (31.04 M parameters, the Mixed-GGNAS baseline variant), no search | all four |
| `attention_unet` | Attention U-Net (8.73 M, the Mixed-GGNAS baseline variant), no search | all four |
| `qenas_no_zico` | NSGA-II on (params, SynFlow) only | BUSI, Kvasir-SEG |
| `qenas_no_synflow` | NSGA-II on (params, ZiCo) only | BUSI, Kvasir-SEG |
| `qenas_random_pick` | final architecture = a uniformly random one of QENAS' Pareto candidates | BUSI, Kvasir-SEG |
| `qenas_zico_pick` | final architecture = the Pareto candidate with the highest ZiCo, chosen without any training (the rule of zero-shot NAS, e.g. ZiCo, TF-MOENAS-style methods) | BUSI, Kvasir-SEG |
| `random_search` | uniform random sampling instead of NSGA-II, same number of evaluations (220), no training | all four |
| CVC frame split | `data.split_unit = image`: frames instead of colonoscopy sequences are split | CVC-ClinicDB (QENAS, U-Net) |

**Fairness.** The baselines are trained with exactly the protocol QENAS uses for its own selected architecture:

* loss-weight selection on validation (α grid, 20-epoch runs);
* final training: AdamW, warm-up + poly, 100 epochs, early stopping only on overfitting;
* the best-validation checkpoint;
* the flip-TTA decision on validation;
* a single test evaluation.

Same loss, augmentation, resolution and batch size. The selection ablations re-use QENAS' search and candidate
runs of the same seed and only change the selection rule, so they isolate that component.

## Data and splits

| dataset | images | split (train / validation / test) | leakage unit |
|---|---:|---|---|
| BUSI | 645 | 413 / 103 / 129, stratified benign/malignant | image (exact duplicates handled) |
| CVC-ClinicDB | 612 | 395 / 96 / 121 | colonoscopy sequence (29) |
| IDRiD (optic disc) | 81 | 43 / 11 / 27 (official test split) | image |
| Kvasir-SEG | 1000 | 704 / 176 / 120 (official `val.txt` = test split) | image |

**The split is identical for every condition and every seed** (`data.split_seed = 42`), so all methods are tested
on exactly the same images and paired tests are possible. The run seed changes only the search, the
initialisation, the data order and the augmentation.

## Repetitions

Seeds **42, 43, 44** for every condition: 78 experiments, plus 9 external evaluations.

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
* paired comparisons on the same test images: per-image Dice averaged over seeds, two-sided Wilcoxon signed-rank
  test, mean difference with a 95 % bootstrap CI (10 000 resamples), Holm–Bonferroni correction within each
  table and dataset;
* search quality: hypervolume of the non-dominated set of all evaluated architectures (union-normalised,
  3 objectives, reference point 1.1), NSGA-II vs random search with the same number of evaluations;
* proxy validity: Spearman ρ between params / SynFlow / ZiCo and the candidates' 20-epoch validation Dice.

## Running it

Locally, the whole plan:

```bash
python study.py run --gpus 0
```

The progress of every job:

```bash
python study.py status
```

All paper tables and figures, written to `results/analysis/`:

```bash
python study.py analyse --data-root ../Datasets
```

On Kaggle (two T4 GPUs, 12 h sessions), use the notebook cell in README.md §9. Each session continues where the
previous one stopped:

* finished experiments are skipped;
* unfinished ones resume from their last checkpoint;
* jobs that cannot finish within the session are not started;
* progress is pushed to a GitHub branch every 30 minutes.

**Estimated compute:** about 120 GPU-hours, i.e. about 60 hours with two GPUs in parallel. That is about
5–6 Kaggle sessions (roughly two weeks of the free 30 h/week quota). The estimate is conservative; measured
durations replace it as jobs finish.

## Outputs

* `results/<DATASET>/<method>/seed_<s>/`: one experiment (all stage outputs, figures, logs; only the final weights
  are kept);
* `results/CVC_frame_split/...`: the leakage study;
* `results/KVASIR/<method>/seed_<s>/external_CVC/`: external validation;
* `results/study_status.json`, `results/study_logs/`: scheduler state and one log per job;
* `results/analysis/`: tables T1–T16 (Markdown + JSON), figures, and a README mapping every file to its research
  question.
