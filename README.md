# QENAS — Quick Evolutionary NAS for medical image segmentation

QENAS searches U-shaped segmentation networks built from **five block types**: four manually designed blocks
(Residual, Dense, Inception, ConvNeXt) and a DARTS block. These are the blocks of the Mixed-GGNAS search space,
implemented from the official code.

The search uses **NSGA-II (pymoo)** with **three training-free objectives**. No weight is updated during the
search.

| objective | measures | direction |
|---|---|---|
| f1 = number of parameters | model complexity | minimised |
| f2 = 1/(1+ln(1+SynFlow)) | signal propagation of the initialised network (data-free) | SynFlow maximised |
| f3 = 1/(1+ZiCo) | gradient signal-to-noise of the segmentation loss on two training batches (official ZiCo code) | ZiCo maximised |

After the search:

1. A representative subset (≤ 8) of the final Pareto front is trained for 20 epochs.
2. The architecture with the strongest **learning capability** is selected (validation ΔDice with pathology
   gates).
3. The loss weights are chosen on the validation set.
4. The selected architecture is trained normally (100 epochs) with a validation set. The best-validation
   checkpoint is kept, and the validation set decides whether flip test-time augmentation is used.
5. It is tested **once**. The test reports:
   * the Mixed-GGNAS metrics: **Dice (= DSC)**, **mIoU**, **Params**, **FLOPs**;
   * **HD95** (boundary);
   * **Sensitivity** (clinical).

There is **one pipeline and one protocol**. The method has no options, modes or alternative paths.

* Method: [METHODOLOGY.md](METHODOLOGY.md)
* Every decision, with reasons and evidence: [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)
* Dataset structure and splits: [DATASETS.md](DATASETS.md)

---

## 1. Layout

```
GGNAS/
├── Datasets/                      BUSI (Breast Ultrasound Image)/  CVC-ClinicD (Polyp)/  IDRID/
└── QENAS/
    ├── main.py                    command-line entry point (the whole pipeline)
    ├── qenas/
    │   ├── config.py              the QENAS protocol (one fixed configuration) + per-dataset data facts
    │   ├── experiment.py          experiment directories, stage state, safe resume
    │   ├── pipeline.py            the nine stages (split → … → report)
    │   ├── datasets/              discovery (per dataset), leakage-free splits, preprocessing, augmentation, loaders
    │   ├── models/                official Mixed-GGNAS blocks, NAS-Unet DARTS ops/cells + genotypes, QENASNet,
    │   │                          chromosome encoding, model summary (params, FLOPs)
    │   ├── search/                objectives (params, SynFlow, ZiCo), cache, pymoo problem/operators,
    │   │                          NSGA-II ask/tell loop with checkpoints and hypervolume, Pareto subset selection
    │   ├── training/              loss (BCE + soft mIoU), metrics (Dice, IoU, mIoU, sensitivity, HD95), optimiser,
    │   │                          trainer, overfitting-based early stopping, learning-capability selection
    │   ├── visualization/         publication figures, architecture diagram
    │   └── utils/                 atomic I/O, checkpoints, seeding, environment, logging/ETA
    ├── scripts/inspect_datasets.py   dataset inspection (no model, no training)
    ├── tests/                     unit tests + an end-to-end pipeline test (pytest)
    ├── evidence/                  one-off studies behind design decisions (not part of the pipeline)
    ├── results/<DATASET>/experiment_YYYYMMDD_HHMMSS/   created by runs of this version
    ├── old results/               runs of the previous (two-objective) version — kept for reference only
    └── cache/                     zero-cost evaluation cache, preprocessed-image cache (created by runs)
```

## 2. Installation

Python ≥ 3.9 (tested with 3.13.7). Install PyTorch for your platform/CUDA from <https://pytorch.org>, then:

```bash
pip install -r requirements.txt
```

The datasets must be in `GGNAS/Datasets/` (default) or another folder given with `--data-root`. The folder names
only need to *start with* `BUSI`, `CVC` and `IDRID`.

## 3. Verify the installation

Run the unit tests and the end-to-end pipeline test (all nine stages on a small synthetic dataset; a few minutes
on a CPU):

```bash
python -m pytest -q tests
```

Inspect the three datasets (structure, pairing, integrity, resulting splits; no model is built):

```bash
python scripts/inspect_datasets.py
```

## 4. Running QENAS

Each command runs the complete pipeline for one dataset, from the split to the final test and report:

```bash
python main.py --dataset BUSI
```

```bash
python main.py --dataset CVC
```

```bash
python main.py --dataset IDRID
```

The results are written to `results/<DATASET>/experiment_YYYYMMDD_HHMMSS/`. The research summary (`summary.txt`)
is printed at the end.

| Stage | What happens | Main outputs |
|---|---|---|
| `split` | dataset discovery, integrity checks, deterministic leakage-free split | `dataset_report.json`, `split.json` |
| `search` | training-free NSGA-II (params, SynFlow, ZiCo), population 20, generation 0 + 10 generations | `search_results.csv/.json`, `search_population_history.csv/.json`, `nsga2_generations.csv/.json/.md` (incl. hypervolume), `pareto_front.csv/.json/.md`, `search_summary.json`, Pareto and progress figures |
| `pareto` | representative subset of the final front (≤ 8) | `pareto_candidates.csv/.json`, 3-D front figure |
| `candidates` | 20-epoch supervised training of each candidate | `candidate_training.csv`, `candidate_histories/` |
| `selection` | learning-capability selection | `selection.json`, `candidate_comparison.csv/.json/.md`, candidate figures |
| `loss_weights` | α/β grid on validation (α = 0.5 run reused) | `loss_weights.csv/.json`, figure |
| `final` | supervised training with validation (100 epochs), best-validation checkpoint restored, flip-TTA decision on validation | `final_training.csv`, `final_validation.json`, curves |
| `test` | **single** test evaluation of the best-validation checkpoint | `final_metrics.json`, `final_result.csv/.json/.md` |
| `report` | model summary, architecture figure, tables, reproducibility record, research summary | `model_summary.txt/.json`, `reproducibility.json`, `summary.txt` |

### Run settings

The method is fixed in `qenas/config.py`. The command line only takes run settings, which never change the
method:

| option | meaning |
|---|---|
| `--dataset BUSI\|CVC\|IDRID` | dataset (required) |
| `--data-root PATH` | folder containing the dataset folders (default `../Datasets`; Kaggle's `/kaggle/input` is searched automatically) |
| `--device auto\|cpu\|cuda\|cuda:N` | device (default `auto`: CUDA when available) |
| `--num-workers N` | data-loader worker processes (default 0) |
| `--seed N` | random seed (default 42). Different seeds give independent repetitions of the same method |
| `--no-resume` | start a new experiment instead of resuming or reporting the latest one |
| `--stop-after STAGE` | stop cleanly after a stage; the same command without it continues from there |

Examples:

Run on a GPU:

```bash
python main.py --dataset BUSI --device cuda
```

Data stored elsewhere:

```bash
python main.py --dataset CVC --data-root /path/to/Datasets
```

Search only (e.g. on a CPU machine), then continue the same experiment later (e.g. on a GPU machine):

```bash
python main.py --dataset IDRID --stop-after search
```

```bash
python main.py --dataset IDRID --device cuda
```

An independent repetition with another seed:

```bash
python main.py --dataset BUSI --seed 1
```

A new experiment with the default seed after a finished one:

```bash
python main.py --dataset BUSI --no-resume
```

`python main.py --help` lists the options. Each run's `config.json` contains the fully resolved configuration.

## 5. Interruptions and resume

Nothing long runs without a checkpoint:

* the search is checkpointed after every generation;
* every training run (candidates, α runs, final training) is checkpointed after every epoch, including
  optimiser, scheduler, scaler and RNG states.

If a run stops for any reason, **run the same command again**. The experiment with the same configuration is
found and resumed from the last completed generation or epoch.

* A finished experiment is never modified: re-running the same command prints its summary.
* If the configuration changed (a different code version or seed) and the latest experiment is unfinished,
  QENAS refuses to resume it and lists the differing keys. Use `--no-resume` to start a new experiment.
* The device and the data root are not part of the configuration fingerprint. You can therefore run the search
  on a CPU and continue training on a GPU machine: copy `QENAS/` and `Datasets/` together.

## 6. Compute time

* **Search.** On the development CPU (measured under full load), ZiCo takes about 51 s per architecture on
  average (two forward/backward passes at 256×256, batch 8) and SynFlow about 8 s. That makes about 3–4 hours
  for the ≤ 220 evaluations of the search. On a GPU the search takes minutes.
* **Supervised training.** It dominates the cost:
  * on a CPU, one BUSI epoch takes several minutes, and the full pipeline (8 × 20 candidate epochs, 4 × 20 α
    epochs, 100 final epochs) takes days;
  * a CUDA GPU turns this into hours and is strongly recommended.

**OneDrive:** this project lives in a OneDrive folder. Sync clients briefly lock new files. QENAS retries locked
writes automatically, but long runs produce many checkpoint writes. Pausing sync during long runs, or excluding
`QENAS/results` and `QENAS/cache` from sync, is recommended.

## 7. Outputs of one experiment

```
results/BUSI/experiment_YYYYMMDD_HHMMSS/
├── config.json  environment.json  state.json  reproducibility.json  summary.txt
├── dataset_report.json  split.json
├── search_results.csv/.json  search_population_history.csv/.json  search_summary.json
├── nsga2_generations.csv/.json/.md  pareto_front.csv/.json/.md  pareto_candidates.csv/.json
├── candidate_training.csv  candidate_results.json  candidate_histories/
├── selection.json  candidate_comparison.csv/.json/.md
├── loss_weights.csv/.json  loss_weight_histories/
├── final_training.csv  final_validation.json
├── final_metrics.json  final_result.csv/.json/.md
├── model_summary.txt/.json
├── checkpoints/   search_state.pt, candidates/cand_XX/{last,best}.pt, loss_weights/…, final/{last,best}.pt
├── figures/       (PNG, 300 dpi)
│   pareto_initial_population, pareto_generation_XXX, pareto_front_final, search_progress, pareto_front_3d,
│   pareto_front_final_with_candidates, candidates_val_dice, candidates_val_loss, candidates_delta_dice,
│   params_vs_val_dice, synflow_vs_val_dice, zico_vs_val_dice, learning_capability_ranking,
│   loss_weight_selection, final_training_curves, train_vs_val_dice, train_vs_val_loss, selected_architecture
└── logs/run.log
```

`final_result.md` holds the final table:

* Dice / DSC (%), mIoU (%), Params (M), FLOPs (G, per training batch as in Mixed-GGNAS);
* HD95 (px), Sensitivity (%);
* IoU (%), GFLOPs per image, and whether flip TTA was used.

## 8. Scientific safeguards (summary)

* **Search:** training-free; no weight is ever updated.
  * Three objectives (parameters, SynFlow, ZiCo). No TOPSIS, no weighted sum, no proxy combination.
  * ZiCo is computed with the unmodified official code on two training batches. The search never uses
    validation or test images.
* **Isolation:** the test split is loaded only by the final test stage, exactly once. Nothing is ever selected
  on test data.
* **Splits:** leakage-free:
  * CVC is split by colonoscopy sequence;
  * BUSI is stratified, with a conflicting duplicate removed;
  * IDRiD keeps its official split.
* **Consistency:** the same network class and the same initialisation are used for SynFlow, ZiCo, candidate
  training and final training. There is no scale selection or collapse anywhere.
* **Selection:** the learning-capability criterion uses falsifiable gates (finite, generalising, not
  overfitting, significant learning trend, better than trivial) and no hand-tuned weights.
* **Honesty:** results are reported as they are. Every adaptation of official code and every design choice is
  documented in DESIGN_DECISIONS.md.

## 9. Known limitations

* BUSI and IDRiD have no patient identifiers, so their splits are image-level (BUSI exact duplicates are
  handled).
* The dataset-specific DARTS genotypes were derived by the Mixed-GGNAS authors on their own (unknown) split
  (DESIGN_DECISIONS D5).
* SynFlow and ZiCo are zero-cost *proxies*. Their correlation with trained segmentation quality is not
  guaranteed. ZiCo also has a documented bias towards networks with more layers (D14). Every run reports both
  against the candidates' validation Dice (`synflow_vs_val_dice`, `zico_vs_val_dice`, with Spearman ρ).
* Bit-exact reproducibility is verified on CPU. Some CUDA kernels may differ in the last bits.
