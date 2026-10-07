# QENAS — Quick Evolutionary NAS for medical image segmentation

QENAS searches U-shaped segmentation networks built from **five block types**: four manually designed blocks
(Residual, Dense, Inception, ConvNeXt) and a DARTS block. These are the blocks of the Mixed-GGNAS search
space, implemented from the official code. The search uses **NSGA-II (pymoo)** with **two training-free
objectives**:

* the number of parameters (minimised);
* **SynFlow** (maximised, as the positive objective `f2 = 1/(1+ln(1+S))`).

After the search:

1. A representative subset of the final Pareto front is trained for 20 epochs.
2. The architecture with the strongest **learning capability** is selected (validation ΔDice with pathology
   gates).
3. The loss weights are chosen on the validation set.
4. The selected architecture is trained normally (no scale selection or collapse).
5. It is tested **once**.

* Method: [METHODOLOGY.md](METHODOLOGY.md)
* Every open decision, with reasons: [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)
* Dataset structure and splits: [DATASETS.md](DATASETS.md)

---

## 1. Layout

```
GGNAS/
├── Datasets/                      BUSI (Breast Ultrasound Image)/  CVC-ClinicD (Polyp)/  IDRID/
└── QENAS/
    ├── main.py                    command-line entry point (whole pipeline)
    ├── configs/                   smoke.yaml (tiny end-to-end check), example_override.yaml
    ├── qenas/
    │   ├── config.py              defaults, per-dataset data facts, YAML/CLI overrides, fingerprint
    │   ├── experiment.py          experiment directories, stage state, safe resume
    │   ├── pipeline.py            the nine stages (split → … → report)
    │   ├── datasets/              discovery (per dataset), leakage-free splits, preprocessing, augmentation, loaders
    │   ├── models/                official Mixed-GGNAS blocks, NAS-Unet DARTS ops/cells + genotypes, QENASNet,
    │   │                          chromosome encoding, model summary (params, MACs)
    │   ├── search/                objectives (params, SynFlow), cache, pymoo problem/operators,
    │   │                          NSGA-II ask/tell loop with checkpoints, Pareto subset selection
    │   ├── training/              loss (BCE + soft mIoU), metrics, optimiser/schedules, trainer,
    │   │                          overfitting-based early stopping, learning-capability selection
    │   ├── visualization/         publication figures, architecture diagram
    │   └── utils/                 atomic I/O, checkpoints, seeding, environment, logging/ETA
    ├── scripts/inspect_datasets.py
    ├── tests/                     unit tests (pytest)
    ├── results/<DATASET>/experiment_YYYYMMDD_HHMMSS/   (created by runs)
    └── cache/                     SynFlow evaluation cache, preprocessed-image cache (created by runs)
```

## 2. Installation

Python ≥ 3.9 (tested with 3.13.7). Install PyTorch for your platform/CUDA from <https://pytorch.org>, then:

```bash
pip install -r requirements.txt
```

The datasets must be in `GGNAS/Datasets/` (default) or another folder given with `--data-root`. The folder
names only need to *start with* `BUSI`, `CVC` and `IDRID`.

## 3. Verify the installation (5 minutes on a CPU)

```bash
python -m pytest -q tests
```

```bash
python scripts/inspect_datasets.py
```

```bash
python main.py --dataset BUSI --smoke
```

`--smoke` runs every stage with tiny settings (64×64 images, 24/12/12 samples, a few generations and epochs)
into `results/BUSI/smoke_*`. Its numbers are meaningless; it only proves the pipeline works end to end.

## 4. Running the full pipeline

Each command runs the full QENAS pipeline for one dataset from start to finish:

```bash
python main.py --dataset BUSI
```

```bash
python main.py --dataset CVC
```

```bash
python main.py --dataset IDRID
```

Use `--device cuda` (default `auto` picks CUDA when available).

| Stage | What happens | Main outputs |
|---|---|---|
| `split` | dataset discovery, integrity checks, deterministic leakage-free split | `dataset_report.json`, `split.json` |
| `search` | NSGA-II, population 10, generation 0 + 10 generations, training-free | `search_results.csv`, `search_population_history.csv`, `nsga2_generations.csv/.json/.md`, `pareto_front.csv/.json/.md`, Pareto figures |
| `pareto` | representative subset of the final front (≤ 8) | `pareto_candidates.csv/.json` |
| `candidates` | 20-epoch supervised training of each candidate | `candidate_training.csv`, `candidate_histories/` |
| `selection` | learning-capability selection | `selection.json`, `candidate_comparison.csv/.json/.md`, candidate figures |
| `loss_weights` | α/β grid on validation (α=0.5 run reused) | `loss_weights.csv/.json`, figure |
| `final` | ordinary supervised training (100 epochs), best-val checkpoint restored | `final_training.csv`, `final_validation.json`, curves |
| `test` | **single** test evaluation of the best checkpoint | `final_metrics.json`, `final_result.csv/.json/.md` |
| `report` | model summary, architecture figure, tables, reproducibility record, research summary | `model_summary.txt/.json`, `reproducibility.json`, `summary.txt` |

The research summary (`summary.txt`) is printed at the end.

### Common options

```text
--population-size 10   --generations 10   --pareto-candidates 8   --seed 42
--candidate-epochs 20  --epochs 100        --batch-size 8          --learning-rate 1e-3
--weight-decay 5e-5    --optimizer adamw   --scheduler poly        --alphas 0.2 0.4 0.5 0.6 0.8
--scale 3              --base-channels 32  --darts-genotype auto   --image-size H W
--idrid-target OD      --early-stopping-patience 20 | --no-early-stopping
--device auto|cpu|cuda --amp auto|true|false   --num-workers 0
--resume (default) | --no-resume | --experiment-dir PATH   --stop-after STAGE
--config my_overrides.yaml   (any key of qenas/config.py::DEFAULT_CONFIG)
```

`python main.py --help` lists everything. Each run's `config.json` contains the fully resolved configuration.

## 5. Interruptions and resume

Nothing long runs without a checkpoint:

* the search is checkpointed after every generation;
* every training run (candidates, α runs, final training) is checkpointed after every epoch, including
  optimiser, scheduler, scaler and RNG states.

If a run stops for any reason, **run the same command again**. The latest unfinished experiment of that
dataset is detected and resumed from the last completed generation or epoch. A run killed with SIGKILL and
resumed was verified to reproduce the uninterrupted results bit for bit (on CPU).

* If you change a scientific option, QENAS refuses to resume and lists the differing keys. Re-run with the
  original options, or pass `--no-resume` to start a new experiment.
* Completed experiments are never modified. Re-running prints their summary; `--no-resume` starts a new one.
* `--stop-after search` (or any stage) stops cleanly. The same command without it continues from there.
* The device is *not* part of the fingerprint, so you can run the search on a CPU and continue training on a
  GPU machine. Copy `QENAS/` and `Datasets/` together.

## 6. Compute time

The training-free search costs about 2–7 s per architecture on a CPU (measured): ≈ 6–8 minutes for BUSI
with population 10 and 10 generations. Supervised training dominates the cost.

* **CPU:** on the development machine (CPU-only PyTorch), one training step at batch 8 and 256×256 takes
  5–11 s, so one BUSI epoch takes several minutes. The full pipeline (8 × 20 candidate epochs, 4 × 20 α
  epochs, 100 final epochs) then takes **days**.
* **GPU:** a CUDA GPU is strongly recommended and turns this into hours.

To reduce cost without changing the method:

* use fewer Pareto candidates (`--pareto-candidates`);
* use a smaller α grid (`--alphas 0.4 0.5 0.6`);
* use fewer final epochs.

Any such change is recorded in `config.json`.

**OneDrive:** this project lives in a OneDrive folder. Sync clients briefly lock new files; QENAS retries
locked writes automatically, but long runs produce many checkpoint writes. Pausing sync during long runs, or
excluding `QENAS/results` and `QENAS/cache` from sync, is recommended.

## 7. Outputs of one experiment

```
results/BUSI/experiment_YYYYMMDD_HHMMSS/
├── config.json  environment.json  state.json  reproducibility.json  summary.txt
├── dataset_report.json  split.json
├── search_results.csv  search_population_history.csv  search_summary.json
├── nsga2_generations.csv/.json/.md  pareto_front.csv/.json/.md  pareto_candidates.csv/.json
├── candidate_training.csv  candidate_results.json  candidate_histories/
├── selection.json  candidate_comparison.csv/.json/.md
├── loss_weights.csv/.json  loss_weight_histories/
├── final_training.csv  final_validation.json  final_metrics.json  final_result.csv/.json/.md
├── model_summary.txt/.json
├── checkpoints/   search_state.pt, candidates/cand_XX/{last,best}.pt, loss_weights/…, final/{last,best}.pt
├── figures/       (PNG 300 dpi + PDF)
│   pareto_initial_population, pareto_generation_XXX, pareto_front_final, pareto_front_evolution,
│   pareto_front_final_with_candidates, pareto_front_log_synflow,
│   candidates_val_dice, candidates_val_loss, candidates_delta_dice, params_vs_val_dice,
│   synflow_vs_val_dice, learning_capability_ranking, loss_weight_selection,
│   final_training_curves, train_vs_val_dice, train_vs_val_loss, selected_architecture
└── logs/run.log
```

## 8. Scientific safeguards (summary)

* **Search:** training-free (no data, no optimiser) with exactly two objectives (parameters, SynFlow). No
  third proxy, no TOPSIS, no weighted sum.
* **Isolation:** the test split is loaded only by the final test stage, exactly once. Nothing is ever
  selected on test data.
* **Splits:** leakage-free:
  * CVC is split by colonoscopy sequence;
  * BUSI is stratified, with a conflicting duplicate removed;
  * IDRID keeps its official split.
* **Consistency:** the same network class and the same initialisation are used for SynFlow, candidate
  training and final training. There is no scale selection or collapse anywhere.
* **Selection:** the learning-capability criterion uses falsifiable gates (finite, generalising, not
  overfitting, significant learning trend, better than trivial) and no hand-tuned weights.
* **Honesty:** poor results are reported as they are. Every adaptation of official code is documented in
  DESIGN_DECISIONS.md.

## 9. Known limitations

* BUSI and IDRiD have no patient identifiers, so their splits are image-level (BUSI exact duplicates are
  handled).
* The dataset-specific DARTS genotypes were derived by the Mixed-GGNAS authors on their own (unknown) split.
  `--darts-genotype nasunet` gives a dataset-independent alternative (DESIGN_DECISIONS D4).
* SynFlow is a zero-cost *proxy*. Its correlation with trained segmentation quality is not guaranteed, and the
  candidate stage reports it (`synflow_vs_val_dice` figure with Spearman ρ).
* Bit-exact reproducibility is verified on CPU. Some CUDA kernels may differ in the last bits.
