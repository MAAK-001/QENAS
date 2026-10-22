# QENAS 3.0 — context for writing the paper

Hand-over from the analysis chat (2026-10-22). Read this first in a new chat.

## 1. What QENAS is (final version = QENAS 3.0 = study v2)
QENAS ("Quick ENAS") is a fast evolutionary NAS for medical image segmentation. Its aim is not a new
component but a large cut in search cost with competitive accuracy. Pipeline (one fixed configuration, `main.py`):
1. **Training-free NSGA-II search** (pymoo): population 20 × 10 generations; objectives f1 = #parameters,
   f2 = 1/(1+ZiCo) (ZiCo from 2 training batches, official code); uniform crossover p 0.9, random-reset
   mutation p 1/8. Search space: U-shaped net, 8 genes (4 encoder, 4 decoder), 5 blocks
   (0 Residual, 1 Dense, 2 Inception, 3 ConvNeXt, 4 DARTS cell) → 5^8 = 390,625 architectures; skip
   connections with spatial–channel attention (SCA); search space and blocks from Mixed-GGNAS (Hu et al. 2025).
2. **Successive halving** of 16 candidates (final front + fill-up): 16×5 → 8×10 → 4×20 epochs; selection =
   highest mean validation Dice of the last 5 epochs among finalists passing gates G1–G5.
3. **Loss weight**: L = α·BCE + (1−α)·soft-mIoU, α ∈ {0.2,0.4,0.5,0.6,0.8} chosen on validation (20 epochs each).
4. **Final training** 120 epochs (AdamW, lr 1e-3, poly + warm-up, best-val-Dice weights); single test pass, no TTA.

Datasets: BUSI, CVC-ClinicDB (sequence-level split), IDRiD (optic disc), Kvasir-SEG; external test Kvasir→CVC
(612 frames). Split seed 42 fixed; 3 runs per model (seeds 45, 46, 47); GPU Tesla T4 (Kaggle).
Compared models (same split): nnU-Net v2 (2d, 100 epochs), Attention U-Net, U-Net; published results:
Hu et al. 2025 (Mixed-GGNAS paper, Tables 4/5/7/8) and Jha et al. 2021 (Kvasir-SEG).

## 2. Key results (test Dice %, QENAS best run / QENAS mean of 3 / nnU-Net mean)
| Dataset | QENAS best | QENAS mean | nnU-Net | Att. U-Net | U-Net | QENAS params |
|---|---|---|---|---|---|---|
| BUSI | 80.32 | 79.30 | 78.87 | 74.48 | 72.75 | ~1.0 M |
| CVC-ClinicDB | 72.16 | 69.59 | 76.48 | 60.05 | 48.77 | 0.6–2.6 M |
| IDRiD | 96.13 | 95.70 | 95.90 | 92.46 | 91.90 | ~0.9–1.6 M |
| Kvasir-SEG | 85.78 | 84.98 | 87.31 | 84.20 | 81.33 | 1.3–2.5 M |
| Kvasir→CVC (ext.) | 80.52 | 78.63 | 81.32 | 70.72 | 69.31 | |
nnU-Net has 33.5 M parameters (16–32× more). Best runs: BUSI/CVC/IDRiD seed 47, Kvasir seed 46.
QENAS vs nnU-Net: no significant difference on BUSI and IDRiD; nnU-Net better on CVC, Kvasir, external.
QENAS significantly better than Attention U-Net and U-Net almost everywhere.

**Time (use IDRiD everywhere — the only dataset with published search times, Hu et al. Table 8):**
QENAS best IDRiD run: stage 1 = 8 min, stage 2 = 9 min, stage 3 = 4 min (search + selection ≈ 22 min),
final training 7 min, total ≈ 30 min. Published NAS search times on IDRiD: 3–26 h (RTX 4070 Ti Super).
Stage 1 is 4–8 min on every dataset; search + selection on other datasets: 74–102 min (be transparent).

Ablations (mean of 3 runs): α search helps BUSI (+0.47, sig.), hurts IDRiD (−0.42, sig.), n.s. elsewhere;
Dice-based selection beats random candidate (+1.35) and zero-shot max-ZiCo pick (+0.96) on Kvasir (sig.),
tie on BUSI; NSGA-II beats random search in hypervolume 12/12 runs.

## 3. Decisions made with the user (do not change without asking)
- Comparison figures/tables: **QENAS = best run** (labelled "QENAS (best run)"), other models = mean ± SD of
  3 runs. Tables additionally show a "QENAS (mean ± SD, 3 runs)" row. Fair mean-vs-mean versions exist in `review/`.
- **Never single out Mixed-GGNAS**; it is one of the published methods, same marker/colour as the others.
- Published CVC-ClinicDB values are excluded from comparisons (random frame split, not comparable).
- Figures: Times New Roman, one font size (9 pt), 600 dpi PNG, **no titles/captions inside images** (file name
  = title), clear axis labels and legends, no star markers (selected point = different colour), Pareto front =
  thin straight line.
- User prefers short, simple answers; ask before doing anything not requested.

## 4. Where everything is (all under `C:\Users\aquee\OneDrive\Desktop\GGNAS\QENAS\`)
| Path | Content |
|---|---|
| `qenas/` | QENAS 3.0 code (search, models, training, study) |
| `main.py`, `study.py` | run QENAS / the comparison study |
| `README.md` | run commands (incl. Kaggle cell, figure/table scripts) |
| `DESIGN_DECISIONS.md` | every design decision with evidence (D1–D50) |
| `STUDY.md`, `PAPER_POSITIONING.md` | study design, paper positioning notes |
| `results/<DATASET>/<method>/seed_<s>/` | raw results of every run (metrics, histories, predictions) |
| `results/analysis/` | study analysis tables T1–T15 (markdown + json) |
| `results/figures/1 Comparison with other models/` | 11 comparison figures |
| `results/figures/2 QENAS stages/` | 7 stage figures (BUSI best run) |
| `results/figures/3 Architecture/` | network architecture, candidate blocks, search & training stages, found BUSI architecture |
| `results/figures/4 Qualitative results/` | predicted masks of all models, per dataset + external |
| `results/figures/review/` | extra figures (ablation, hypervolume, proxies, block usage, all 12 architectures, fair Dice) |
| `results/tables/QENAS Tables.docx` | **all 22 paper tables** (bold = best per column) |
| `results/paper/tables/*.csv`, `report.json` | raw table data from `scripts/paper_analysis.py` |
| `scripts/paper_figures.py`, `paper_architecture.py`, `paper_diagrams.py`, `paper_tables.py`, `render_tables_docx.js` | regenerate figures / tables |
| `Old Results/` | preliminary study v1 (QENAS 2.1) — not part of the final paper |
| `../Datasets/` | the four datasets |

Target venue discussed: IEEE Transactions on Evolutionary Computation (or a Q1 medical-imaging/AI journal).
Nothing has been committed to git since the study finished; commit only when the user asks.
