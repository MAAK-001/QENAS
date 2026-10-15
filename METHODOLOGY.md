# QENAS — Methodology (QENAS 3.0)

**QENAS (Quick Evolutionary NAS)** searches U-shaped segmentation networks in a mixed search space of manually
designed and DARTS blocks. The search is **training-free**: NSGA-II optimises two objectives computed on the
*initialised* network, without updating a single weight:

* the number of parameters (complexity);
* **ZiCo**, a measure of the gradients that the segmentation loss produces on real training images.

Supervised training is used only after the search, to:

1. compare 16 candidates of the final population by their *learning capability*, with successive halving (most
   candidates get a short run, the promising ones a full 20-epoch run);
2. choose the loss weights on the validation set;
3. train the selected architecture normally with a validation set.

The test set is used exactly once, at the very end. There is one pipeline and one protocol. Nothing in the
method is optional. (QENAS 2.x additionally optimised SynFlow, selected by ΔDice among 8 candidates trained for
20 epochs each, trained for 100 epochs and decided flip-TTA on validation; DESIGN_DECISIONS lists every change
and the study-v1 evidence for it.)

```
Dataset ─► dataset-specific train / validation / test split (leakage-free)
        ─► search space (8 genes × 5 block types = 5^8 = 390,625 architectures)
        ─► random initial population (generation 0)
        ─► NSGA-II (pymoo), population 20, 10 generations; every individual evaluated training-free:
               f1 = #trainable parameters                (minimise)
               f2 = 1 / (1 + ZiCo)                       (minimise ⇔ maximise ZiCo)
        ─► 16 candidates: the final non-dominated front, filled up in NSGA-II's survival order
        ─► successive halving: 16 × 5 epochs → best 8 to 10 epochs → best 4 to 20 epochs (160 epochs)
        ─► learning-capability selection (mean validation Dice of the last 5 epochs, with pathology gates)
        ─► loss-weight (α, β) selection on validation
        ─► ordinary supervised training (120 epochs, best-validation checkpoint)
        ─► single test evaluation: Dice (DSC), mIoU, Params, FLOPs, HD95, Sensitivity
```

---

## 1. Search space

A chromosome is `[g1, …, g8]`, `gi ∈ {0,1,2,3,4}`:

| gene value | block | source |
|---|---|---|
| 0 | Residual block | He et al. 2016; official Mixed-GGNAS implementation |
| 1 | Dense block (3 layers, growth 32) | Huang et al. 2017; official Mixed-GGNAS implementation |
| 2 | Inception block (factorised strip convolutions) | Szegedy et al.; official Mixed-GGNAS implementation |
| 3 | ConvNeXt block (depthwise k×k, 4× MLP, residual) | Liu et al. 2022 / Ying et al. 2023; official Mixed-GGNAS implementation |
| 4 | DARTS block (NAS-Unet down/up cell from a DARTS genotype) | Liu et al. 2019; Weng et al. 2019; official Mixed-GGNAS genotypes |

Genes 1–4 are the encoder positions (each halves the resolution and doubles the channels). Genes 5–8 are the
decoder positions (each doubles the resolution and halves the channels). With base width *b* = 32:

| feature | E0 (stem) | E1 | E2 | E3 | E4 (bottleneck) | D1 | D2 | D3 | D4 |
|---|---|---|---|---|---|---|---|---|---|
| channels | b | 2b | 4b | 8b | 16b | 8b | 4b | 2b | b |
| resolution | H | H/2 | H/4 | H/8 | H/16 | H/8 | H/4 | H/2 | H |

Decoder position *j* receives the encoder feature at its output resolution as a skip connection
(D1←E3, D2←E2, D3←E1, D4←E0). A 1×1 convolution maps D4 to one logit per pixel.

*Manual blocks* are wrapped by the official Mixed-GGNAS transitions:

* encoder: `BN → AvgPool 2×2 → 3×3 grouped conv (C_in→C_out) → block`;
* decoder: `BN → 2×2 transposed conv (C_in→C_out)`; skip: `BN → s + s·σ_s(s)·σ_c(s)` (spatial-channel
  attention); then `concat → 1×1 conv → block`.

*DARTS blocks* are NAS-Unet cells with the genotype that Mixed-GGNAS published for the dataset:

* two inputs `s0, s1`, 1×1 preprocessing to `c = C_out/4` channels;
* four intermediate nodes, each the sum of two operations from the genotype;
* output = concatenation of the four nodes.

Down cells (encoder) take `s1` = previous feature and `s0` = the feature two levels back (2× resolution,
preprocessed with stride 2). Up cells (decoder) take `s1` = previous decoder feature and `s0` = the encoder skip
at the output resolution.

**Convolution scale.** Mixed-GGNAS selects among three convolution scales per block with weights learned from
the *supervised* loss. This needs training, so it is not part of QENAS. Every manual block uses one fixed scale
(scale 3 of Mixed-GGNAS Table 1) in the search, in candidate training and in final training. There is no
scale-selection stage, no scale collapse and no differentiable scale weight anywhere.

The same `QENASNet` class with the same seeded initialisation is used for ZiCo, candidate training and final
training. The networks ZiCo scores are therefore bit-identical to the networks that start supervised training.

## 2. Objectives (training-free)

**f1 — complexity.** `f1(a) = Σ_θ 1[θ trainable] · numel(θ)` (minimise).

**f2 — ZiCo** (Li et al., ICLR 2023). The official functions (`getgrad`, `caculate_zico`, `getzico` from
github.com/SLDGroup/ZiCo) are used without any change to their logic:

1. initialise (same seed); **train mode** (normalisation layers active);
2. for each of the **two fixed training batches**: zero the gradients, forward, compute the training loss
   `L = 0.5·BCE + 0.5·soft-mIoU` against the ground-truth masks, and backpropagate;
3. collect the gradient of every `Conv2d` / `Linear` weight;
4. for every layer *l*, over the two batches, compute `std(g)` and `mean(|g|)` of each weight gradient. Then
   `ZiCo = Σ_l log( Σ_{weights with std > 0} mean|g| / std(g) )`.

With two batches each ratio is ≥ 1, so ZiCo ≥ 0. The objective is `f2 = 1 / (1 + ZiCo) ∈ (0, 1]`.

The two ZiCo batches are the first two batches of training epoch 0: 8 images each (IDRiD 4), shuffled and
augmented exactly as in training. They are fixed for the whole search and identical for every architecture, as
in the official ZiCo search. ZiCo is the only objective that sees the dataset. It sees only **training** images
and their masks, and no weight is updated, so the search stays training-free. Validation and test images are
never used by the search.

Both objectives are positive and minimised. Pareto dominance is invariant to strictly monotone transforms, so
the ranking equals that of "minimise parameters, maximise ZiCo". The two are not combined into a single score.

**How the objectives relate** (study v1, 220 random architectures per dataset and seed; DESIGN_DECISIONS D14):

* ZiCo is almost independent of model size (Kendall τ −0.09 to −0.18), so the two objectives describe a genuine
  size–quality trade-off;
* SynFlow, the third objective of QENAS 2.x, agreed with ZiCo and mostly rewarded larger networks; removing it was
  never significantly worse and significantly better on Kvasir-SEG, so QENAS 3.0 does not use it (D10).

## 3. NSGA-II

pymoo's `NSGA2` (non-dominated sorting, crowding distance, binary tournament mating selection,
(μ+λ) rank-and-crowding survival), driven through pymoo's ask-and-tell interface:

* sampling: uniform integers per gene (`IntegerRandomSampling`);
* crossover: `UniformCrossover`, probability 0.9 (each gene from either parent with probability 0.5, i.e. the
  binomial crossover with CR = 0.5 used by Mixed-GGNAS);
* mutation: `ChoiceRandomMutation` (random resetting of categorical genes), per-gene probability 1/8;
* duplicate elimination on;
* **population 20**, generation 0 (random) + **10 evolutionary generations** (as in Mixed-GGNAS). With the two
  objectives, about 22 % of a random 20-individual population is non-dominated, so rank-based selection acts
  (with the three objectives of QENAS 2.x it was 40–51 %, and the whole final population ended non-dominated;
  D16).

Evaluations are cached. The cache key covers the chromosome, the architecture definition, the objective
definitions and the SHA-256 of the exact ZiCo batches. After every generation the whole algorithm state is
checkpointed, including its random generator. Each generation's population is recorded with its Pareto rank,
crowding distance and non-domination flag.

Search progress is measured by the **hypervolume** of every generation's non-dominated set. It is computed in the
objective space min–max normalised over all evaluated architectures, with reference point 1.1 in each objective
(figure `search_progress`).

## 4. Candidates

Sixteen candidates are taken from the final population. No scalarisation, no TOPSIS:

* if the final non-dominated front has more than 16 points: the extreme point of each objective, plus greedy
  farthest-point samples in the min–max-normalised objective space (k-centre coverage);
* if it has at most 16 (the usual case): the whole front, filled up with the next individuals of the final
  population in NSGA-II's own survival order (Pareto rank, then larger crowding distance).

## 5. Successive halving and learning-capability selection

Every candidate run uses identical conditions:

* same initialisation seed, same data order and augmentation stream (common random numbers);
* AdamW (lr 1e-3, weight decay 5e-5) with a constant learning rate;
* loss with α = β = 0.5;
* batch size 8 (IDRiD 4).

Every epoch records training/validation loss, Dice, IoU, mIoU, sensitivity and the validation HD95.

**Criterion.** The mean validation Dice over the last 5 epochs a run has reached.

**Successive halving** (Jamieson & Talwalkar 2016; reduction factor 2):

| rung | candidates | trained to | promoted |
|---|---:|---:|---|
| 1 | 16 | 5 epochs | the best 8 by the criterion (epochs 1–5) |
| 2 | 8 | 10 epochs | the best 4 by the criterion (epochs 6–10) |
| 3 | 4 | 20 epochs | finalists |

That is 160 candidate-epochs, the budget of 8 candidates × 20 epochs. A promoted run continues from its checkpoint;
with a constant learning rate it is the same run as one trained without interruption. Ties in a promotion are
broken by fewer parameters. A run with a non-finite loss is eliminated.

**Selection among the 4 finalists.** A finalist is eligible only if it passes all gates on its 20-epoch run:

| gate | condition | rationale |
|---|---|---|
| G1 | all losses finite | diverged training is disqualified |
| G2 | `ΔLoss = Loss_val(1) − Loss_val(E) > 0` | the improvement must generalise |
| G3 | **not** (val loss ↑ significantly and train loss ↓ significantly over the 2nd half; one-sided Kendall τ, p < 0.05) | overfitting |
| G4 | val Dice has a significant increasing trend over all epochs (Kendall τ > 0, p < 0.05) | genuine, stable learning (not an erratic curve or a lucky last epoch) |
| G5 | `Dice_val(E)` > Dice of the trivial all-foreground predictor on the validation masks | the model has learned to segment |

Among eligible finalists the highest criterion (mean validation Dice of epochs 16–20) wins; ties → fewer
parameters. If no finalist passes all gates, gates are relaxed in the order G4 → G3 → G2 → G5, and only the gates
that actually had to be relaxed are reported. G1 is never relaxed.

## 6. Loss and loss weights

`L = α·BCE + β·L_soft-mIoU`, with `α + β = 1`, where:

* `L_soft-mIoU = 1 − ½(IoU_fg + IoU_bg)`;
* soft IoU = `(Σpg + 1) / (Σ(p + g − pg) + 1)`, computed per image.

α is chosen from {0.2, 0.4, 0.5, 0.6, 0.8}:

* each value trains the selected architecture with the candidate protocol (20 epochs);
* the criterion is the mean validation Dice over the last 5 epochs (ties → α closest to 0.5);
* the α = 0.5 run is identical to the selected finalist's 20-epoch candidate run and is reused.

Whether this search improves the test result, or only costs time, is measured in study v2 (`qenas_fixed_alpha`,
DESIGN_DECISIONS D49).

This is validation-set model selection; the test set is not involved.

## 7. Final training

* fresh initialisation (same seed);
* AdamW (lr 1e-3, weight decay 5e-5);
* poly schedule with one warm-up epoch (official Mixed-GGNAS schedule);
* 120 epochs, the selected α/β;
* checkpoint every epoch, plus a best-validation-Dice checkpoint.

Early stopping happens only on *overfitting*: no improvement for 20 epochs **and** a significantly rising
validation loss together with a significantly falling training loss. Stagnation alone never stops training. The
best checkpoint is restored and evaluated on train (no augmentation) and validation. No test-time augmentation is
used (D27).

## 8. Test evaluation

The test split is loaded for the first time, and evaluated once, only after these are all fixed:

* the architecture;
* α/β;
* all hyper-parameters;
* the end of final training.

The experiment state records that the test set was used, and re-running never re-evaluates it.

## 9. Metrics of the final architecture

All metrics are computed per image at the network resolution (threshold sigmoid > 0.5) and averaged over the
split.

The metrics of the Mixed-GGNAS paper (Sec. 4.1: Dice, mIoU, parameters, FLOPs):

* **Dice** = Dice similarity coefficient (**DSC**) = 2|P∩G| / (|P|+|G|). Dice and DSC are the same metric, so
  it is reported once;
* **mIoU** = ½(IoU_fg + IoU_bg) (class mean, as in Mixed-GGNAS); the foreground IoU = |P∩G|/|P∪G| is also
  reported;
* **Params (M)** = trainable parameters;
* **FLOPs (G)** = thop operation count for one training batch (8 images, IDRiD 4), the convention of the
  Mixed-GGNAS tables. It reproduces their U-Net values exactly (437.94 G and 547.42 G). FLOPs per image are
  reported as well.

Added metrics:

* **HD95** (boundary): 95th-percentile symmetric Hausdorff distance between prediction and ground-truth
  boundaries, in pixels, defined as max(P95 d(∂P→∂G), P95 d(∂G→∂P)). Chosen over the average Hausdorff distance,
  whose ranking errors are documented (Aydin et al., 2021). Both masks empty → 0; exactly one empty → the image
  diagonal (counted and reported).
* **Sensitivity** (recall, clinical) = |P∩G| / |G|.

Empty ground truth and empty prediction count as Dice = IoU = 1. None of the four datasets has empty ground
truths.

Also recorded:

* the inference time per image (median of 50 forward passes, batch 1);
* every test image's metrics (`test_per_image.csv`) and binary prediction (`test_predictions.npz`), for paired
  statistics and qualitative figures.

## 10. Datasets and splits

| dataset | images | train / validation / test | split unit |
|---|---:|---|---|
| BUSI (benign + malignant) | 645 | 413 / 103 / 129, stratified by class | image (exact duplicates handled) |
| CVC-ClinicDB | 612 | 395 / 96 / 121 | colonoscopy sequence |
| IDRiD, optic disc | 81 | 43 / 11 / 27 (official test split) | image |
| Kvasir-SEG | 1000 | 704 / 176 / 120 (official `val.txt` = test split) | image |

The split is fixed (`data.split_seed = 42`); the run seed changes only the search, initialisation, data order and
augmentation. Results of a run are in `results/<DATASET>/qenas/seed_<seed>/`. DATASETS.md documents the folder
layouts and integrity checks.

## 11. Evaluation study

The comparison with U-Net, Attention U-Net and nnU-Net, the ablations, random search, repeated seeds and external
validation are described in STUDY.md and run by `study.py`. They do not change QENAS. Study v1 (QENAS 2.1) is
documented in `reports/QENAS_Study_v1.pdf`; study v2 tests QENAS 3.0.
