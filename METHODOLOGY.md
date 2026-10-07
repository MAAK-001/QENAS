# QENAS — Methodology

**QENAS (Quick Evolutionary NAS)** searches U-shaped segmentation networks in a mixed search space of
manually designed and DARTS blocks. The search itself is **training-free**: NSGA-II optimises two
objectives that can be computed at initialisation (parameter count and SynFlow). Supervised training is used
only to (i) compare a small, representative set of final Pareto architectures by their *learning capability*
over a short budget, (ii) choose the loss weights on the validation set, and (iii) train the selected
architecture normally. The test set is used exactly once, at the very end.

```
Dataset ─► dataset-specific train / validation / test split (leakage-free)
        ─► search space (8 genes × 5 block types = 5^8 = 390,625 architectures)
        ─► random initial population (generation 0)
        ─► NSGA-II (pymoo) for G generations, each individual evaluated training-free:
               f1 = #trainable parameters            (minimise)
               f2 = 1 / (1 + ln(1 + SynFlow))         (minimise ⇔ maximise SynFlow)
        ─► final non-dominated front ─► representative subset (≤ k, diversity-preserving)
        ─► E-epoch supervised training of every candidate (default E = 20)
        ─► learning-capability selection (ΔDice with pathology gates)
        ─► loss-weight (α, β) selection on validation
        ─► ordinary supervised training (no scale selection / collapse), best-val checkpoint
        ─► validation report ─► single test evaluation
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

Genes 1–4 are the encoder positions (each halves the resolution and doubles the channels), genes 5–8 the
decoder positions (each doubles the resolution and halves the channels). With base width *b* (default 32):

| feature | E0 (stem) | E1 | E2 | E3 | E4 (bottleneck) | D1 | D2 | D3 | D4 |
|---|---|---|---|---|---|---|---|---|---|
| channels | b | 2b | 4b | 8b | 16b | 8b | 4b | 2b | b |
| resolution | H | H/2 | H/4 | H/8 | H/16 | H/8 | H/4 | H/2 | H |

Decoder position *j* receives the encoder feature at its output resolution as skip connection
(D1←E3, D2←E2, D3←E1, D4←E0). A 1×1 convolution maps D4 to one logit per pixel.

*Manual blocks* are wrapped by the official Mixed-GGNAS transitions:
encoder `BN → AvgPool 2×2 → 3×3 grouped conv (C_in→C_out) → block`;
decoder `BN → 2×2 transposed conv (C_in→C_out)`, skip `BN → s + s·σ_s(s)·σ_c(s)` (spatial-channel attention),
`concat → 1×1 conv → block`.

*DARTS blocks* are NAS-Unet cells: two inputs `s0, s1`, 1×1 preprocessing to `c = C_out/4` channels, four
intermediate nodes (each the sum of two operations from the genotype), output = concatenation of the four
nodes. Down cells (encoder) take `s1` = previous feature and `s0` = the feature two levels back (2×
resolution, preprocessed with stride 2); up cells (decoder) take `s1` = previous decoder feature and
`s0` = the encoder skip at the output resolution. Down-sampling/up-sampling operations act only on the cell
inputs, as in NAS-Unet.

**Convolution scale.** Mixed-GGNAS defines three convolution scales per manual block (Table 1 of the paper)
and selects among them with learnable weights trained on the *supervised* training loss (its Eq. 1).
Because that mechanism requires supervised training, it is **not** part of QENAS: every manual block uses one
fixed, configured scale (default 3) in the search, in candidate training and in final training. There is no
scale-selection stage, no scale collapse and no differentiable scale weight anywhere.

The same `QENASNet` class with the same arguments is used for SynFlow, candidate training and final training.
Moreover, all of them start from the **same seeded initialisation**, so the network scored by SynFlow is
bit-identical to the network that starts supervised training.

## 2. Objectives (training-free)

**f1 — complexity.** `f1(a) = Σ_θ 1[θ trainable] · numel(θ)` (minimise).

**f2 — SynFlow** (Tanaka et al., 2020; Abdelfattah et al., 2021), computed with the official procedure:

1. initialise the network (fixed seed); eval mode; bypass normalisation layers (zero-cost-NAS `bn=False`);
2. replace every parameter/buffer by its absolute value (`linearize`);
3. forward one all-ones input `1 ∈ R^{1×3×H×W}` in float64; `R = Σ output`;
4. backpropagate; `S = Σ_{w ∈ Conv, ConvTranspose, Linear weights} |w · ∂R/∂w|`;
5. restore the signs (`nonlinearize`).

`S ≥ 0`, larger is better (more synaptic flow; `S = 0` ⇔ disconnected network). Because `S` is a product of
layer-wise weight magnitudes along paths, it is exponential in depth (observed range ≈ 10^5 … 10^38), so the
objective is defined on the log scale and converted into a positive minimisation objective:

```
raw SynFlow      S ∈ [0, ∞)                      (maximise)
log-SynFlow      L = ln(1 + S) ∈ [0, ∞)          (maximise; log1p keeps L ≥ 0, finite at S = 0)
objective        f2 = 1 / (1 + L) ∈ (0, 1]        (minimise; strictly decreasing in S)
```

NSGA-II's non-dominated sorting is invariant to strictly monotone transformations, so the Pareto ranking is
exactly the ranking under "minimise parameters, maximise SynFlow". No sign flip and no negative value is
involved. Non-finite or negative values raise an error.

No third objective is used. No proxy is combined with another proxy.

## 3. NSGA-II

pymoo's `NSGA2` (non-dominated sorting, crowding distance, binary tournament mating selection,
(μ+λ) rank-and-crowding survival), driven through pymoo's ask-and-tell interface:

* sampling: uniform integers per gene (`IntegerRandomSampling`);
* crossover: `UniformCrossover`, probability 0.9 (each gene from either parent with probability 0.5 —
  the binomial crossover with CR = 0.5 used by Mixed-GGNAS);
* mutation: `ChoiceRandomMutation` (random resetting of categorical genes), per-gene probability 1/8;
* duplicate elimination on;
* population 10, generation 0 (random) + 10 evolutionary generations (Mixed-GGNAS: 10 × 10).

Evaluations are cached (key: chromosome + architecture signature + SynFlow configuration + torch version).
After every generation the whole algorithm state (including its random generator) is checkpointed and the
population is recorded with its Pareto rank, crowding distance and non-domination flag.

## 4. Pareto candidates

The final non-dominated front is kept intact (no scalarisation, no TOPSIS). If it has at most *k* (default 8)
points all are evaluated; otherwise the subset consists of both extreme points plus greedy farthest-point
samples in the min–max-normalised objective space (k-centre coverage).

## 5. Learning-capability evaluation and selection

Each candidate is trained for *E* = 20 epochs under identical conditions: same initialisation seed, same data
order and augmentation stream (common random numbers), AdamW (lr 1e-3, weight decay 5e-5), constant learning
rate, loss with α = β = 0.5, batch size 8 (IDRID 4). Every epoch records training/validation loss, Dice, IoU
and the learning rate.

Primary criterion: `ΔDice = Dice_val(E) − Dice_val(1)`. A candidate is eligible only if it passes all gates:

| gate | condition | rationale |
|---|---|---|
| G1 | all losses finite | diverged training is disqualified |
| G2 | `ΔLoss = Loss_val(1) − Loss_val(E) > 0` | the improvement must generalise |
| G3 | **not** (val loss ↑ significantly and train loss ↓ significantly over the 2nd half; one-sided Kendall τ, p < 0.05) | overfitting |
| G4 | val Dice has a significant increasing trend over all epochs (Kendall τ > 0, p < 0.05) | genuine, stable learning (not an erratic curve or a lucky last epoch) |
| G5 | `Dice_val(E)` > Dice of the trivial all-foreground predictor on the validation masks | the model has learned to segment |

Among eligible candidates the largest ΔDice wins; candidates within the measured epoch-to-epoch Dice noise
of the leader (median |ΔDice_t| over the last 5 epochs) are considered tied and the tie is broken by higher
final validation Dice, then larger ΔLoss, then fewer parameters. If no candidate passes all gates, gates are
relaxed in the order G4 → G3 → G2 → G5 and only the gates that were actually necessary to relax are
reported; G1 is never relaxed.

## 6. Loss and loss weights

`L = α·BCE + β·L_soft-mIoU`, `α + β = 1`, where
`L_soft-mIoU = 1 − ½(IoU_fg + IoU_bg)` with soft IoU `(Σpg + 1) / (Σ(p + g − pg) + 1)` per image.

α is chosen from {0.2, 0.4, 0.5, 0.6, 0.8} by training the selected architecture with the candidate protocol
(20 epochs) and maximising the mean validation Dice over the last 5 epochs (ties → α closest to 0.5). The run
with α = 0.5 is identical to the candidate run and is reused. This is validation-set model selection; the
test set is not involved.

## 7. Final training

Fresh initialisation (same seed), AdamW (lr 1e-3, weight decay 5e-5), poly schedule with one warm-up epoch
(official Mixed-GGNAS schedule), default 100 epochs, the selected α/β, checkpoint every epoch, best-validation-
Dice checkpoint. Early stopping happens only on *overfitting* (no improvement for 20 epochs **and** a
significantly rising validation loss with a significantly falling training loss); stagnation alone never
stops training. The best checkpoint is restored and evaluated on train (no augmentation) and validation.

## 8. Test evaluation

Only after the architecture, α/β and all hyper-parameters are fixed and final training has finished, the test
split is loaded for the first time and evaluated once with the best-validation checkpoint. The experiment
state records that the test set was used; re-running never re-evaluates it.

## 9. Metrics

Per image (threshold sigmoid > 0.5), averaged over the dataset: Dice = 2|P∩G|/(|P|+|G|),
IoU = |P∩G|/|P∪G| (foreground), mIoU = ½(IoU_fg + IoU_bg) (class mean, as in Mixed-GGNAS),
loss = mean per-image loss. Empty ground truth and empty prediction count as Dice = IoU = 1.
