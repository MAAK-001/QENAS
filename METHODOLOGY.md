# QENAS — Methodology

**QENAS (Quick Evolutionary NAS)** searches U-shaped segmentation networks in a mixed search space of manually
designed and DARTS blocks. The search is **training-free**: NSGA-II optimises three objectives computed on the
*initialised* network, without updating a single weight:

* the number of parameters (complexity);
* **SynFlow**, a data-independent measure of signal propagation;
* **ZiCo**, a measure of the gradients that the segmentation loss produces on real training images.

Supervised training is used only after the search, to:

1. compare a small, representative set of final Pareto architectures by their *learning capability* over a
   short budget;
2. choose the loss weights on the validation set;
3. train the selected architecture normally with a validation set. The validation set also decides whether
   flip test-time augmentation is used.

The test set is used exactly once, at the very end. There is one pipeline and one protocol. Nothing in the
method is optional.

```
Dataset ─► dataset-specific train / validation / test split (leakage-free)
        ─► search space (8 genes × 5 block types = 5^8 = 390,625 architectures)
        ─► random initial population (generation 0)
        ─► NSGA-II (pymoo), population 20, 10 generations; every individual evaluated training-free:
               f1 = #trainable parameters                (minimise)
               f2 = 1 / (1 + ln(1 + SynFlow))           (minimise ⇔ maximise SynFlow)
               f3 = 1 / (1 + ZiCo)                       (minimise ⇔ maximise ZiCo)
        ─► final non-dominated front ─► representative subset (≤ 8, diversity-preserving)
        ─► 20-epoch supervised training of every candidate
        ─► learning-capability selection (ΔDice with pathology gates)
        ─► loss-weight (α, β) selection on validation
        ─► ordinary supervised training (100 epochs, best-validation checkpoint) → flip-TTA decision on validation
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

The same `QENASNet` class with the same seeded initialisation is used for SynFlow, ZiCo, candidate training and
final training. The networks the proxies score are therefore bit-identical to the networks that start supervised
training.

## 2. Objectives (training-free)

**f1 — complexity.** `f1(a) = Σ_θ 1[θ trainable] · numel(θ)` (minimise).

**f2 — SynFlow** (Tanaka et al., NeurIPS 2020; zero-cost NAS proxy of Abdelfattah et al., ICLR 2021). The
official procedure is used:

1. initialise (seed); eval mode; bypass normalisation layers (zero-cost-NAS `bn=False`);
2. replace every parameter by its absolute value;
3. forward one all-ones input `1 ∈ R^{1×3×H×W}` in float64; `R = Σ output`;
4. backpropagate; `S = Σ_{w ∈ Conv, ConvTranspose, Linear weights} |w · ∂R/∂w|`;
5. restore the signs.

`S` is exponential in depth (≈ 10^24 … 10^38), so the objective is defined on the log scale:
`f2 = 1 / (1 + ln(1 + S)) ∈ (0, 1]`.

**f3 — ZiCo** (Li et al., ICLR 2023). The official functions (`getgrad`, `caculate_zico`, `getzico` from
github.com/SLDGroup/ZiCo) are used without any change to their logic:

1. initialise (same seed); **train mode** (normalisation layers active);
2. for each of the **two fixed training batches**: zero the gradients, forward, compute the training loss
   `L = 0.5·BCE + 0.5·soft-mIoU` against the ground-truth masks, and backpropagate;
3. collect the gradient of every `Conv2d` / `Linear` weight;
4. for every layer *l*, over the two batches, compute `std(g)` and `mean(|g|)` of each weight gradient. Then
   `ZiCo = Σ_l log( Σ_{weights with std > 0} mean|g| / std(g) )`.

With two batches each ratio is ≥ 1, so ZiCo ≥ 0. The objective is `f3 = 1 / (1 + ZiCo) ∈ (0, 1]`.

The two ZiCo batches are the first two batches of training epoch 0: 8 images each (IDRiD 4), shuffled and
augmented exactly as in training. They are fixed for the whole search and identical for every architecture, as
in the official ZiCo search. ZiCo is the only objective that sees the dataset. It sees only **training** images
and their masks, and no weight is updated, so the search stays training-free. Validation and test images are
never used by the search.

All three objectives are positive and minimised. Pareto dominance is invariant to strictly monotone transforms,
so the ranking equals that of "minimise parameters, maximise SynFlow, maximise ZiCo". No proxy is combined with
another into a single score.

**Do the objectives work together?** A training-free study of 65 architectures on BUSI
(`evidence/objective_agreement/`, DESIGN_DECISIONS D14) measured how the objectives relate:

* SynFlow and ZiCo agree (Kendall τ = +0.37; +0.41 after removing the effect of model size), so they are not
  conflicting quality signals;
* SynFlow trades off against the parameter count (τ = 0.46 with size), which is the intended
  cost-versus-quality trade-off;
* ZiCo is independent of model size (τ = −0.03), i.e. it adds information that size and SynFlow do not
  carry.

## 3. NSGA-II

pymoo's `NSGA2` (non-dominated sorting, crowding distance, binary tournament mating selection,
(μ+λ) rank-and-crowding survival), driven through pymoo's ask-and-tell interface:

* sampling: uniform integers per gene (`IntegerRandomSampling`);
* crossover: `UniformCrossover`, probability 0.9 (each gene from either parent with probability 0.5, i.e. the
  binomial crossover with CR = 0.5 used by Mixed-GGNAS);
* mutation: `ChoiceRandomMutation` (random resetting of categorical genes), per-gene probability 1/8;
* duplicate elimination on;
* **population 20**, generation 0 (random) + **10 evolutionary generations** (as in Mixed-GGNAS). With three
  objectives, 51 % of a 10-individual population is non-dominated, so rank-based selection hardly acts; with 20
  it is 33 % (D16).

Evaluations are cached. The cache key covers the chromosome, the architecture definition, the objective
definitions and the SHA-256 of the exact ZiCo batches. After every generation the whole algorithm state is
checkpointed, including its random generator. Each generation's population is recorded with its Pareto rank,
crowding distance and non-domination flag.

Search progress is measured by the **hypervolume** of every generation's non-dominated set. It is computed in the
objective space min–max normalised over all evaluated architectures, with reference point 1.1 in each objective
(figure `search_progress`).

## 4. Pareto candidates

The final non-dominated front is kept intact (no scalarisation, no TOPSIS). If it has at most *k* = 8 points, all
of them are evaluated. Otherwise the subset is:

* the extreme point of each of the three objectives;
* plus greedy farthest-point samples in the min–max-normalised objective space (k-centre coverage).

## 5. Learning-capability evaluation and selection

Each candidate is trained for *E* = 20 epochs under identical conditions:

* same initialisation seed, same data order and augmentation stream (common random numbers);
* AdamW (lr 1e-3, weight decay 5e-5) with a constant learning rate;
* loss with α = β = 0.5;
* batch size 8 (IDRiD 4).

Every epoch records training/validation loss, Dice, IoU, mIoU, sensitivity and the validation HD95.

Primary criterion: `ΔDice = Dice_val(E) − Dice_val(1)`. A candidate is eligible only if it passes all gates:

| gate | condition | rationale |
|---|---|---|
| G1 | all losses finite | diverged training is disqualified |
| G2 | `ΔLoss = Loss_val(1) − Loss_val(E) > 0` | the improvement must generalise |
| G3 | **not** (val loss ↑ significantly and train loss ↓ significantly over the 2nd half; one-sided Kendall τ, p < 0.05) | overfitting |
| G4 | val Dice has a significant increasing trend over all epochs (Kendall τ > 0, p < 0.05) | genuine, stable learning (not an erratic curve or a lucky last epoch) |
| G5 | `Dice_val(E)` > Dice of the trivial all-foreground predictor on the validation masks | the model has learned to segment |

Among eligible candidates the largest ΔDice wins. Candidates within the measured epoch-to-epoch Dice noise of the
leader (median |ΔDice_t| over the last 5 epochs) count as tied. Ties are broken by:

1. higher final validation Dice;
2. then larger ΔLoss;
3. then fewer parameters.

If no candidate passes all gates, gates are relaxed in the order G4 → G3 → G2 → G5, and only the gates that
actually had to be relaxed are reported. G1 is never relaxed.

## 6. Loss and loss weights

`L = α·BCE + β·L_soft-mIoU`, with `α + β = 1`, where:

* `L_soft-mIoU = 1 − ½(IoU_fg + IoU_bg)`;
* soft IoU = `(Σpg + 1) / (Σ(p + g − pg) + 1)`, computed per image.

α is chosen from {0.2, 0.4, 0.5, 0.6, 0.8}:

* each value trains the selected architecture with the candidate protocol (20 epochs);
* the criterion is the mean validation Dice over the last 5 epochs (ties → α closest to 0.5);
* the α = 0.5 run is identical to the candidate run and is reused.

This is validation-set model selection; the test set is not involved.

## 7. Final training

* fresh initialisation (same seed);
* AdamW (lr 1e-3, weight decay 5e-5);
* poly schedule with one warm-up epoch (official Mixed-GGNAS schedule);
* 100 epochs, the selected α/β;
* checkpoint every epoch, plus a best-validation-Dice checkpoint.

Early stopping happens only on *overfitting*: no improvement for 20 epochs **and** a significantly rising
validation loss together with a significantly falling training loss. Stagnation alone never stops training. The
best checkpoint is restored and evaluated on train (no augmentation) and validation.

**Test-time augmentation decision.** The best checkpoint is also evaluated on validation with horizontal-flip TTA:
the foreground probabilities of the image and of its mirror (flipped back) are averaged. TTA is used for the test
if and only if it increases the validation Dice.

## 8. Test evaluation

The test split is loaded for the first time, and evaluated once, only after these are all fixed:

* the architecture;
* α/β;
* the TTA decision;
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

Empty ground truth and empty prediction count as Dice = IoU = 1. None of the three datasets has empty ground
truths.
