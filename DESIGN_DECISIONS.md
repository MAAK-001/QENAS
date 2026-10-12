# QENAS — Design decisions

Every decision that is not dictated by an official implementation, and every adaptation of official code, is
listed as **Decision / Reason (evidence) / Alternative considered / Why the alternative was rejected**. Rule
applied throughout: a choice is adopted when it is proved by a reference implementation, published evidence or a
measurement in this project. A merely plausible change is not adopted.

**Sources consulted:**

* Mixed-GGNAS: the paper (Hu et al., *ESWA* 289, 2025, 128338) and the official code (github.com/Hmxki/Mixed-GGNAS);
* NAS-Unet (github.com/tianbaochou/NasUnet);
* Synaptic-Flow (github.com/ganguli-lab/Synaptic-Flow) and the zero-cost-NAS reference (Abdelfattah et al.,
  ICLR 2021);
* ZiCo: the paper (Li et al., ICLR 2023) and the official code (github.com/SLDGroup/ZiCo);
* ZiCo-BC (Bhardwaj et al., ICCVW 2023);
* NAS-Bench-Suite-Zero (Krishnakumar et al., NeurIPS 2022);
* Metrics Reloaded (Maier-Hein et al., *Nat. Methods* 2024);
* Aydin et al. (*Eur. Radiol. Exp.* 2021);
* thop 0.1.1;
* pymoo 0.6.2.

---

## Pipeline

### D1 — One pipeline, one protocol, no options
* **Decision:** QENAS has exactly one sequence of stages (split → search → pareto → candidates → selection →
  loss_weights → final → test → report) and one configuration (`qenas/config.py`). The command line accepts
  only run settings that cannot change the method:
  * dataset, data root, device, data-loader workers, seed;
  * `--no-resume` and `--stop-after`.
* **Removed alternative paths of the previous version:**
  * optional train+validation refit (`--refit`) and re-use of another experiment's stages (`--reuse-from`);
  * forced TTA on/off;
  * YAML configuration overrides and a separate smoke configuration;
  * choice of optimiser, scheduler, convolution scale, DARTS genotype, IDRiD target, BUSI `normal` images,
    duplicate handling and normalisation;
  * AMP/gradient-clipping overrides;
  * hyper-parameter flags;
  * disabling the loss-weight search or the early stopping.
* **Reason:** an experiment is only interpretable if every reported number comes from the same, single protocol.
  The removed paths were either off by default (refit, reuse) or never-used variants. Removing them also removes
  ~25 % of the code that could go wrong.
* **What remains adjustable, and why:**
  * the **seed**, because independent repetitions are needed to report variability;
  * device, workers and data root, which do not affect results (excluded from the configuration fingerprint).
* **The evaluation study** (baselines, ablations, random search, repeated seeds; STUDY.md) does not add options to
  the method: its conditions are defined in `qenas/methods.py` and run by `study.py` (D44); `main.py` runs only
  QENAS.
* **How the end-to-end check is done now:** the unit tests run the real pipeline on a small synthetic dataset
  (`tests/test_pipeline.py`) instead of a special smoke mode.

### D2 — Package layout `qenas/…`
* **Reason:** a top-level `datasets` package would shadow the widely installed HuggingFace `datasets` package.
  One importable package keeps the modules (datasets / models / search / training / visualization / utils)
  separate without name clashes.

---

## Search space and network

### D3 — Manual blocks are the official Mixed-GGNAS implementations
* **Reason:** `qenas/models/blocks.py` reproduces `raw_cells.py` of the official code:
  * layer order, `groups=16`;
  * dense growth 32 × 3 layers;
  * Inception strip branches;
  * ConvNeXt inverted bottleneck with BatchNorm and post-residual GELU;
  * residual paths.
* **Alternative:** textbook ResNet/DenseNet/Inception/ConvNeXt blocks. **Rejected:** they would not be the
  Mixed-GGNAS search space.
* Consequence: `model.base_channels` must be a multiple of 16 (base width 32, as in the official code).

### D4 — Convolution scale: one fixed scale, no selection, no collapse
* **Reason:** in Mixed-GGNAS the scale weights are learned from the *supervised* training loss (paper Eq. 1;
  the official code trains `cells_weight` with labels from epoch 60 on). That needs supervised training, so it
  cannot be part of a training-free search, and QENAS has no scale-selection/collapse stage.
* **Decision:** every manual block uses scale 3, the first variant of Mixed-GGNAS Table 1, identically in the
  search, in candidate training and in final training:
  * 3×3 convolutions for Residual/Dense (the canonical He/Huang blocks);
  * strips (3, 5, 7) for Inception;
  * 5×5 depthwise convolution for ConvNeXt.
* **Alternatives:**
  (a) learn the scale weights training-free by maximising a proxy — invented machinery with no published basis;
  (b) keep all three branches with equal weights — triples the cost, and Mixed-GGNAS never produces such an
  architecture;
  (c) put the scale in the chromosome — 15^8 instead of the specified 5^8 architectures.
  **Rejected** for those reasons.

### D5 — DARTS block = the official, dataset-matched DARTS genotype
* **Reason:** in Mixed-GGNAS the DARTS block is a NAS-Unet cell whose operations were found by a supervised
  DARTS search; the GA only decides *where* it is used. Running that DARTS search inside QENAS would violate the
  training-free search. The official repository publishes the derived genotypes (`darts_cell_busi`,
  `darts_cell_cvc`, `darts_cell_idrid`), and QENAS uses the one of the dataset verbatim. The primitive
  operations are the NAS-Unet operations vendored by Mixed-GGNAS (GroupNorm with 2 channels/group, cweight,
  dilated/depthwise/transposed convolutions, pooling).
* **Caveat (reported honestly):** the genotypes were derived by the Mixed-GGNAS authors on their own split of
  the same datasets, whose membership is unknown. They are a fixed structural prior (16 discrete operation
  choices), not access to our test labels, but a strict reader may see weak information transfer. The previous
  version offered a dataset-independent genotype as an option. Under D1 it was removed; the caveat stays.
* **Kvasir-SEG** has no published Mixed-GGNAS genotype. It uses `darts_cell_cvc`, the genotype derived on
  CVC-ClinicDB: the same imaging modality (colonoscopy polyps) and the only polyp dataset of the Mixed-GGNAS
  paper. Caveat for the external validation (RQ4): this structural prior was derived (by Mixed-GGNAS, on their
  split) on CVC-ClinicDB, the external test dataset; no CVC-ClinicDB image or weight enters the Kvasir-SEG models.

### D6 — DARTS cell wiring follows NAS-Unet; resolution handling explicit
* **Reason:** NAS-Unet feeds a down cell with (two-levels-back feature at 2× resolution, previous feature) and an
  up cell with (encoder skip at the output resolution, previous decoder feature). The official Mixed-GGNAS code
  instead passes the previous-previous decoder output to up cells (no encoder skip) and patches the resulting
  size mismatches with nearest-neighbour interpolation.
  * QENAS uses the internally consistent NAS-Unet wiring.
  * At the first encoder position both inputs are the stem output, so `preprocess0` uses stride 1 there.
  * Genotypes are validated: an operation that would break spatial consistency raises an error.
* **Alternative:** the official interpolation hack. **Rejected:** it hides shape errors and starves DARTS
  decoder cells of skip connections.

### D7 — U-shaped framework: standard skip alignment; no ViT branch, no deep supervision
* **Reason:**
  * Skips connect the encoder feature at the decoder position's *output* resolution (D1←E3 … D4←stem), the
    standard U-Net alignment.
  * The ViT fusion module's hyper-parameters are tuned by a supervised GA in Mixed-GGNAS, and its call is
    commented out in the official `CellModel.forward`.
  * The multi-scale (deep supervision + Smooth-L1) loss is replaced by the specified BCE + soft-mIoU loss.
* **Alternative:** the official skip shift (bottleneck used twice, full-resolution stem feature never used).
  **Rejected:** it loses full-resolution detail for no documented reason.

### D8 — Spatial-channel attention on the skip: `s + s·σ_s(s)·σ_c(s)`
* **Reason:** the official decoder computes `en + satt(en)·catt(en)`, where both terms already contain `en`,
  i.e. `en + en²·σ_s·σ_c`. That quadratic term is not the "attention-enhanced feature" the paper describes.
  Measured on BUSI (256×256, same initialisation), it inflates SynFlow by 10^2–10^5 unequally across encoder
  choices (all-Residual 2.0e24 → 3.9e26, all-ConvNeXt 8.5e37 → 1.8e43).
* **Alternative:** the quadratic form. **Rejected:** it distorts the proxy comparison between architectures.

### D9 — Binary output: one logit per pixel
* **Reason:** binary segmentation with BCE + soft-mIoU. The official two-channel softmax is equivalent but
  redundant.

---

## Objectives and search

### D10 — Three objectives: parameters, SynFlow and ZiCo (third objective)
* **Decision:** f1 = #parameters (minimise), f2 = SynFlow (maximise), f3 = ZiCo (maximise). All three are
  computed at initialisation; no weight is updated.
* **Why a third objective at all.** SynFlow is data-independent: it never sees an image or a mask, so nothing in
  the previous search was specific to segmentation or to the dataset. Published evidence also shows SynFlow can
  be uninformative for pixel-level prediction. On TransNAS-Bench-101 its Kendall τ with the trained result is
  0.00 for both autoencoding and surface-normal prediction (ZiCo paper, Table 5).
* **Why ZiCo** (research of the literature):
  1. It is computed from the gradients of the *training loss* on real training images and their *masks*, so it
     brings in both the segmentation task and the particular dataset, while staying zero-cost (two
     forward/backward passes, no update).
  2. It is the first proxy shown to be consistently better than the parameter count, a strong baseline that most
     proxies fail to beat (ZiCo paper; NAS-Bench-Suite-Zero finds #params/FLOPs among the strongest baselines).
  3. It has been used **inside NSGA-II on a segmentation task**: a ZiCo-guided NSGA-II macro search on Cityscapes
     found a network with 80.7 % mIoU, vs 79.7 % for the hand-designed FFNet at the same latency (ZiCo-BC,
     Table 1).
  4. Its theory (gradient mean/variance → convergence speed and generalisation) matches what QENAS selects for
     after the search: the candidates' learning capability.
* **Alternatives considered:**
  * NASWOT and SWAP (label-free activation-pattern proxies; no segmentation-specific evidence);
  * MeCo (data- and label-independent by design);
  * label–gradient alignment (needs some training to become informative);
  * effective-receptive-field proxies (published for detection only);
  * the previous version's own SA-SWAP/SA-JacCov (never validated).
  **Rejected** because none has comparable evidence on dense prediction together with use of the labels.
* **Honest limit:** no zero-cost proxy is *guaranteed* to rank segmentation networks correctly. Every run reports
  the Spearman ρ between each proxy and the candidates' validation Dice (`zico_vs_val_dice`,
  `synflow_vs_val_dice`).

### D11 — SynFlow: official procedure, normalisation layers bypassed
* **Procedure:** Synaptic-Flow `SynFlow.score` + zero-cost-NAS `compute_synflow_per_weight`:
  * eval mode;
  * `linearize` (absolute value of every state-dict tensor);
  * a single all-ones input of the data shape, float64;
  * `R = sum(output)`;
  * score `|θ·∂R/∂θ|` over the Conv/Linear weights;
  * signs restored.
* **QENAS modifications:**
  (i) `ConvTranspose2d` weights are included (the decoder up-samples with them; excluding them makes SynFlow
  blind to the decoder);
  (ii) **all normalisation layers are bypassed**. zero-cost-NAS evaluates a copy without BatchNorm
  (`bn=False`); Synaptic-Flow evaluates BN in eval mode, which is the identity at initialisation. The DARTS
  operations use GroupNorm, which normalises with per-sample statistics and cannot be linearised. Measured on
  BUSI: all-DARTS ln(1+S) = 12.8 with GroupNorm active vs 78.3 without, while BN-only networks are unchanged
  (unit-tested).
* **Input / initialisation:** deterministic all-ones tensor `[1, 3, H, W]` at the training resolution. PyTorch
  default initialisation under the global seed, inside `torch.random.fork_rng` (the global random stream is
  untouched).

### D12 — ZiCo: the official code, unchanged; how it is integrated
* **Official logic kept exactly:** `getgrad`, `caculate_zico` and `getzico` from
  `SLDGroup/ZiCo/ZeroShotProxy/compute_zico.py` are reproduced line by line in `qenas/search/objectives.py`,
  including the unused variables:
  * network in **train mode**;
  * for each batch: `zero_grad` → forward → loss → backward → store the gradients of every `nn.Conv2d` and
    `nn.Linear` weight;
  * per layer: `np.std` and the mean absolute value over the batches; `log(Σ mean|g|/std)` over the weights
    with non-zero std; layers whose sum is 0 are skipped;
  * the sum over layers.
* `nn.ConvTranspose2d` is not a subclass of `nn.Conv2d`, so — exactly as in the official code — the decoder's
  transposed convolutions do not contribute (tested).
* **The only code differences, none of which changes a value:**
  1. the device is a parameter (the official code calls `.cuda()`);
  2. gradients are copied out of the parameter (`.copy()`), because on the CPU `.numpy()` would alias the
     gradient tensor (on CUDA the official `.cpu()` already copies).
  A unit test checks the result against an independent implementation of the definition.
* **Integration choices (outside ZiCo's logic), each following the official search
  (`evolution_search.py`):**
  * *Number of batches:* 2 (official `--maxbatch` default). With two batches every ratio is
    (|g1|+|g2|)/|g1−g2| ≥ 1, so ZiCo ≥ 0.
  * *Which batches:* the first two batches of training epoch 0, shuffled and augmented exactly as in training.
    They are taken once and reused for every architecture (official: `trainbatches` are built once from the
    shuffled, augmented training loader before the search). The batch size is the training batch size (8,
    IDRiD 4), as the official search uses its training batch size.
  * *Loss:* the QENAS training loss with the neutral weighting α = β = 0.5 — the loss the candidates are trained
    with. The official search uses its task's training loss (CrossEntropy), and ZiCo-BC used "the same loss as
    the one used to train" the segmentation network.
  * *Network:* the exact QENASNet that will be trained, with the same seeded initialisation (D15). The official
    search builds its PlainNet models with `no_reslink=True`, a switch of the Zen-NAS PlainNet search space.
    QENAS blocks have no such switch, and removing their residual paths would score a different architecture
    than the one trained, so the real network is scored.
* **Data used by the search:** only these two training batches. The search therefore remains **training-free**
  (no weight is ever updated) but is no longer data-free. Validation and test images are never used.
* **Not applied: ZiCo-BC's bias correction.** It changes the ZiCo score (it subtracts a depth/width penalty with
  a tuned hyper-parameter). The bias it targets — larger scores for networks with more layers — is measured and
  reported in D14.

### D13 — Positive, minimised objectives `f2 = 1/(1+ln(1+S))`, `f3 = 1/(1+ZiCo)`
* **Reason:** pymoo minimises. The requirement is a positive objective with the correct direction and no blind
  sign flip.
  * `S ≥ 0` spans ~10^5–10^38 (exponential in depth), so its natural scale is `ln(1+S) ≥ 0`.
  * ZiCo is already a sum of logarithms and ≥ 0.
  * `1/(1+x)` maps a non-negative "larger is better" score to (0, 1]: strictly decreasing, positive, and 1
    only for a zero score.
  * Pareto dominance is invariant to monotone transforms, so the ranking equals "maximise SynFlow, maximise
    ZiCo".
* **Alternatives:**
  * `−S` / `−ZiCo`: negative values;
  * `1/S`: its 1e-5…1e-38 range collapses crowding-distance normalisation onto one point and is undefined at
    S = 0;
  * `C − x`: needs an arbitrary bound C.
  **Rejected.**

### D14 — The three objectives work together (measured)
* **Question:** do the objectives conflict in a way that harms the search? Only params vs a quality proxy should
  be a trade-off; two quality proxies working against each other would split the front.
* **Evidence:** `results/analysis/T6_objective_agreement`, `T7_block_effects` (study, `study.py analyse`). They are
  computed from the random-search archives: 220 uniformly random architectures per dataset and seed, all four
  datasets, 3 seeds — an unbiased sample, unlike the NSGA-II archive. The quantities are:
  * the Kendall τ between every pair of minimised objectives (τ > 0: they improve together; τ < 0: trade-off);
  * the τ between SynFlow and ZiCo after removing the effect of model size (rank residuals);
  * the least-squares effect of each block type on each objective.
* **Decision rule:** an objective is replaced only if two *quality* objectives are in significant conflict
  (τ(f2, f3) < 0). A pilot measurement on 65 BUSI architectures, taken when ZiCo was added, found harmony
  (τ(f2, f3) = +0.37; +0.41 after removing size) and ZiCo independent of size. The study re-measures this on all
  datasets; the decision is revisited if T6 contradicts it. **Numbers: to be filled in from T6/T7 after the
  study.**
* **Known bias (reported, not corrected — D12):** ZiCo grows with the number of layers (ZiCo-BC), so it is
  expected to reward the layer-rich DARTS and Inception blocks (T7). Whether such architectures segment better
  is decided by the 20-epoch learning-capability stage, not by the proxies. T9 reports both proxies against the
  trained validation Dice.

### D15 — The scored network is the trained network
* **Reason:** SynFlow, ZiCo, candidate training and final training all build the network with the same class and
  the same seeded initialisation. The parameters the proxies measured are exactly the starting point of
  supervised training, which removes one source of search/train mismatch.

### D16 — NSGA-II configuration: population 20, 10 generations
* **Decision:**
  * pymoo `NSGA2` with `IntegerRandomSampling`, `UniformCrossover(prob=0.9)` and `ChoiceRandomMutation`
    (per-gene p = 1/8, every offspring);
  * `eliminate_duplicates=True`;
  * **population 20**, generation 0 + **10 generations** (≤ 220 evaluations);
  * ask/tell, so every evaluation is cached, logged and checkpointed.
* **Operators:** genes are *nominal* (block types have no order), so uniform crossover (= Mixed-GGNAS' binomial
  crossover, CR = 0.5) and random-resetting mutation fit; p = 1/L is the standard mutation rate. SBX/polynomial
  mutation assume an ordinal distance between block types and were rejected.
* **Population (measured):** NSGA-II selects by non-domination rank first. Once most of the population is rank 0,
  survival and mating are decided by crowding distance alone and the search loses its pressure towards better
  objectives. Adding an objective makes more points non-dominated. The pilot measurement (D14) found 51 % of a
  random population of 10 non-dominated with three objectives, but 33 % with 20 — below the 44 % the
  two-objective search had with 10. Doubling the population is the smallest change that restores the selection
  pressure. **Evidence of the study:** `results/analysis/T8_nondominated_share` (all datasets, 3 seeds; to be
  filled in).
* **Generations: 10**, unchanged from Mixed-GGNAS and the previous version (≤ 220 evaluations). No evidence
  supports a different number, so none is adopted. Convergence is visible in every run's hypervolume curve
  (D17), and NSGA-II is compared with random search at the same budget (`T5_search_vs_random`).
* **Cost:** measured per run in `T13_time` (the search stage). On a GPU the search takes minutes; on a CPU, hours.

### D17 — Search progress measured by hypervolume
* **Decision:** after the search, the hypervolume of every generation's non-dominated set is computed. It uses
  the objective space min–max normalised over all evaluated architectures and reference point 1.1 per objective
  (pymoo `HV`). It is reported in `nsga2_generations.*`, in `search_summary.json` and in the figure
  `search_progress`, together with the population's median parameters, SynFlow and ZiCo per generation.
* **Reason:** hypervolume is the standard Pareto-compliant quality indicator of a multi-objective population. It
  shows directly whether NSGA-II improves the front over the random initial population.

### D18 — Representative Pareto subset
* **Decision:** all front points if |front| ≤ k = 8; otherwise the extreme point of each of the three
  objectives, plus greedy farthest-point sampling in the min–max-normalised objective space.
* **Reason:** the subset spans the whole trade-off surface and covers it evenly (k-centre 2-approximation),
  deterministically, without preference weights.
* **Alternative:** top-k by crowding distance. **Rejected:** crowding distance is infinite for extremes and only
  local, so it can pick clustered points.

### D19 — Evaluation cache keyed by everything the objectives depend on
* **Decision:** the cache key hashes:
  * the chromosome;
  * the architecture definition (channels, scale, genotype);
  * the objective definitions and version;
  * the input size and initialisation seed;
  * the **SHA-256 of the exact ZiCo batches** (images and masks), which covers dataset, split, augmentation,
    normalisation and seed;
  * the PyTorch version.
* **Reason:** with ZiCo the objectives depend on data. A cache keyed only by the architecture could silently
  reuse a value computed on other data. Entries of the previous (two-objective) version are never matched.

---

## Supervised stages

### D20 — Candidate training protocol (20 epochs)
* **Decision:** each candidate is trained under identical conditions (common random numbers):
  * AdamW, lr 1e-3, weight decay 5e-5 (paper §4.2); batch 8 (IDRiD 4);
  * **constant** learning rate; α = β = 0.5; no early stopping;
  * identical initialisation seed, data order and augmentation stream for all candidates.
* **Reason:** ΔDice must reflect the architecture, not a learning-rate annealing artefact or random luck.
* **Alternative:** the final poly schedule truncated to 20 epochs. **Rejected:** annealing to zero flattens all
  curves and compresses the ΔDice differences.

### D21 — Learning-capability criterion (gates + ΔDice)
* **Decision:** see METHODOLOGY §5:
  * five falsifiable gates: finite; generalising loss decrease; no overfitting trend; significant Dice learning
    trend; beats the trivial predictor;
  * then maximal ΔDice, with a data-derived noise tolerance;
  * lexicographic tie-breaks: final Dice → ΔLoss → fewer parameters.
* **Reason:** ΔDice is the specified primary criterion of QENAS. The gates prevent pathological winners
  (unstable, overfitting, diverged or non-learning). Thresholds are standard significance levels or estimated
  from the curves, with no hand-tuned weights.
* **Alternatives:** a weighted sum (arbitrary weights); highest final Dice (not the QENAS criterion); area under
  the curve (conflates level and improvement); Pareto ranking over the criteria (degenerates, because the
  max-ΔDice candidate is always non-dominated). **Rejected.**
* **Known property (disclosed):** ΔDice rewards improvement, not level. A candidate that starts high and ends
  higher can lose to one that starts low and improves more. The noise-tolerant tie-break on final Dice and gate
  G5 limit, but do not remove, this property. It is kept because learning capability is the defining selection
  idea of QENAS.

### D22 — Loss: BCE + soft mIoU (two-class mean), per image, smoothing 1
* **Reason:** "mIoU" in the Mixed-GGNAS evaluation is the class-mean IoU (its mIoU exceeds its Dice, the
  signature of including the background). The soft version over foreground and background is bounded in [0, 1],
  well defined for empty masks with Laplace smoothing 1, and computed per image to match the per-image
  evaluation.

### D23 — Loss-weight selection
* **Decision:**
  * grid α ∈ {0.2, 0.4, 0.5, 0.6, 0.8}, β = 1 − α;
  * each value is a 20-epoch run of the selected architecture with the candidate protocol;
  * criterion = mean validation Dice of the last 5 epochs; ties → α closest to 0.5;
  * the α = 0.5 run is bit-identical to the candidate run and is always reused (verified: CPU training is
    deterministic).
* **Reason:** deterministic, cheap, symmetric around the neutral weighting, and includes the official 0.6/0.4.
* **Disclosure:** the same validation set is used for candidate selection, α selection, checkpoint selection and
  the TTA decision. This is validation-set model selection; the test set is not touched.

### D24 — Final optimiser and schedule
* **Decision:** AdamW (lr 1e-3, wd 5e-5) with the official Mixed-GGNAS schedule:
  * one epoch of linear warm-up from 1e-3·lr;
  * then poly decay `(1 − t/T)^0.9`, stepped per iteration;
  * 100 epochs.
* **Reason:** warm-up stabilises the first AdamW steps of a network trained from scratch; the decay lets the
  final epochs converge. This is the DeepLab policy used by the official code.

### D25 — Early stopping only on overfitting
* **Decision:** stop only if all three hold:
  1. ≥ 30 epochs;
  2. no validation-Dice improvement for 20 epochs;
  3. a significant increase of the validation loss together with a significant decrease of the training loss
     over those 20 epochs (one-sided Kendall τ, p < 0.05).
* **Reason:** stagnation alone never stops training; the classic overfitting signature does. The best
  checkpoint is restored before testing in every case.

### D26 — AMP on CUDA only; no gradient clipping
* **Decision:** AMP (fp16 + loss scaling) is used on CUDA and never on the CPU; losses and metrics are always
  computed in float32/float64. No gradient clipping.
* **Reason:** no instability was observed without clipping. AMP only changes speed and memory, not the protocol.

### D27 — Horizontal-flip test-time augmentation, decided on validation
* **Decision:** the best checkpoint is evaluated on validation with and without flip TTA (average of the
  foreground probabilities of the image and of its mirror, flipped back). TTA is used for the single test
  evaluation if and only if it increases the validation Dice. There is no forced on/off (D1).
* **Evidence:** `results/analysis/T11_tta` — for every run of every method, whether TTA was chosen and the
  validation Dice with and without it (to be filled in). "Always on" would be adopted only if TTA improved the
  validation Dice in every run; a per-model decision on validation is the rule that never uses test data.
  Horizontal flips are in the training augmentation for every dataset, so mirrored inputs are in-distribution.

### D28 — No refit on train + validation
* **Decision:** the final model is the best-validation checkpoint of the run on the training split.
* **Reason:** a model refitted on train + validation can no longer be checked on any held-out data before the
  single test, and its stopping epoch would have to be transferred from another run. No measurement supports a
  gain that would justify that loss of control, so the refit is not part of the method. (An earlier exploratory
  measurement on BUSI found the validation Dice nearly flat between 310 and 413 training images; it is not
  repeated in the study.)

### D29 — Final training length 100 epochs
* **Evidence:** `results/analysis/T12_training` — best epoch, epochs run and early-stopping frequency of every
  final training (to be filled in). The schedule would only be lengthened if the best epochs cluster at the end
  of the budget. A longer run could not silently overfit anyway: the best checkpoint is restored, and D25 stops
  on overfitting.

---

## Evaluation of the final architecture

### D30 — Reported metrics
* **Decision:** the final test reports:
  1. **all metrics of the Mixed-GGNAS paper** (Sec. 4.1: "Dice coefficient (%), Mean intersection over union
     (%), as well as model parameters (M) and floating point operations (G)"):
     * **Dice**;
     * **mIoU** (class mean);
     * **Params (M)**;
     * **FLOPs (G)**;
  2. **DSC** — this *is* the Dice coefficient, already in (1), so it is not reported twice (the table column is
     named "Dice / DSC");
  3. **HD95** as the boundary metric (D31);
  4. **Sensitivity** (recall) as the clinical metric.
* The foreground IoU, the test loss and the per-image FLOPs are recorded too. Per-epoch validation histories also
  contain sensitivity and HD95.
* **Computation:**
  * per image at the network resolution (threshold 0.5), averaged over the split;
  * empty ground truth + empty prediction → Dice = IoU = 1;
  * sensitivity is undefined for an empty ground truth, so such images are excluded from its mean (none of the
    datasets has one).

### D31 — HD95 instead of the average Hausdorff distance
* **Decision:** HD95 = max(P95 of d(∂P → ∂G), P95 of d(∂G → ∂P)), the definition of the Metrics Reloaded
  reference implementation.
  * Boundary = mask minus its 4-connected erosion; exact Euclidean distance transform; pixels of the evaluated
    resolution.
  * Both masks empty → 0.
  * Exactly one empty (a missed object or a hallucination) → the image diagonal, the largest possible distance.
    Such images are counted (`hd95_empty_pred`).
* **Reason:**
  * Metrics Reloaded recommends a percentile Hausdorff distance as the boundary metric, because the maximum is
    dominated by single outlier pixels.
  * The average Hausdorff distance has documented *ranking errors* (Aydin et al., 2021: it misranked 179 of 200
    segmentations in their analysis, vs 52 for their balanced variant).
  * Lesion and polyp segmentation needs a metric that is sensitive to far-off boundary errors but robust to
    single pixels.
* **Alternative:** the pooled variant (percentile of both directed sets together, MedPy). **Rejected** in favour
  of the symmetric-maximum definition, which matches the Hausdorff definition and the Metrics Reloaded
  implementation.

### D32 — FLOPs in the Mixed-GGNAS convention (verified)
* **Decision:** FLOPs = the `thop.profile` operation count (thop 0.1.1 rules, reimplemented in
  `models/summary.py`) for **one training batch** — 8 images, IDRiD 4 — in G. The per-image value is also
  reported.
* **Evidence:** the Mixed-GGNAS tables report 437.94 G for the standard U-Net (31.04 M) at 256×256 and 547.42 G
  on IDRiD. That is not the per-image count (54.74 G). thop's count for a batch of 8 at 256×256 is exactly
  437.94 G, and for a batch of 4 at 512×320 it is 547.42 G (unit test `tests/test_flops.py`). Using the same
  convention makes QENAS's FLOPs directly comparable with the Mixed-GGNAS tables.
* thop counts a transposed convolution per *output* position (C_in × k per output element). That is its
  convention, kept for comparability.

---

## Data

### D33 — BUSI: 647 benign + malignant images, conflicting duplicate removed → 645
* **Reason:** Mixed-GGNAS uses the 647 lesion images (Table 3); `normal` images have empty masks and are not
  used. Multiple lesion masks of one image are merged by union. Pixel hashing found one image in both
  `benign (433)` and `malignant (145)` with *different* masks (IoU 0.77): the ground truth is ambiguous, so both
  copies are excluded (identical-mask duplicates are kept once).
* **Split:** stratified by class, 20 % test, then 20 % of the remainder for validation → 413 / 103 / 129. BUSI has
  no patient identifiers, so the split is image-level (a limitation; exact duplicates are handled).

### D34 — IDRiD: optic-disc target, official split
* **Reason:** the IDRiD segmentation sub-challenge provides five binary targets. Mixed-GGNAS reports Dice 96.6 %
  on "IDRID", which is only attainable for the optic disc (lesion Dice values are far lower), so the target is
  the **optic disc**.
* The official 54/27 split is kept; 11 of the 54 training images form the validation set (43 / 11 / 27). There is
  no patient metadata, so the validation split is image-level.
* **Resolution:** 320 × 512 (H × W), as in the paper (512 × 320), close to the native 2848 × 4288 aspect;
  batch size 4.

### D35 — CVC-ClinicDB: sequence-level split
* **Reason:** the 612 frames come from 29 colonoscopy sequences (metadata.csv). Consecutive frames are
  near-duplicates, so a random frame split leaks. Whole sequences are assigned to splits → 395 / 96 / 121
  frames (18 / 5 / 6 sequences). This is stricter (and gives lower scores) than the random frame split of the
  Mixed-GGNAS paper.
* **Evidence that a random frame split inflates the scores** (study RQ5, `results/analysis/T16_cvc_leakage`): QENAS
  and U-Net are also run with a frame-level split (392 / 98 / 122, `data.split_unit = image`), 3 seeds each.
  An earlier single run had shown 89.6 % instead of 66.4 % test Dice for QENAS. The random split is leaky
  (near-duplicate frames of one sequence on both sides), so it is **not adopted**. Its results may only be
  reported as a labelled comparison with the Mixed-GGNAS protocol.
* PNG copies are used (the TIFs are not decodable by Pillow and contain the same images). The PNG masks have
  anti-aliasing artefacts, so they are binarised at > 127.

### D36 — Preprocessing
* RGB with 3 channels for all datasets (grey images are replicated).
* Images: anti-aliased bilinear resize. Masks: binarised at native resolution, union of mask files, then
  nearest-neighbour resize.
* Fixed sizes without letterboxing, as in Mixed-GGNAS (BUSI/CVC 256 × 256).
* Per-channel normalisation statistics from the training split only.

### D37 — Augmentation (training split only)
* Horizontal flip (0.5); random affine with p = 0.5 (rotation ±15°, scale 0.9–1.1, translation ±5 %; bilinear for
  images, nearest for masks, with identical parameters); brightness/contrast ±10 %.
* Vertical flips only for CVC: endoscopy has no canonical orientation, whereas ultrasound depth and fundus
  anatomy do.
* The random stream is derived from (seed, epoch, index), so it is reproducible and identical after a resume.
  The two ZiCo batches are drawn from this same stream (epoch 0).

### D38 — Test isolation by construction
* The test split is listed in `split.json`, but its pixels are never loaded until `stage_test`. That stage
  refuses to run unless all previous stages are complete, evaluates once, and records `test_evaluated` (a re-run
  never re-evaluates).
* The search uses two training batches only. Normalisation statistics, augmentation, early stopping, checkpoint
  selection, candidate selection, α selection and the TTA decision use train/validation only.
* Discovery reads test files only for integrity checks and exact-duplicate grouping, which are prerequisites of
  a leakage-free split.

---

## Engineering

### D39 — Checkpointing and resume
* Atomic writes (temporary file + `os.replace`, retried while Windows/OneDrive briefly lock a file) and a `.prev`
  fallback for every checkpoint; corruption is detected and reported.
* Search: the pickled pymoo algorithm (including its `numpy.random.Generator`), archive and history after every
  generation. A search checkpoint is only resumed with the identical configuration *and* identical ZiCo batches.
* Training: model, optimiser, scheduler, AMP scaler, epoch, history, best metric and RNG states after every
  epoch.
* A configuration fingerprint guards every resume. Re-running a finished experiment reports it and never
  modifies it.

* **Pruning:** once a stage is done, the checkpoints it alone used are deleted (search state after the search,
  candidate runs after the candidate stage, α runs after the loss-weight stage), because their results are
  stored in the stage's JSON/CSV outputs. A finished experiment keeps only `checkpoints/final/best.pt`. This keeps
  78 study experiments within the Kaggle disk and GitHub file-size limits (a U-Net checkpoint is 124 MB).
* **Timer:** at the end of every run, the time from start to end (the sum of all stage times, which includes
  earlier sessions of a resumed run) and the wall-clock time of the current session are logged, stored as
  `total_seconds` in `state.json`, and appended to `summary.txt`.

### D40 — Determinism
* Seeds for Python, NumPy and PyTorch; `torch.use_deterministic_algorithms(True, warn_only=True)`; deterministic
  cuDNN; seeded per-epoch data order and augmentation.
* Bit-exactness is verified on CPU. On GPU some kernels (e.g. the transposed-convolution backward) may differ in
  the last bits.

### D41 — Result folders
* Every experiment is identified by (dataset, method, seed) and lives in `results/<DATASET>/<method>/seed_<s>/`
  (the CVC leakage study in `results/CVC_frame_split/`). Re-running the same command resumes or reports that
  directory; a changed configuration is refused. Results of earlier versions were deleted before the study.

### D42 — Kvasir-SEG: official split
* **Decision:** Kvasir-SEG (Jha et al., MMM 2020; 1000 polyp images) is the fourth dataset, at 256 × 256 like
  CVC-ClinicDB, with vertical flips (endoscopy). The authors publish a train/val split "for fair comparison":
  `val.txt` (120) is the test split, and 20 % of `train.txt` (880) is the validation split → 704 / 176 / 120.
* JPEG masks (≤ 2.1 % grey edge pixels) are binarised at > 127. Pixel hashing found no duplicates. There are no
  patient identifiers, so the split is image-level (as for BUSI).
* **Reason:** the most widely used polyp benchmark with an official split. It gives the polyp results more
  statistical power than CVC-ClinicDB's 6 test sequences, and it enables the external validation on
  CVC-ClinicDB (RQ4).

### D43 — The split does not depend on the run seed
* **Decision:** `data.split_seed = 42` is fixed; the run seed (`--seed`) changes only the search, initialisation,
  data order and augmentation.
* **Reason:** repeated runs measure the variability of the method, and every method and seed is tested on the
  same images, which makes paired per-image tests possible (D45).

### D44 — Evaluation study: conditions and baselines
* **Decision:** the study conditions (STUDY.md) are defined in one registry, `qenas/methods.py`, and run by
  `study.py`; `main.py` runs only QENAS.
  * The objective ablations change only the optimised objectives. All three objectives are still evaluated and
    recorded, so the cache and analyses stay complete.
  * The selection ablations copy QENAS' search and candidate runs of the same seed and change only the
    selection rule.
  * Random search keeps NSGA-II's evaluation budget.
* **Baselines:** U-Net (31.04 M) and Attention U-Net (8.73 M), the exact variants of the Mixed-GGNAS tables
  (identified by parameters and FLOPs, `tests/test_flops.py`). They are trained with QENAS' own protocol for its
  selected architecture (α grid on validation → final training → TTA decision → single test). Mixed-GGNAS itself
  cannot be re-run: its repository has no searched architectures, weights or training instructions, and its
  search takes about 26 h per dataset. Its published numbers (own splits) are reported for reference.
* **External validation** (RQ4): the Kvasir-SEG best-validation checkpoint is applied unchanged, with its own
  preprocessing and TTA decision, to all 612 CVC-ClinicDB frames.

### D45 — Statistics of the study
* **Decision:**
  * mean ± SD over 3 seeds;
  * paired, two-sided Wilcoxon signed-rank tests on per-image Dice (averaged over seeds) of the same test
    images, with a 95 % bootstrap CI of the mean difference and Holm–Bonferroni correction within each table and
    dataset;
  * hypervolume for search quality.
* **Reason:** with 3 seeds a test across seeds has almost no power, whereas the per-image pairing uses all test
  images. Wilcoxon makes no normality assumption about Dice, which is bounded and skewed. Holm controls the
  family-wise error without assuming independence.

---

## Changes from the two-objective version and why

| area | previous version | this version | reason |
|---|---|---|---|
| objectives | params, SynFlow | params, SynFlow, **ZiCo** (official code) | a data- and label-aware zero-cost objective (D10, D12) |
| objective agreement | — | measured: no conflict among quality objectives (D14) | requirement that the objectives work together |
| population / generations | 10 / 10 | **20** / 10 | non-dominated share with three objectives (D16) |
| search monitoring | best values per generation | + **hypervolume** and population medians (D17) | evidence that NSGA-II improves the front |
| Pareto subset | extremes of 2 objectives | extremes of 3 objectives + farthest point (D18) | three objectives |
| final metrics | Dice, IoU, mIoU, loss | Mixed-GGNAS metrics (Dice = DSC, mIoU, Params, FLOPs) + **HD95** + **Sensitivity** (D30–D32) | requested evaluation |
| FLOPs | own MAC count per image | thop convention per training batch (= Mixed-GGNAS tables) + per image | comparability, verified on U-Net (D32) |
| options | refit, reuse-from, TTA on/off, YAML configs, smoke mode, optimiser/scheduler/genotype/target choices, … | **none** (D1) | one pipeline, one protocol |
| results | `results/` | fixed folders `results/<DATASET>/<method>/seed_<s>/` (D41) | the evaluation study |
| datasets | BUSI, CVC-ClinicDB, IDRiD | + **Kvasir-SEG** (official split, D42) | polyp benchmark, external validation |
| evaluation | single runs | study: 3 seeds, baselines, ablations, random search, external validation, leakage study (D43–D45) | paper-level evaluation |
