# QENAS — Design decisions

Every decision that the specification left open, or where an official implementation had to be adapted, is
listed here as **Decision / Reason / Alternative considered / Why the alternative was rejected**.
Sources consulted: the Mixed-GGNAS paper (Hu et al., *ESWA* 289, 2025, 128338), the official Mixed-GGNAS code
(github.com/Hmxki/Mixed-GGNAS), NAS-Unet (github.com/tianbaochou/NasUnet), Synaptic-Flow
(github.com/ganguli-lab/Synaptic-Flow), the zero-cost-NAS reference (Abdelfattah et al., ICLR 2021), pymoo 0.6.2.

---

## Architecture

### D1 — Package layout `qenas/…` instead of top-level `datasets/`, `models/`, …
* **Reason:** a top-level `datasets` package shadows the widely installed HuggingFace `datasets` package and
  breaks imports unpredictably. One importable package keeps modules separated as requested
  (datasets / models / search / training / evaluation / visualization / utils) without name clashes.
* **Alternative:** the suggested flat layout. **Rejected:** import-shadowing risk.

### D2 — Manual blocks are the official Mixed-GGNAS implementations
* **Reason:** the specification asks to prefer official, tested code. `qenas/models/blocks.py` reproduces
  `raw_cells.py` (layer order, `groups=16`, dense growth 32 × 3 layers, Inception strip branches, ConvNeXt
  inverted bottleneck with BatchNorm and post-residual GELU, residual paths) and the kernel/dilation table
  (paper Table 1).
* **Alternative:** textbook ResNet/DenseNet/Inception/ConvNeXt blocks. **Rejected:** would not be the
  Mixed-GGNAS search space.
* Consequence: `model.base_channels` must be a multiple of 16.

### D3 — Convolution scale: one fixed configured scale, no selection, no collapse
* **Reason:** in Mixed-GGNAS the scale weights `w_i` are learned from the *supervised training loss*
  `L_train(ŷ(w), y)` (paper Eq. 1; official code trains `cells_weight` with labels from epoch 60 on).
  The specification says that a scale mechanism needing supervised training must not enter the training-free
  search, and that final training must contain no scale-selection/collapse stage. Hence every manual block
  uses the configured scale (`--scale`, default **3** = the first Table-1 variant: 3×3 convolutions for
  Residual/Dense — the canonical He/Huang blocks — strips (3,5,7) for Inception, 5×5 depthwise for
  ConvNeXt) identically in search, candidate training and final training.
* **Alternatives:**
  (a) learn scale weights training-free by maximising SynFlow — invented machinery with no published
  basis; (b) keep all three scale branches with fixed equal weights — triples cost and is not an
  architecture Mixed-GGNAS ever produces; (c) encode the scale in the chromosome — enlarges the space to
  15^8, contradicting the required 5^8. **Rejected** for those reasons.
* The official code also offers "block + identity" variants (6 paths); QENAS uses the plain block, whose
  own internal connectivity (residual/dense/inception/ConvNeXt) is unchanged.

### D4 — DARTS block = fixed, official, dataset-matched DARTS genotype
* **Reason:** in Mixed-GGNAS the DARTS block is a NAS-Unet cell whose operations were found by a DARTS
  search; the GA only decides *where* it is used. Running that DARTS search inside QENAS would require
  supervised bilevel training on the dataset, violating the training-free search. The official repository
  publishes the derived genotypes (`darts_cell_busi`, `darts_cell_cvc`, `darts_cell_idrid`); QENAS uses them
  verbatim (`--darts-genotype auto`). The primitive operations are the NAS-Unet operations vendored by
  Mixed-GGNAS (GroupNorm with 2 channels/group, cweight, dilated/depthwise/transposed convolutions, pooling).
* **Alternatives:** (a) DARTS search inside QENAS — supervised, rejected; (b) a hand-made "DARTS-like" block —
  explicitly forbidden.
* **Caveat (reported honestly):** the dataset-specific genotypes were derived by the Mixed-GGNAS authors on
  their own split of the same dataset, whose membership is unknown. They are a fixed structural prior (16
  discrete operation choices), not access to our test labels, but a strict reader may view this as weak
  information transfer. For a fully dataset-independent run use `--darts-genotype nasunet` (the NAS-Unet cell
  searched on PROMISE12). We recommend reporting both if this matters for a claim.

### D5 — DARTS cell wiring follows NAS-Unet; resolution handling made explicit
* **Reason:** NAS-Unet feeds a down cell with (two-levels-back feature at 2× resolution, previous feature)
  and an up cell with (encoder skip at the output resolution, previous decoder feature). The official
  Mixed-GGNAS code instead passes the "previous-previous" decoder output to up cells (no encoder skip) and fixes
  the resulting size mismatches with nearest-neighbour interpolation. QENAS uses the NAS-Unet wiring, which is
  internally consistent and gives DARTS decoder positions the same skip information as manual ones. At the
  first encoder position both inputs are the stem output (same resolution), so `preprocess0` uses stride 1
  there. Genotypes are validated: an operation that would break spatial consistency raises an error instead
  of being silently resized.
* **Alternative:** replicate the official interpolation hack. **Rejected:** hides shape errors and starves
  DARTS decoder cells of skip connections.

### D6 — U-shaped framework: standard skip alignment; no ViT branch, no deep supervision
* **Reason:** skips connect the encoder feature at the decoder position's *output* resolution
  (D1←E3 … D4←stem), the standard U-Net alignment. The official code feeds each decoder the encoder output at
  its *input* resolution and up-samples it with an extra transposed convolution (the bottleneck is used twice
  and the full-resolution stem feature is never used). The ViT fusion module and the multi-scale (deep
  supervision + Smooth-L1) loss are not part of QENAS: the ViT hyper-parameters are tuned by a supervised GA
  in Mixed-GGNAS, and the QENAS loss is specified as BCE + soft-mIoU. (The ViT call is also commented out in
  the official `CellModel.forward`.)
* **Alternative:** replicate the official skip shift. **Rejected:** loses full-resolution detail for no
  documented reason.

### D7 — Spatial-channel attention on the skip: `s + s·σ_s(s)·σ_c(s)`
* **Reason:** the official decoder computes `en + satt(en) * catt(en)` where both terms already contain
  `en`, i.e. `en + en²·σ_s·σ_c`. The quadratic term is not "attention-enhanced features" as described in the
  paper and makes the network non-homogeneous in its activations. Measured on BUSI (256×256, same
  initialisation): it inflates SynFlow by 10^2–10^5 depending on the encoder blocks (e.g. all-Residual
  2.0e24 → 3.9e26, all-ConvNeXt 8.5e37 → 1.8e43; no overflow), i.e. the proxy would partly measure the
  squared activation magnitude arriving through the skip rather than the architecture's synaptic flow, and
  it would do so unequally across encoder choices. QENAS uses the standard combined gating `s·σ_s·σ_c`.
* **Alternative:** keep the official quadratic form. **Rejected:** distorts the SynFlow comparison between
  architectures and deviates from the described intent.

### D8 — Binary output: one logit per pixel
* **Reason:** BCE + soft-mIoU for binary segmentation; the official two-channel softmax is equivalent but
  redundant. **Alternative:** two channels + CE. **Rejected:** the specified loss is BCE.

---

## Objectives and search

### D9 — SynFlow: official procedure, with normalisation layers bypassed
* **Procedure:** Synaptic-Flow `SynFlow.score` + zero-cost-NAS `compute_synflow_per_weight`: eval mode,
  `linearize` (abs of every state-dict tensor), single all-ones input of the data shape, float64,
  `R = sum(output)`, score `|θ·∂R/∂θ|` over Conv/Linear weights (biases and normalisation parameters
  excluded, as in both references), signs restored.
* **QENAS modifications:** (i) `ConvTranspose2d` weights are included (the decoder up-samples with transposed
  convolutions; excluding them makes SynFlow blind to the decoder); (ii) **all normalisation layers are
  bypassed**. zero-cost-NAS' `synflow` measure is computed on a copy of the network *without* BatchNorm
  (`bn=False`); Synaptic-Flow evaluates BN in eval mode, which at initialisation is the identity. Both are
  therefore the synaptic flow of the normalisation-free network. The DARTS operations use GroupNorm, which
  normalises with per-sample statistics and cannot be linearised. Measured on BUSI (256×256): an all-DARTS
  network scores ln(1+S) = 12.8 with GroupNorm active but 78.3 without it, while BN-only networks are
  unchanged to four significant digits (e.g. all-Residual 2.040e24 in both cases; unit-tested). Leaving
  GroupNorm active would penalise DARTS blocks by ~30 orders of magnitude for a numerical artefact.
* **Input:** deterministic all-ones tensor `[1, 3, H, W]` at the training resolution (official construction;
  no random input). Same tensor for every architecture.
* **Initialisation:** PyTorch default initialisation under a fixed seed (`search.synflow.init_seed`, default
  = global seed) inside `torch.random.fork_rng`, so the global random stream is untouched and every
  evaluation is reproducible.

### D10 — Positive, correctly directed SynFlow objective `f2 = 1/(1 + ln(1 + S))`
* **Reason:** pymoo minimises; the requirement is a positive objective with the correct direction and no
  blind sign flip. `S ≥ 0` and larger is better. Because `S` spans ~10^5–10^38 (exponential in depth), the
  natural scale is `L = ln(1+S) ≥ 0` (log1p: finite and non-negative even at S = 0). `f2 = 1/(1+L)` is the
  classical positive reciprocal transform of a non-negative maximisation criterion: strictly decreasing,
  in (0, 1], and 1 only for a network with no flow. Pareto dominance is unchanged by monotone transforms.
* **Alternatives:** `−S` or `−ln S` (negative values, explicitly disallowed); `1/S` (positive but its
  1e-5…1e-38 range collapses crowding-distance normalisation onto one point and is undefined at S = 0);
  `C − ln S` (needs an arbitrary bound C). **Rejected** for those reasons.

### D11 — The scored network is the trained network
* **Reason:** candidate and final training initialise the model with the same seed used for SynFlow, so the
  parameters that SynFlow measured are exactly the starting point of supervised training. This removes one
  source of search/train mismatch.

### D12 — NSGA-II configuration
* **Decision:** pymoo `NSGA2`, `IntegerRandomSampling`, `UniformCrossover(prob=0.9)`,
  `ChoiceRandomMutation` (per-gene p = 1/8), `eliminate_duplicates=True`, population 10, generation 0 +
  10 generations (Mixed-GGNAS uses 10 × 10). The search runs through pymoo's ask/tell interface so that each
  evaluation can be cached, logged and checkpointed.
* **Reason:** genes are *nominal* (block types have no order), so uniform crossover (= Mixed-GGNAS' binomial
  crossover with CR = 0.5) and random-resetting mutation are the appropriate operators; p = 1/L is the
  standard mutation rate.
* **Alternative:** SBX + polynomial mutation with rounding (pymoo's integer example). **Rejected:** they
  assume ordinal distance between block types.
* Larger populations/generations are a single CLI flag away (SynFlow costs ~2–7 s per architecture on CPU).

### D13 — Representative Pareto subset
* **Decision:** all points if |front| ≤ k (default 8); otherwise both extremes + greedy farthest-point
  sampling in the min–max-normalised objective space.
* **Reason:** guarantees the subset spans the full trade-off and covers it evenly (k-centre 2-approximation),
  deterministic, no preference weights.
* **Alternative:** top-k by crowding distance. **Rejected:** crowding distance is infinite for both extremes
  and only local, so it can pick clustered points.

---

## Supervised stages

### D14 — Candidate training protocol (20 epochs)
* **Decision:** AdamW, lr 1e-3, weight decay 5e-5 (paper §4.2), batch 8 (IDRID 4), **constant** learning
  rate, α = β = 0.5, no early stopping; identical initialisation seed, data order and augmentation stream for
  all candidates (common random numbers).
* **Reason:** ΔDice must reflect the architecture, not a learning-rate annealing artefact or random luck;
  α = 0.5 is the neutral weighting before α is optimised.
* **Alternative:** the final poly schedule truncated to 20 epochs. **Rejected:** annealing to zero forces all
  curves to flatten and compresses ΔDice differences.

### D15 — Learning-capability criterion (gates + ΔDice)
* **Decision:** see METHODOLOGY §5. Five falsifiable gates (finite, generalising loss decrease, no
  overfitting trend, significant Dice learning trend, beats the trivial predictor), then maximal ΔDice with a
  data-derived noise tolerance and lexicographic tie-breaks (final Dice → ΔLoss → fewer parameters).
* **Reason:** keeps ΔDice as the primary criterion as specified while preventing pathological winners
  (unstable, overfitting, diverged, or non-learning architectures). Thresholds are standard significance
  levels (p < 0.05) or estimated from the curves themselves — no hand-tuned weights.
* **Alternatives:** weighted sum of ΔDice/final Dice/ΔLoss (arbitrary weights — forbidden); highest final
  Dice (explicitly not wanted); area under the learning curve (conflates level and improvement);
  Pareto ranking over the learning criteria (the max-ΔDice candidate is always non-dominated, so it
  degenerates). **Rejected.**

### D16 — Loss: BCE + soft mIoU (two-class mean), per image, smoothing 1
* **Reason:** "mIoU" in the Mixed-GGNAS evaluation is the class-mean IoU (its mIoU exceeds its Dice, the
  signature of including background). The soft version over foreground and background is bounded in
  [0, 1], well defined for empty masks with Laplace smoothing 1, and computed per image to match per-image
  evaluation.
* **Alternative:** foreground-only soft IoU / Dice loss. **Rejected:** not the specified loss.

### D17 — Loss-weight optimisation
* **Decision:** grid α ∈ {0.2, 0.4, 0.5, 0.6, 0.8}, β = 1 − α, each a 20-epoch run of the selected
  architecture with the candidate protocol (same seed/data order); criterion = mean validation Dice of the
  last 5 epochs; ties → α closest to 0.5. The α = 0.5 run is bit-identical to the candidate run and is
  reused (verified: CPU training is deterministic).
* **Reason:** deterministic, cheap, symmetric around the neutral weighting and includes the official 0.6/0.4.
  The last-5-epoch mean is robust to single-epoch noise on small validation sets.
* **Disclosure:** the same validation set is used for candidate selection, α selection, early stopping and
  checkpoint selection — this is validation-set model selection; the test set is not touched.
* **Alternative:** optimise α jointly in the search or on test data. **Rejected:** would break the
  training-free search / test isolation.

### D18 — Final optimiser and schedule
* **Decision:** AdamW (lr 1e-3, wd 5e-5) with the official Mixed-GGNAS schedule: one epoch linear warm-up
  from 1e-3·lr, then poly decay `(1 − t/T)^0.9` stepped per iteration; 100 epochs by default (configurable).
* **Reason for a scheduler:** warm-up stabilises the first AdamW steps of a network trained from scratch
  (BatchNorm statistics are still uncalibrated); the decay lets the final epochs converge instead of
  oscillating at the initial step size — the standard DeepLab policy and the one used by the official code.
* **Alternatives:** constant LR (does not converge as tightly), cosine (available via `--scheduler cosine`).

### D19 — Early stopping only on overfitting
* **Decision:** stop only if (i) ≥ 30 epochs, (ii) no validation-Dice improvement for 20 epochs, and
  (iii) significant increase of validation loss together with significant decrease of training loss over
  those 20 epochs (one-sided Kendall τ, p < 0.05).
* **Reason:** stagnation alone never stops training, as required; the classic overfitting signature does.
  The best-validation-Dice checkpoint is restored before testing in every case.

### D20 — Gradient clipping and AMP
* Gradient clipping is off by default (no instability observed; `--grad-clip` enables it). AMP (fp16 + loss
  scaling) is enabled automatically on CUDA and disabled on CPU; losses and metrics are always computed in
  float32/float64.

### D21 — Metrics
* Per-image Dice and IoU at threshold 0.5, averaged over the split; the empty-GT/empty-prediction case counts
  as 1. Class-mean mIoU is reported in addition for comparability with Mixed-GGNAS. Training metrics are
  measured on augmented batches during training; after final training the train split is re-evaluated
  without augmentation with the best checkpoint.

---

## Data

### D22 — BUSI: 647 benign + malignant images, conflicting duplicate removed → 645
* **Reason:** Mixed-GGNAS uses the 647 lesion images (Table 3); `normal` images have empty masks
  (`--config` key `data.busi_include_normal` to include them). Multiple lesion masks of one image
  (`_mask_1`, `_mask_2`) are merged by union. Pixel hashing found one image present as both
  `benign (433)` and `malignant (145)` with *different* masks (IoU 0.77) → ambiguous ground truth, both
  copies excluded (identical-mask duplicates would be kept once).
* **Split:** stratified by class (benign/malignant), 20 % test, then 20 % of the remainder for validation
  → 413 / 103 / 129. BUSI has no patient identifiers, so the split is image-level (limitation; exact duplicates
  are handled).

### D23 — CVC-ClinicDB: sequence-level split
* **Reason:** 612 frames come from 29 colonoscopy sequences (metadata.csv); consecutive frames are
  near-duplicates, so a random frame split leaks. Whole sequences are assigned to splits (≈20 % test, ≈20 % of
  the rest validation) → 395 / 96 / 121 frames (18 / 5 / 6 sequences). Note: this is stricter (and gives lower
  scores) than the random frame split of the Mixed-GGNAS paper.
* PNG copies are used (the TIFs are not decodable by Pillow and contain the same images). The PNG masks have
  anti-aliasing artefacts (up to 8 % intermediate grey pixels in a few masks) → binarised at > 127.

### D24 — IDRiD: optic disc target, official split
* **Reason:** the IDRiD segmentation sub-challenge provides five binary targets. Mixed-GGNAS reports Dice
  96.6 % on "IDRID", which is only attainable for the optic disc (lesion Dice values are far lower), so the
  default target is **OD** (`--idrid-target MA|HE|EX|SE|LESIONS` available; for lesions a missing mask file
  means "lesion absent", as defined by the dataset). The official 54/27 split is kept; 11 of the 54 training
  images form the validation set (43 / 11 / 27). No patient metadata → image-level validation split.
* **Resolution:** 320 × 512 (H × W), as in the paper (512 × 320), close to the native 2848 × 4288 aspect;
  batch size 4.

### D25 — Preprocessing
* RGB with 3 channels for all datasets (BUSI grey images are stored as RGB; replication keeps one
  architecture definition). Images: anti-aliased bilinear resize. Masks: binarised at native resolution,
  union of mask files, then nearest-neighbour resize. Resize to the paper's fixed sizes (BUSI/CVC 256 × 256)
  without letterboxing, as in Mixed-GGNAS. Per-channel normalisation statistics from the training split only.

### D26 — Augmentation (training split only)
* Horizontal flip (0.5), random affine with p = 0.5 (rotation ±15°, scale 0.9–1.1, translation ±5 %, bilinear
  for images / nearest for masks with identical parameters), brightness/contrast ±10 %. Vertical flips only
  for CVC (endoscopy has no canonical orientation; ultrasound depth and fundus anatomy do). The random stream
  is derived from (seed, epoch, index) → reproducible and identical after a resume.

### D27 — Test isolation by construction
* The test split is listed in `split.json` but its pixels are never loaded until `stage_test`, which refuses to
  run unless all previous stages are complete, evaluates once, and records `test_evaluated` in the
  experiment state (a re-run never re-evaluates). Normalisation statistics, augmentation, early stopping,
  checkpoint selection, candidate selection and α selection use train/validation only. Discovery reads test
  files only for integrity checks (readability, size agreement, binary masks) and exact-duplicate grouping,
  which are prerequisites of a leakage-free split.

---

## Engineering

### D28 — Checkpointing and resume
* Atomic writes (temporary file + `os.replace`, retried when Windows/OneDrive briefly locks a file) and a
  `.prev` fallback for every checkpoint; corruption is detected and reported. Search: pickled pymoo algorithm
  (including its `numpy.random.Generator`), archive and history after every generation. Training: model,
  optimiser, scheduler, AMP scaler, epoch, history, best metric, RNG states after every epoch. Completed
  runs drop optimiser state to save disk. A configuration fingerprint guards every resume. Verified: a run
  killed with SIGKILL mid-training and resumed produces bit-identical histories and test metrics.

### D29 — Determinism
* Seeds for Python/NumPy/PyTorch, `torch.use_deterministic_algorithms(True, warn_only=True)`, cuDNN
  deterministic, seeded per-epoch data order and augmentation. Bit-exactness is verified on CPU; on GPU some
  kernels (e.g. transposed-convolution backward) may still differ in the last bits.
