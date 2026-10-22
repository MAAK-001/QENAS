# QENAS — Design decisions (QENAS 3.0)

Every decision that is not dictated by an official implementation, and every adaptation of official code, is
listed as **Decision / Reason (evidence) / Alternative considered / Why the alternative was rejected**. Rule
applied throughout: a choice is adopted when it is proved by a reference implementation, published evidence or a
measurement in this project. A merely plausible change is not adopted.

**Evidence of this project.** Evaluation study v1 (QENAS 2.1, 78 experiments, seeds 42–44; code: git tag
`study-v1`; results: GitHub branch `kaggle-study`, local copy `Old Results/study_v1/`; report
`Old Results/QENAS_Study_v1.pdf`)
is cited as "v1 Tn" (table n of `Old Results/study_v1/analysis/`). Evaluation study v2 (QENAS 3.0, seeds 45–47; STUDY.md)
tests every decision changed here.

**Sources consulted:**

* Mixed-GGNAS: the paper (Hu et al., *ESWA* 289, 2025, 128338) and the official code (github.com/Hmxki/Mixed-GGNAS);
* NAS-Unet (github.com/tianbaochou/NasUnet);
* ZiCo: the paper (Li et al., ICLR 2023) and the official code (github.com/SLDGroup/ZiCo);
* ZiCo-BC (Bhardwaj et al., ICCVW 2023);
* Synaptic-Flow (github.com/ganguli-lab/Synaptic-Flow) and the zero-cost-NAS reference (Abdelfattah et al.,
  ICLR 2021) — QENAS 2.x only;
* NAS-Bench-Suite-Zero (Krishnakumar et al., NeurIPS 2022);
* successive halving (Jamieson & Talwalkar, AISTATS 2016), Hyperband (Li et al., JMLR 2018) and ASHA (Li et al.,
  MLSys 2020);
* nnU-Net (Isensee et al., *Nat. Methods* 18, 2021) and nnunetv2 2.8.1 (github.com/MIC-DKFZ/nnUNet);
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
  * `--stop-after`.
* **Removed alternative paths of earlier versions:**
  * optional train+validation refit (`--refit`) and re-use of another experiment's stages (`--reuse-from`);
  * forced TTA on/off, and (3.0) the TTA decision itself (D27);
  * YAML configuration overrides and a separate smoke configuration;
  * choice of optimiser, scheduler, convolution scale, DARTS genotype, IDRiD target, BUSI `normal` images,
    duplicate handling and normalisation;
  * AMP/gradient-clipping overrides;
  * hyper-parameter flags;
  * disabling the loss-weight search or the early stopping.
* **Reason:** an experiment is only interpretable if every reported number comes from the same, single protocol.
  The removed paths were either off by default (refit, reuse) or never-used variants.
* **What remains adjustable, and why:**
  * the **seed**, because independent repetitions are needed to report variability;
  * device, workers and data root, which do not affect results (excluded from the configuration fingerprint).
* **The evaluation study** (baselines, ablations, random search, repeated seeds; STUDY.md) does not add options to
  the method: its conditions are defined in `qenas/methods.py` and run by `study.py` (D44); `main.py` runs only
  QENAS.
* **How the end-to-end check is done:** the unit tests run the real pipeline on a small synthetic dataset
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
  choices), not access to our test labels, but a strict reader may see weak information transfer.
* **Kvasir-SEG** has no published Mixed-GGNAS genotype. It uses `darts_cell_cvc`, the genotype derived on
  CVC-ClinicDB: the same imaging modality (colonoscopy polyps) and the only polyp dataset of the Mixed-GGNAS
  paper. Caveat for the external validation: this structural prior was derived (by Mixed-GGNAS, on their split)
  on CVC-ClinicDB, the external test dataset; no CVC-ClinicDB image or weight enters the Kvasir-SEG models.

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
  Measured on BUSI (256×256, same initialisation, QENAS 2.x), it inflated the SynFlow proxy by 10^2–10^5
  unequally across encoder choices, i.e. it changes signal propagation in an architecture-dependent way.
* **Alternative:** the quadratic form. **Rejected:** it is not the described operation and distorts the
  comparison between architectures.

### D9 — Binary output: one logit per pixel
* **Reason:** binary segmentation with BCE + soft-mIoU. The official two-channel softmax is equivalent but
  redundant.

---

## Objectives and search

### D10 — Two objectives: parameters and ZiCo (SynFlow removed in QENAS 3.0)
* **Decision:** f1 = #parameters (minimise), f2 = ZiCo (maximise). Both are computed at initialisation; no weight
  is updated.
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
* **Why SynFlow was removed (v1 evidence):**
  * removing it was never significantly worse and was significantly **better** on Kvasir-SEG: +3.10 pp test Dice,
    Holm p = 2.3e-4 (v1 T4; BUSI −0.81 pp, n.s.);
  * as the second objective next to the parameter count, ZiCo beat SynFlow on the same test images: Kvasir-SEG
    +1.32 pp (Wilcoxon p = 0.0025), BUSI +0.58 pp (n.s.) (v1 runs `qenas_no_synflow` vs `qenas_no_zico`);
  * SynFlow agrees with ZiCo (Kendall τ +0.17…+0.39, still +0.13…+0.41 after removing model size) and trades off
    against the parameter count (τ −0.41…−0.51): it adds little beyond ZiCo and mostly rewards larger networks
    (v1 T6, D14);
  * with three objectives the whole final population was non-dominated in 12 of 12 runs (D16);
  * SynFlow was not correlated with the trained quality of the candidates (Spearman ρ −0.19…+0.23, n.s.; v1 T9).
* **Alternatives considered for a second quality objective (QENAS 2.x research):** NASWOT and SWAP (label-free
  activation-pattern proxies; no segmentation-specific evidence); MeCo (data- and label-independent by design);
  label–gradient alignment (needs some training); effective-receptive-field proxies (detection only). **Rejected**
  because none has comparable evidence on dense prediction together with use of the labels; v1 shows that a
  second, agreeing quality proxy does not help.
* **Honest limit:** no zero-cost proxy is *guaranteed* to rank segmentation networks correctly; inside the elite
  of a search ZiCo did not predict the 20-epoch validation Dice (v1 T9: ρ −0.28…+0.37, n.s.). The proxies narrow
  390,625 architectures down to an elite; the supervised successive halving chooses within it (D20, D21).

### D11 — SynFlow (QENAS 2.x only)
* QENAS 2.x computed SynFlow with the official Synaptic-Flow / zero-cost-NAS procedure (eval mode, `linearize`,
  all-ones float64 input, `|θ·∂R/∂θ|` over Conv/ConvTranspose/Linear weights, normalisation layers bypassed) and
  minimised `1/(1+ln(1+S))`. It was removed in 3.0 (D10); the 2.x code is preserved at git tag `study-v1`.

### D12 — ZiCo: the official code, unchanged; how it is integrated
* **Official logic kept exactly:** `getgrad`, `caculate_zico` and `getzico` from
  `SLDGroup/ZiCo/ZeroShotProxy/compute_zico.py` are reproduced line by line in `qenas/search/objectives.py`,
  including the unused variables (byte-identical to QENAS 2.x):
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
  (no weight is ever updated) but is not data-free. Validation and test images are never used.
* **Not applied: ZiCo-BC's bias correction.** It changes the ZiCo score (it subtracts a depth/width penalty with
  a tuned hyper-parameter). The bias it targets — larger scores for networks with more layers — is measured
  (D14): one DARTS or Inception gene adds +30…+96 and +66 ZiCo (v1 T7).

### D13 — Positive, minimised objective `f2 = 1/(1+ZiCo)`
* **Reason:** pymoo minimises. The requirement is a positive objective with the correct direction and no blind
  sign flip. ZiCo is a sum of logarithms and ≥ 0; `1/(1+x)` maps it to (0, 1]: strictly decreasing, positive,
  and 1 only for a zero score. Pareto dominance is invariant to monotone transforms, so the ranking equals
  "maximise ZiCo".
* **Alternatives:** `−ZiCo` (negative values); `C − ZiCo` (needs an arbitrary bound C). **Rejected.**

### D14 — Objective relations (measured in study v1)
* **Evidence:** v1 T6/T7, computed from the random-search archives (220 uniformly random architectures per
  dataset and seed, all four datasets, 3 seeds — an unbiased sample, unlike the NSGA-II archive):
  * params vs ZiCo: Kendall τ −0.09 to −0.18 — ZiCo is almost independent of model size, so the two objectives
    carry different information and a genuine trade-off curve exists;
  * params vs SynFlow −0.41 to −0.51 and SynFlow vs ZiCo +0.17 to +0.39 (+0.13 to +0.41 with size removed): the
    two quality proxies agreed, so SynFlow was largely redundant (D10);
  * block effects (v1 T7, one gene vs Residual): DARTS +30…+96 ZiCo, Inception +66, Dense +21, ConvNeXt +26;
    ConvNeXt adds the most parameters (+0.42 M).
* **Consequence:** the two remaining objectives do not conflict as quality signals; the size/quality trade-off is
  the intended one. The known depth bias of ZiCo (D12) is left to the supervised selection, which judges the
  trained networks.

### D15 — The scored network is the trained network
* **Reason:** ZiCo, candidate training and final training all build the network with the same class and the
  same seeded initialisation. The parameters ZiCo measured are exactly the starting point of supervised
  training, which removes one source of search/train mismatch.

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
  objectives.
  * v1 T8 and the v1 random-search archives: the non-dominated share of a random population of 20 is
    **21–22 % with (params, ZiCo)**, 29–34 % with (params, SynFlow) and 40–51 % with three objectives;
  * v1 NSGA-II runs: with three objectives the *entire* final population was non-dominated in 12 of 12 runs;
    with (params, ZiCo) 11–20 of 20 (mean 14.2).
  Population 20 with two objectives therefore keeps a rank-based selection pressure. It is kept unchanged.
* **Generations: 10**, unchanged from Mixed-GGNAS (≤ 220 evaluations). NSGA-II beat random search with the same
  budget in hypervolume in 12 of 12 v1 runs (v1 T5); v2 re-tests this with two objectives (STUDY.md RQ4).
* **Cost:** the search took 5–10 min per run on a Tesla T4 in v1 (v1 T13), including SynFlow; 3.0 evaluates only
  ZiCo.

### D17 — Search progress measured by hypervolume
* **Decision:** after the search, the hypervolume of every generation's non-dominated set is computed. It uses
  the objective space min–max normalised over all evaluated architectures and reference point 1.1 per objective
  (pymoo `HV`). It is reported in `nsga2_generations.*`, in `search_summary.json` and in the figure
  `search_progress`, together with the population's median parameters and ZiCo per generation.
* **Reason:** hypervolume is the standard Pareto-compliant quality indicator of a multi-objective population. It
  shows directly whether NSGA-II improves the front over the random initial population.

### D18 — The 16 candidates of the final population
* **Decision:** k = 16 candidates. If the final front has more than 16 points: the extreme point of each objective
  plus greedy farthest-point sampling in the min–max-normalised objective space (k-centre 2-approximation). If it
  has at most 16 (the usual case with two objectives, D16): the whole front, filled up with the next individuals
  of the final population in NSGA-II's own survival order (Pareto rank, then larger crowding distance).
* **Reason:** successive halving (D20) needs a fixed number of candidates for its fixed budget. Inside the elite
  the zero-cost objectives do not predict trained quality (v1 T9: no significant Spearman ρ on any dataset for
  params, SynFlow or ZiCo), so the next-ranked members of the final population are as promising as the last front
  members; supervised training then decides. The fill-up uses NSGA-II's own ranking, so no new preference is
  introduced.
* **Alternatives:** (a) only the front, with a variable number of candidates — a variable budget and, with fewer
  than 16, a successive halving that ends with fewer finalists; (b) top-k by crowding distance alone — crowding is
  infinite for extremes and only local, so it can pick clustered points. **Rejected.**

### D19 — Evaluation cache keyed by everything the objectives depend on
* **Decision:** the cache key hashes:
  * the chromosome;
  * the architecture definition (channels, scale, genotype);
  * the objective definitions and version (3 in QENAS 3.0);
  * the input size and initialisation seed;
  * the **SHA-256 of the exact ZiCo batches** (images and masks), which covers dataset, split, augmentation,
    normalisation and seed;
  * the PyTorch version.
* **Reason:** with ZiCo the objectives depend on data. A cache keyed only by the architecture could silently
  reuse a value computed on other data. Entries of earlier versions are never matched.

---

## Supervised stages

### D20 — Candidate training: successive halving 16 × 5 → 8 × 10 → 4 × 20 epochs
* **Decision:** the 16 candidates are trained by successive halving (Jamieson & Talwalkar 2016; the rung schedule
  of Hyperband/ASHA with reduction factor η = 2):
  * rung 1: all 16 candidates to 5 epochs;
  * rung 2: the best 8 by the criterion (D21) continue to 10 epochs;
  * rung 3: the best 4 continue to 20 epochs (the finalists).
  Budget: 16 × 5 + 8 × 5 + 4 × 10 = **160 candidate-epochs**, the budget of v1's 8 candidates × 20 epochs.
  Every run uses identical conditions (common random numbers): AdamW, lr 1e-3, weight decay 5e-5 (paper §4.2);
  batch 8 (IDRiD 4); **constant** learning rate; α = β = 0.5; no early stopping; the same initialisation seed,
  data order and augmentation stream. A run that diverges is eliminated (gate G1). Ties in a promotion: fewer
  parameters.
* **Continuation is exact:** with a constant learning rate and an epoch-indexed data order, a run stopped after
  rung k and continued later is the run that would have been trained without interruption (unit test
  `test_run_continued_rung_by_rung_equals_an_uninterrupted_run`; `Trainer.fit(until=…)` keeps the full
  optimiser and RNG state).
* **Evidence (v1 candidate runs, 8 candidates × 20 epochs, 12 runs):** the ranking after 5 epochs agrees with the
  ranking after 20 (Kendall τ between the 5-epoch and the end-of-run criterion 0.36 on average, range 0.00–0.71;
  epochs 6–10: 0.37). Halving 8 → 4 at 5 epochs would have kept the eventual best candidate in **11 of 12** runs,
  and halving 4 → 2 at 10 epochs in 8 of 12. Early rungs are therefore informative enough to discard the clearly
  weaker half, while the final choice needs the full 20 epochs (QENAS keeps 4 finalists, not 2).
* **Reason:** v1 showed that the selection among 8 candidates was unreliable and that the proxies cannot
  pre-select (v1 T9, T10). Successive halving evaluates twice as many candidates with the same compute and spends
  the long runs only on the promising ones.
* **Alternative:** the final poly schedule truncated to the rung length. **Rejected:** annealing makes a stopped
  run differ from a continued one and compresses the differences between candidates.

### D21 — Learning-capability criterion: mean validation Dice of the last 5 epochs + gates
* **Decision:** among the finalists that pass the five gates (finite; generalising loss decrease; no overfitting
  trend; significant Dice learning trend; beats the trivial all-foreground predictor — METHODOLOGY §5), the one
  with the highest **mean validation Dice over its last 5 epochs** is selected; ties → fewer parameters. The same
  criterion ranks the candidates at every rung (at rung 1 it is the mean over epochs 1–5). If no finalist passes
  every gate, gates are relaxed in the fixed order G4 → G3 → G2 → G5 (G1 never), and only the gates that actually
  blocked every finalist are reported.
* **Reason (v1 evidence):** QENAS 2.x selected by ΔDice = Dice(20) − Dice(1). ΔDice rewards a poor start as much
  as a good end: the ΔDice choice was the candidate with the best 20-epoch validation Dice in only **3 of 12** runs
  (mean rank 3.3 of 8; v1 T10), and the argmax of ΔDice equalled the argmax of the end-of-run criterion in **1 of
  12** runs. The end-of-run validation Dice measures the quantity that matters; averaging the last 5 epochs reduces
  single-epoch noise, so the data-derived noise tolerance of 2.x is no longer needed. The gates keep excluding
  pathological winners (an erratic curve with a lucky end, overfitting, no learning).
* **Alternatives:** keep ΔDice (refuted by v1); the single last-epoch Dice (noisier); the best epoch's Dice
  (optimistic maximum of a noisy curve); a weighted score (arbitrary weights). **Rejected.**
* **Measured in v2** (STUDY.md RQ3): learning-capability selection vs a random candidate and vs the zero-shot rule
  (max ZiCo), and the agreement of the rungs and what ΔDice would have chosen (T8).

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
  * the α = 0.5 run is identical to the selected finalist's 20-epoch candidate run and is reused.
* **v1 evidence:** α = 0.5 was never chosen in the 12 QENAS runs (0.2 six times, 0.4 three, 0.6 two, 0.8 once);
  the validation criterion of the chosen α exceeded α = 0.5 by +1.35 pp on average. This gain is measured on the
  runs that choose α, i.e. it is an optimistic (best-of-five) estimate, and the grid costs 5–24 min, 17–24 % of a
  QENAS run (v1 T13). Whether it improves the *test* result was not measured in v1.
* **Status:** kept unchanged in 3.0 and tested in study v2 against a fixed α = 0.5 on the same architectures
  (D49). If the grid gives no test benefit, it is removed in the next version.
* **Disclosure:** the same validation set is used for candidate selection, α selection and checkpoint selection.
  This is validation-set model selection; the test set is not touched.

### D24 — Final optimiser and schedule
* **Decision:** AdamW (lr 1e-3, wd 5e-5) with the official Mixed-GGNAS schedule:
  * one epoch of linear warm-up from 1e-3·lr;
  * then poly decay `(1 − t/T)^0.9`, stepped per iteration;
  * **120 epochs** (D29).
* **Reason:** warm-up stabilises the first AdamW steps of a network trained from scratch; the decay lets the
  final epochs converge. This is the DeepLab policy used by the official code.

### D25 — Early stopping only on overfitting
* **Decision:** stop only if all three hold:
  1. ≥ 30 epochs;
  2. no validation-Dice improvement for 20 epochs;
  3. a significant increase of the validation loss together with a significant decrease of the training loss
     over those 20 epochs (one-sided Kendall τ, p < 0.05).
* **Reason:** stagnation alone never stops training; the classic overfitting signature does. The best
  checkpoint is restored before testing in every case. In v1 it stopped 3/3 BUSI QENAS runs and 1/3 runs on the
  other datasets (v1 T12).

### D26 — AMP on CUDA only; no gradient clipping
* **Decision:** AMP (fp16 + loss scaling) is used on CUDA and never on the CPU; losses and metrics are always
  computed in float32/float64. No gradient clipping.
* **Reason:** no instability was observed without clipping (no candidate or final run diverged in v1). AMP only
  changes speed and memory, not the protocol.

### D27 — No test-time augmentation (removed in QENAS 3.0)
* **Decision:** the test (and every validation and external evaluation) uses the plain network prediction.
* **Reason (v1 T11):** QENAS 2.x used horizontal-flip TTA if it raised the validation Dice. TTA was chosen in 6
  of 12 QENAS runs (43 of 66 runs over all conditions), and its mean validation effect was within ±0.5 pp for
  QENAS (BUSI −0.24, CVC-ClinicDB +0.32, IDRiD −0.02, Kvasir-SEG +0.43 pp) and within ±0.65 pp for every
  condition. It doubled the inference cost and added a decision step without a measurable benefit.
* **Alternative:** always on. **Rejected:** no evidence of a benefit; doubles latency.

### D28 — No refit on train + validation
* **Decision:** the final model is the best-validation checkpoint of the run on the training split.
* **Reason:** a model refitted on train + validation can no longer be checked on any held-out data before the
  single test, and its stopping epoch would have to be transferred from another run. No measurement supports a
  gain that would justify that loss of control.

### D29 — Final training length 120 epochs
* **Reason (v1 T12):** with 100 epochs, several runs reached their best validation Dice at the very end of the
  budget, i.e. they were still improving: on Kvasir-SEG U-Net (best epoch 93.0 on average), Attention U-Net
  (91.7), QENAS without SynFlow (98.3; the 3.0 search space) and QENAS without ZiCo (88.7); on IDRiD Attention
  U-Net (98.0). QENAS 2.1 itself peaked at 65–75. 120 epochs give the late learners room without changing the
  schedule's shape; a longer run cannot silently overfit, because the best checkpoint is restored and D25 stops on
  overfitting. The same budget applies to the baselines (fair comparison).
* **Alternative:** 150 epochs. **Rejected** for compute: 120 covers the observed late peaks (≤ 99) with a 20 %
  margin.

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
* The foreground IoU, the test loss, the per-image FLOPs and the inference time per image are recorded too.
  Per-epoch validation histories also contain sensitivity and HD95.
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
  convention makes QENAS's FLOPs directly comparable with the Mixed-GGNAS tables. nnU-Net's network is counted
  with the same code (D46).
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
* **Evidence (v1 T16, 3 seeds each):** with a random frame split (392 / 98 / 122) the test Dice rises from 69.76 %
  to 90.05 % for QENAS (+20.29 pp) and from 48.28 % to 76.69 % for U-Net (+28.41 pp). The random split measures
  mostly the leakage, so it is **not adopted**; frame-split results may only be reported as a labelled comparison
  with the Mixed-GGNAS protocol. (The frame-split switch existed only for that v1 study and is no longer in the
  code.) This question is answered and not repeated in study v2 (D48).
* PNG copies are used (the TIFs are not decodable by Pillow and contain the same images). The PNG masks have
  anti-aliasing artefacts, so they are binarised at > 127.

### D36 — Preprocessing
* RGB with 3 channels for all datasets (grey images are replicated).
* Images: anti-aliased bilinear resize. Masks: binarised at native resolution, union of mask files, then
  nearest-neighbour resize.
* Fixed sizes without letterboxing, as in Mixed-GGNAS (BUSI/CVC/Kvasir 256 × 256).
* Per-channel normalisation statistics from the training split only.

### D37 — Augmentation (training split only)
* Horizontal flip (0.5); random affine with p = 0.5 (rotation ±15°, scale 0.9–1.1, translation ±5 %; bilinear for
  images, nearest for masks, with identical parameters); brightness/contrast ±10 %.
* Vertical flips only for CVC-ClinicDB and Kvasir-SEG: endoscopy has no canonical orientation, whereas
  ultrasound depth and fundus anatomy do.
* The random stream is derived from (seed, epoch, index), so it is reproducible and identical after a resume.
  The two ZiCo batches are drawn from this same stream (epoch 0).

### D38 — Test isolation by construction
* The test split is listed in `split.json`, but its pixels are never loaded until the test stage (`stage_test`,
  for nnU-Net `stage_nnunet_test`). That stage refuses to run unless all previous stages are complete, evaluates
  once, and records `test_evaluated` (a re-run never re-evaluates).
* The search uses two training batches only. Normalisation statistics, augmentation, early stopping, checkpoint
  selection, candidate selection and α selection use train/validation only.
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
  epoch. Successive halving additionally saves every candidate's history after each run, so a resumed candidate
  stage never retrains an eliminated candidate.
* A configuration fingerprint guards every resume. Re-running a finished experiment reports it and never
  modifies it.
* **Pruning:** once a stage is done, the checkpoints it alone used are deleted (search state after the search,
  eliminated candidates after each rung and all candidate runs after the candidate stage, α runs after the
  loss-weight stage), because their results are stored in the stage's JSON/CSV outputs. A finished experiment
  keeps only `checkpoints/final/best.pt`. nnU-Net's working folder (exported images, preprocessed data, weights)
  is deleted when its job, including the external validation, has finished; its plans and logs are kept.
* **Timer:** at the end of every run, the time from start to end (the sum of all stage times, which includes
  earlier sessions of a resumed run) and the wall-clock time of the current session are logged, stored as
  `total_seconds` in `state.json`, and appended to `summary.txt`.

### D40 — Determinism
* Seeds for Python, NumPy and PyTorch; `torch.use_deterministic_algorithms(True, warn_only=True)`; deterministic
  cuDNN; seeded per-epoch data order and augmentation.
* Bit-exactness is verified on CPU. On GPU some kernels (e.g. the transposed-convolution backward) may differ in
  the last bits. nnU-Net's own data loading is not seeded by nnU-Net (D47).

### D41 — Result folders
* Every experiment is identified by (dataset, method, seed) and lives in `results/<DATASET>/<method>/seed_<s>/`.
  Re-running the same command resumes or reports that directory; a changed configuration is refused.
* Study v1 results: GitHub branch `kaggle-study` (local copy `Old Results/study_v1/`); study v2 results: branch
  `kaggle-study-v2`. Result folders are not committed to `main`.

### D42 — Kvasir-SEG: official split
* **Decision:** Kvasir-SEG (Jha et al., MMM 2020; 1000 polyp images) is the fourth dataset, at 256 × 256 like
  CVC-ClinicDB, with vertical flips (endoscopy). The authors publish a train/val split "for fair comparison":
  `val.txt` (120) is the test split, and 20 % of `train.txt` (880) is the validation split → 704 / 176 / 120.
* JPEG masks (≤ 2.1 % grey edge pixels) are binarised at > 127. Pixel hashing found no duplicates. There are no
  patient identifiers, so the split is image-level (as for BUSI).
* **Reason:** the most widely used polyp benchmark with an official split. It gives the polyp results more
  statistical power than CVC-ClinicDB's 6 test sequences, and it enables the external validation on
  CVC-ClinicDB.

### D43 — The split does not depend on the run seed
* **Decision:** `data.split_seed = 42` is fixed; the run seed (`--seed`) changes only the search, initialisation,
  data order and augmentation.
* **Reason:** repeated runs measure the variability of the method, and every method and seed is tested on the
  same images, which makes paired per-image tests possible (D45) — also between study v1 and v2.

### D44 — Evaluation study: conditions and baselines
* **Decision:** the study conditions (STUDY.md) are defined in one registry, `qenas/methods.py`, and run by
  `study.py`; `main.py` runs only QENAS.
  * The inheriting conditions (`qenas_fixed_alpha`, `qenas_random_pick`, `qenas_zico_pick`) copy QENAS' search,
    16 candidates and successive-halving runs of the same dataset and seed and change only one rule.
  * Random search keeps NSGA-II's evaluation budget and objectives.
* **Baselines:** U-Net (31.04 M) and Attention U-Net (8.73 M), the exact variants of the Mixed-GGNAS tables
  (identified by parameters and FLOPs, `tests/test_flops.py`), trained with QENAS' own protocol for its selected
  architecture (α grid on validation → 120-epoch final training → single test); and nnU-Net with its own pipeline
  (D46). Mixed-GGNAS itself cannot be re-run: its repository has no searched architectures, weights or training
  instructions, and its search takes about 26 h per dataset. Its published numbers (own splits) are reported for
  reference.
* **External validation:** the Kvasir-SEG final model is applied unchanged, with its own preprocessing and without
  TTA, to all 612 CVC-ClinicDB frames.

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
* The v1 → v2 comparison (T14) pairs the same test images; each study's per-image Dice is averaged over its own
  seeds.

### D46 — nnU-Net as a baseline (study only)
* **Decision:** nnU-Net v2 (`nnunetv2==2.8.1`) is run with its own, unmodified self-configuring pipeline: dataset
  fingerprint, the default `ExperimentPlanner` plans (`nnUNetPlans`), the `2d` configuration, its preprocessing
  (per-image z-score normalisation), augmentation, Dice + cross-entropy loss with deep supervision and SGD with a
  poly learning rate (`qenas/study/nnunet.py`).
* **Same data:** fold 0 of nnU-Net is the QENAS train/validation split (`splits_final.json`); the images are the
  QENAS preprocessed arrays at the QENAS resolution, exported as 3-channel PNGs (`NaturalImage2DIO`, channel names
  R, G, B as in nnU-Net's official 2D example Dataset120_RoadSegmentation); the test images are exported only in
  the test stage. Predictions are scored with the study's metric code at the network resolution; parameters,
  FLOPs (D32) and latency are measured on nnU-Net's network.
* **Reason:** nnU-Net is the standard strong baseline of medical image segmentation; a Q1 evaluation without it
  is incomplete (v1 report, threats to validity). Running it on the identical split and images makes the paired
  per-image tests possible.
* **Installation on Kaggle:** every nnunetv2 2.x release excludes torch 2.9.*, which can be the preinstalled
  version. nnunetv2 is therefore installed with `--no-deps` and its dependencies with pinned versions (README);
  its 2D networks do not use the 3D convolutions that the torch exclusion concerns. The integration is tested end
  to end (`tests/test_nnunet.py`, real nnU-Net with the 1-epoch trainer).

### D47 — nnU-Net budget and inference
* **Decision:** the official trainer variant `nnUNetTrainer_100epochs` (100 epochs × 250 iterations; default 1000
  epochs); the final checkpoint (nnU-Net's default); no test-time augmentation in the main tables (as every
  method); nnU-Net's default mirroring TTA is reported in addition (T15). `torch.compile` is disabled
  (`nnUNet_compile=f`, speed only) and checkpoints are written every 5 epochs instead of 50 (checkpoint frequency
  only).
* **Reason:** 1000 epochs take ten times as long as 100 (an estimated 10 h or more per run on a Tesla T4) — 12
  such runs are not feasible on Kaggle. 100 epochs × 250 iterations is at least the GPU time of the complete QENAS
  pipeline (search + candidates + α grid + final training: 33–106 min per run in v1, T13), and nnU-Net's poly schedule anneals over the whole
  shortened run, so it is a complete, if shorter, nnU-Net training. Reported honestly as a limitation.
* **Seeds:** nnU-Net does not seed its data loading; the run seed sets the global `random`/`numpy`/`torch`
  generators, so the three runs are independent repetitions but not bit-reproducible.

### D48 — Study v2: only what changed or was never tested
* **Decision:** study v2 runs QENAS 3.0 with **new seeds (45, 46, 47)** on all four datasets, re-runs U-Net and
  Attention U-Net under the v2 protocol (120 epochs, no TTA; needed for a fair comparison), adds nnU-Net, the
  loss-coefficient ablation (D49), the selection ablations for the new selection (BUSI, Kvasir-SEG), random search
  with the new objectives, and the external validation of all four methods. 84 jobs + 12 external evaluations.
* **Not repeated (answered by v1):** the CVC-ClinicDB leakage study (D35), the SynFlow analyses and objective
  relations (D10, D14), the 3-objective non-dominated shares (D16) and the proxy validity inside the elite
  (D18; v2 reports it for the 16 candidates at 5 epochs, T9).
* **Reason:** the v2 changes were motivated by the v1 runs (seeds 42–44); new seeds keep the evaluation of the
  changes independent of the runs that motivated them. The split is unchanged, so v1 and v2 are compared on the
  same test images (T14).

### D49 — Loss-coefficient ablation: α grid vs fixed α = 0.5
* **Decision:** `qenas_fixed_alpha` inherits the QENAS experiment of the same dataset and seed (search, candidates,
  successive halving and therefore the identical selected architecture) and trains it with α = β = 0.5 without the
  grid. QENAS and the ablation differ only in α; on all four datasets, 3 seeds.
* **What is reported (T3, T4):** the chosen α, the validation gain of the grid, the test Dice of both, the paired
  test on the same images, and the time the grid costs (and its share of a run).
* **Decision rule for the next version:** if the grid improves the test Dice significantly (Holm p < 0.05) on no
  dataset, it is removed and α = 0.5 becomes the protocol; if it helps on some datasets, it is kept and the cost is
  reported.

### D50 — Random-candidate ablation: the seed includes the dataset
* **Decision:** `qenas_random_pick` draws its candidate with `derive_seed(seed, "random-candidate-<DATASET>")`.
* **Reason (v1 defect):** v1 derived the index from the seed alone, so all datasets of a seed drew the same index;
  on Kvasir-SEG it coincided with the QENAS choice in all three seeds and the ablation was identical to QENAS (v1
  T3/T4, Δ = 0). Including the dataset makes the draws independent across datasets (unit-tested).

---

## Changes from QENAS 2.1 (study v1) and why

| area | QENAS 2.1 | QENAS 3.0 | reason |
|---|---|---|---|
| objectives | params, SynFlow, ZiCo | **params, ZiCo** | SynFlow redundant, size-biased, removal better on Kvasir-SEG (D10) |
| candidates | ≤ 8 of the final front | **16** of the final population (front + NSGA-II order) | fixed budget for successive halving (D18) |
| candidate training | 8 × 20 epochs | **successive halving 16 × 5 → 8 × 10 → 4 × 20** (same 160 epochs) | twice the candidates for the same compute (D20) |
| selection criterion | ΔDice with noise tolerance + gates | **mean validation Dice of the last 5 epochs** + gates | ΔDice picked the best-ending candidate in 3/12 runs (D21) |
| final training | 100 epochs | **120 epochs** | late best epochs (88–98) in v1 (D29) |
| test-time augmentation | flip TTA if it helps on validation | **none** | effect within ±0.5 pp, doubles latency (D27) |
| random-pick ablation seed | seed only | seed + dataset | v1 defect (D50) |
| study | v1: 78 jobs, seeds 42–44 | v2: 84 jobs, seeds 45–47, + nnU-Net, + α ablation | only what changed or was never tested (D46–D49) |
