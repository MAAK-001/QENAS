# Positioning of QENAS in the literature

Every entry was checked against its paper; "–" means the paper does not report it.

## The closest lines of work

| method | task / data | search space | search | signal used by the search | final architecture chosen by |
|---|---|---|---|---|---|
| **TF-MOENAS** (Do & Luong, ICONIP 2021) | image classification (NAS benchmarks) | cell-based benchmark spaces | multi-objective EA, training-free | NTK condition number, number of linear regions, complexity | the proxy front |
| **E-TF-MOENAS** (Luong et al.; as described in MTF-PDNS, GECCO 2024) | image classification (NAS benchmarks) | cell-based benchmark spaces | NSGA-II, training-free | SynFlow, Jacobian covariance, #params/FLOPs as separate objectives | the proxy front |
| **MTF-PDNS** (GECCO 2024) | image classification (NAS benchmarks) | cell-based benchmark spaces | Pareto-dominance novelty search, training-free | several training-free metrics + complexity | the proxy front |
| **MedNAS / MSTF-NAS** (Wang et al., IEEE TEVC 2024) | medical image **classification** (MedMNIST, mammography) | DARTS-style cells with a multi-scale reduction cell | NSGA-II, training-free | two training-free indicators (e.g. SynFlow, jacob_cov), each combined with FLOPs | the proxy front |
| **AEG-cTFNAS** (Sun et al., MICCAI 2025) | medical image **segmentation** (ACDC, EndoVis 2018) | U-Net with searched normal / reduction / up-sampling cells | compact EA (truncated-normal model), alternating encoder/decoder game, training-free | FLOPs, jacob_cov, SynFlow (from MedNAS) | the search result (no supervised selection) |
| **ZiCo-BC** (Bhardwaj et al., ICCVW 2023) | semantic segmentation of natural images (Cityscapes) | HRNet / FFNet backbones and heads | NSGA-II with hardware latency, training-free | ZiCo (bias-corrected) computed with the training loss, latency | the proxy/latency front |
| **Mixed-GGNAS** (Hu et al., ESWA 2025) | medical image segmentation (CVC-ClinicDB, BUSI, CAMUS, IDRiD) | mixed: 4 manual blocks + DARTS block, U-shape (+ ViT) | GA with supervised training of individuals + DARTS; ≈ 26 h search on IDRiD | validation accuracy after training | the GA, after training |

## What QENAS adds (QENAS 3.0)

1. **A data- and label-aware zero-cost objective for medical segmentation.** The training-free medical NAS methods
   (MedNAS, AEG-cTFNAS) and the TF-MOENAS line use proxies that never see a label: SynFlow is data-free;
   jacob_cov and NTK/linear regions use images but no masks. QENAS optimises the parameter count against ZiCo,
   computed with the *segmentation loss on real training images and their masks* (official ZiCo code, unchanged).
   Study v1 measured that adding the data-free SynFlow (the objective of MedNAS / AEG-cTFNAS / E-TF-MOENAS) on top
   does not help: removing it was never significantly worse and significantly better on Kvasir-SEG (+3.10 pp), and
   ZiCo beat SynFlow as the second objective (DESIGN_DECISIONS D10).
2. **Training-free search in the mixed manual + DARTS search space** of Mixed-GGNAS, which so far was searched only
   with supervised training (≈ 26 h per dataset). QENAS uses the same blocks and genotypes, so the comparison
   isolates the search strategy. The search takes 5–10 min on a Tesla T4; a complete QENAS run 33–106 min (v1).
3. **Supervised selection by successive halving.** Zero-shot NAS methods return the proxy front or its best member.
   Study v1 showed that inside the search elite no proxy predicts trained quality, so QENAS trains 16 candidates of
   the final population with successive halving (16 × 5 → 8 × 10 → 4 × 20 epochs) and selects by end-of-run
   validation Dice with falsifiable pathology gates. Study v2 tests this against a random candidate and the
   zero-shot rule (max ZiCo).
4. **A leakage-controlled evaluation**, beyond what the compared papers report:
   * splits by leakage unit (CVC-ClinicDB by colonoscopy sequence), identical for all methods and seeds;
   * test sets touched once;
   * paired statistics on the same images;
   * nnU-Net on the identical split as the strong reference (study v2);
   * external validation (Kvasir-SEG → CVC-ClinicDB);
   * a measurement of how much the common random frame split inflates CVC-ClinicDB scores (v1: +20 to +28 pp).

## Honest limits for the paper

* Multi-objective, training-free NSGA-II itself is not new (E-TF-MOENAS, MedNAS). The novelty is the
  *combination* above, and it must be argued with the ablations, not assumed.
* In study v1 QENAS 2.1 beat U-Net and Attention U-Net clearly on BUSI, CVC-ClinicDB and IDRiD but not on
  Kvasir-SEG; v2 shows whether QENAS 3.0 changes that, and how it compares with nnU-Net.
* Mixed-GGNAS could not be re-run on our splits: its released code has no searched architectures, weights or
  training instructions, and its search costs ≈ 26 h per dataset. Its published numbers come from its own splits
  and are reported as such.
* AEG-cTFNAS (ACDC, EndoVis 2018) and MedNAS (classification) were evaluated on other datasets. A numerical
  comparison would require re-running them on these datasets.
* nnU-Net is trained for 100 instead of 1000 epochs (compute), with at least the GPU time of the whole QENAS
  pipeline.

## Sources

* Do, T., Luong, N.H.: Training-free multi-objective evolutionary neural architecture search via neural tangent
  kernel and number of linear regions. ICONIP 2021.
* Efficient Multi-Objective NAS via Pareto Dominance-based Novelty Search (MTF-PDNS), GECCO 2024,
  arXiv:2407.20656 (describes E-TF-MOENAS).
* Wang, Y. et al.: MedNAS: Multiscale Training-Free Neural Architecture Search for Medical Image Analysis.
  IEEE TEVC 2024, doi:10.1109/TEVC.2024.3352641.
* Sun, X., Wang, H., Song, P.-C.: Compact Training-free NAS with Alternating Evolution Game for Medical Image
  Segmentation. MICCAI 2025, doi:10.1007/978-3-032-05325-1_11.
* Bhardwaj, K. et al.: ZiCo-BC: A Bias Corrected Zero-Shot NAS for Vision Tasks. ICCV Workshops 2023.
* Li, G. et al.: ZiCo: Zero-shot NAS via Inverse Coefficient of Variation on Gradients. ICLR 2023.
* Hu, M. et al.: Mixed-GGNAS. Expert Systems with Applications 289 (2025) 128338.
* Isensee, F. et al.: nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
  Nature Methods 18 (2021) 203–211.
* Jamieson, K., Talwalkar, A.: Non-stochastic best arm identification and hyperparameter optimization (successive
  halving). AISTATS 2016.
* Jha, D. et al.: Kvasir-SEG: A Segmented Polyp Dataset. MMM 2020.
