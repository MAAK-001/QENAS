# QENAS evaluation study v2 — analysis

Completed experiments: 64. Regenerate with `python study.py analyse` (add `--data-root` for the qualitative figures and `--v1-results` for the v1 → v2 comparison).

| file | question |
|---|---|
| T1_main_results | RQ1: QENAS vs U-Net, Attention U-Net, nnU-Net on identical splits (+ published Mixed-GGNAS) |
| T2_significance_main | RQ1: paired Wilcoxon tests on the same test images, Holm-corrected |
| T3_loss_coefficients, T4 | RQ2: does the α grid help, and what does it cost? |
| T5_selection_ablations, T6 | RQ3: learning-capability selection vs random candidate vs max ZiCo |
| T7_search_vs_random | RQ4: NSGA-II vs random search, same evaluation budget |
| T8_successive_halving | consistency of the rungs; choices of the earlier rules |
| T9_proxies_vs_quality | objectives vs early trained validation Dice |
| T10, T11 | training behaviour, compute time |
| T12, T13 | RQ5: external validation Kvasir-SEG → CVC-ClinicDB |
| T14 | study v1 vs v2 (needs `--v1-results`) |
| T15_nnunet_tta | nnU-Net with its default mirroring TTA (supplementary) |

Figures: `figures/per_image_dice_boxplots.png`, `figures/accuracy_vs_parameters.png`, `figures/search_vs_random.png`, `figures/proxies_vs_val_dice.png`, `figures/ablation_loss_coefficients.png`, `figures/ablation_selection.png`, `figures/qualitative_BUSI.png`, `figures/qualitative_CVC.png`, `figures/qualitative_IDRID.png`, `figures/qualitative_KVASIR.png`
