# QENAS evaluation study — analysis

Completed experiments: 78. Regenerate with `python study.py analyse` (add `--data-root` for the qualitative figures).

| file | question |
|---|---|
| T1_main_results | RQ1: QENAS vs U-Net vs Attention U-Net on identical splits (+ published Mixed-GGNAS) |
| T2_significance_main | RQ1: paired Wilcoxon tests on the same test images, Holm-corrected |
| T3_ablations, T4_significance_ablations | RQ2: contribution of ZiCo, SynFlow and learning-capability selection |
| T5_search_vs_random | RQ3: NSGA-II vs random search, same evaluation budget |
| T6–T8 | objective relations, block effects, non-dominated share (D14, D16) |
| T9_proxies_vs_quality | zero-cost proxies vs trained validation Dice |
| T10_selection_quality | rank of the learning-capability choice |
| T11–T13 | TTA decisions, training behaviour, compute time |
| T14, T15 | RQ4: external validation Kvasir-SEG → CVC-ClinicDB |
| T16_cvc_leakage | RQ5: score inflation of a random frame split on CVC-ClinicDB |

Figures: `figures/per_image_dice_boxplots.png`, `figures/search_vs_random.png`, `figures/proxies_vs_val_dice.png`, `figures/accuracy_vs_parameters.png`, `figures/ablations.png`, `figures/qualitative_BUSI.png`, `figures/qualitative_CVC.png`, `figures/qualitative_IDRID.png`, `figures/qualitative_KVASIR.png`
