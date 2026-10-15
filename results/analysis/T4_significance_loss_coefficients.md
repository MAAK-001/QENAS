### RQ2 — α grid vs fixed α = 0.5 on the same test images

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS vs QENAS, fixed α = 0.5 | 129 | 79.3 | 78.83 | 0.4748 | [-0.09, +1.07] | 0.01312 | 0.01312 | 81 / 47 |
| CVC-ClinicDB | QENAS vs QENAS, fixed α = 0.5 | 121 | 69.59 | 68.29 | 1.297 | [-0.19, +2.82] | 0.2919 | 0.2919 | 66 / 55 |
| IDRiD (optic disc) | QENAS vs QENAS, fixed α = 0.5 | 27 | 95.7 | 96.13 | -0.4213 | [-0.82, -0.10] | 0.006451 | 0.006451 | 6 / 21 |
| Kvasir-SEG | QENAS vs QENAS, fixed α = 0.5 | 120 | 84.98 | 84.85 | 0.1266 | [-0.40, +0.69] | 0.8958 | 0.8958 | 63 / 57 |
