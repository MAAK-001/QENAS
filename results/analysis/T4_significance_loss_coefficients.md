### RQ2 — α grid vs fixed α = 0.5 on the same test images

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS vs QENAS, fixed α = 0.5 | 129 | 79.3 | 78.8 | 0.5072 | [-0.12, +1.13] | 0.003577 | 0.003577 | 81 / 47 |
| CVC-ClinicDB | QENAS vs QENAS, fixed α = 0.5 | 121 | 69.59 | 69.58 | 0.01528 | [-1.63, +1.61] | 0.9742 | 0.9742 | 58 / 63 |
| IDRiD (optic disc) | QENAS vs QENAS, fixed α = 0.5 | 27 | 95.7 | 96.13 | -0.4213 | [-0.82, -0.10] | 0.006451 | 0.006451 | 6 / 21 |
| Kvasir-SEG | QENAS vs QENAS, fixed α = 0.5 | 120 | 85.39 | 85.2 | 0.1899 | [-0.60, +1.04] | 0.8958 | 0.8958 | 63 / 57 |
