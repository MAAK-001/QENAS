### RQ5 — Paired comparison on the 612 CVC-ClinicDB frames

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG → CVC-ClinicDB | QENAS vs U-Net | 612 | 78.63 | 69.31 | 9.317 | [+7.81, +10.86] | 1.666e-24 | 4.997e-24 | 391 / 219 |
| Kvasir-SEG → CVC-ClinicDB | QENAS vs Attention U-Net | 612 | 78.63 | 70.72 | 7.909 | [+6.40, +9.45] | 1.158e-12 | 1.158e-12 | 345 / 264 |
| Kvasir-SEG → CVC-ClinicDB | QENAS vs nnU-Net | 612 | 78.63 | 81.32 | -2.695 | [-3.75, -1.64] | 8.215e-17 | 1.643e-16 | 189 / 423 |
