### RQ1 — Paired comparison on the same test images (per-image Dice averaged over seeds)

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS vs U-Net | 129 | 79.3 | 72.75 | 6.558 | [+4.28, +9.12] | 5.874e-06 | 1.762e-05 | 80 / 49 |
| BUSI | QENAS vs Attention U-Net | 129 | 79.3 | 74.48 | 4.828 | [+2.55, +7.32] | 0.09123 | 0.1594 | 64 / 65 |
| BUSI | QENAS vs nnU-Net | 129 | 79.3 | 78.87 | 0.4343 | [-1.19, +2.28] | 0.0797 | 0.1594 | 53 / 76 |
| CVC-ClinicDB | QENAS vs U-Net | 121 | 69.59 | 48.77 | 20.82 | [+17.88, +23.74] | 7.577e-20 | 2.273e-19 | 115 / 6 |
| CVC-ClinicDB | QENAS vs Attention U-Net | 121 | 69.59 | 60.05 | 9.544 | [+6.56, +12.54] | 2.162e-07 | 2.366e-07 | 80 / 41 |
| CVC-ClinicDB | QENAS vs nnU-Net | 121 | 69.59 | 76.48 | -6.888 | [-9.86, -3.95] | 1.183e-07 | 2.366e-07 | 35 / 86 |
| IDRiD (optic disc) | QENAS vs U-Net | 27 | 95.7 | 91.9 | 3.809 | [+1.43, +6.77] | 0.0005357 | 0.001607 | 22 / 5 |
| IDRiD (optic disc) | QENAS vs Attention U-Net | 27 | 95.7 | 92.46 | 3.247 | [+0.89, +6.47] | 0.003247 | 0.006494 | 20 / 7 |
| IDRiD (optic disc) | QENAS vs nnU-Net | 27 | 95.7 | 95.9 | -0.1913 | [-0.95, +0.69] | 0.1004 | 0.1004 | 10 / 17 |
| Kvasir-SEG | QENAS vs U-Net | 120 | 84.98 | 81.33 | 3.645 | [+1.59, +5.87] | 0.000278 | 0.000556 | 78 / 42 |
| Kvasir-SEG | QENAS vs Attention U-Net | 120 | 84.98 | 84.2 | 0.7768 | [-1.02, +2.66] | 0.7041 | 0.7041 | 56 / 64 |
| Kvasir-SEG | QENAS vs nnU-Net | 120 | 84.98 | 87.31 | -2.331 | [-3.89, -0.81] | 1.463e-05 | 4.388e-05 | 36 / 84 |
