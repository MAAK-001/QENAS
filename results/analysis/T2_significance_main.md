### RQ1 — Paired comparison on the same test images (per-image Dice averaged over seeds)

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS vs U-Net | 129 | 79.87 | 73.04 | 6.837 | [+4.58, +9.31] | 2.087e-07 | 4.174e-07 | 88 / 41 |
| BUSI | QENAS vs Attention U-Net | 129 | 79.87 | 74.47 | 5.404 | [+3.09, +7.82] | 7.33e-06 | 7.33e-06 | 84 / 45 |
| CVC-ClinicDB | QENAS vs U-Net | 121 | 69.76 | 48.28 | 21.48 | [+17.81, +25.32] | 1.128e-18 | 2.256e-18 | 109 / 12 |
| CVC-ClinicDB | QENAS vs Attention U-Net | 121 | 69.76 | 57.43 | 12.34 | [+9.35, +15.60] | 1.909e-13 | 1.909e-13 | 104 / 17 |
| IDRiD (optic disc) | QENAS vs U-Net | 27 | 95.91 | 90.84 | 5.076 | [+2.03, +9.05] | 1.347e-05 | 2.694e-05 | 21 / 6 |
| IDRiD (optic disc) | QENAS vs Attention U-Net | 27 | 95.91 | 93.26 | 2.651 | [+1.14, +4.60] | 0.0001253 | 0.0001253 | 22 / 5 |
| Kvasir-SEG | QENAS vs U-Net | 120 | 82.33 | 81.43 | 0.8953 | [-1.51, +3.41] | 0.8381 | 0.8381 | 61 / 59 |
| Kvasir-SEG | QENAS vs Attention U-Net | 120 | 82.33 | 83.93 | -1.598 | [-3.77, +0.50] | 0.003805 | 0.007611 | 46 / 74 |
