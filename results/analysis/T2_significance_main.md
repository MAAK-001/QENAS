### RQ1 — Paired comparison on the same test images (per-image Dice averaged over seeds)

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS vs U-Net | 129 | 79.3 | 72.27 | 7.036 | [+4.65, +9.73] | 4.968e-06 | 1.49e-05 | 78 / 51 |
| BUSI | QENAS vs Attention U-Net | 129 | 79.3 | 74.63 | 4.679 | [+2.33, +7.29] | 0.12 | 0.2278 | 67 / 62 |
| BUSI | QENAS vs nnU-Net | 129 | 79.3 | 78.81 | 0.4974 | [-1.18, +2.36] | 0.1139 | 0.2278 | 54 / 75 |
| CVC-ClinicDB | QENAS vs U-Net | 121 | 69.59 | 44.12 | 25.48 | [+22.14, +28.83] | 8.002e-21 | 2.401e-20 | 116 / 5 |
| CVC-ClinicDB | QENAS vs Attention U-Net | 121 | 69.59 | 58.75 | 10.84 | [+7.68, +14.00] | 2.045e-07 | 2.65e-07 | 79 / 42 |
| CVC-ClinicDB | QENAS vs nnU-Net | 121 | 69.59 | 76.82 | -7.23 | [-10.10, -4.37] | 1.325e-07 | 2.65e-07 | 38 / 83 |
| IDRiD (optic disc) | QENAS vs U-Net | 27 | 95.7 | 91.74 | 3.962 | [+1.56, +6.85] | 0.0004272 | 0.001282 | 22 / 5 |
| IDRiD (optic disc) | QENAS vs Attention U-Net | 27 | 95.7 | 92.46 | 3.247 | [+0.89, +6.47] | 0.003247 | 0.006494 | 20 / 7 |
| IDRiD (optic disc) | QENAS vs nnU-Net | 27 | 95.7 | 95.97 | -0.2608 | [-1.00, +0.57] | 0.08149 | 0.08149 | 10 / 17 |
| Kvasir-SEG | QENAS vs U-Net | 120 | 85.39 | 81.51 | 3.877 | [+1.64, +6.26] | 0.0002435 | 0.0004869 | 78 / 42 |
| Kvasir-SEG | QENAS vs Attention U-Net | 120 | 85.39 | 84.2 | 1.182 | [-0.80, +3.22] | 0.7335 | 0.7335 | 56 / 64 |
| Kvasir-SEG | QENAS vs nnU-Net | 120 | 85.39 | 87.44 | -2.054 | [-3.79, -0.35] | 2.987e-05 | 8.96e-05 | 36 / 84 |
