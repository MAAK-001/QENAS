### RQ5 — Paired comparison on the 612 CVC-ClinicDB frames

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG → CVC-ClinicDB | QENAS vs U-Net | 612 | 79.64 | 69.24 | 10.4 | [+8.75, +12.08] | 3.633e-28 | 1.09e-27 | 402 / 206 |
| Kvasir-SEG → CVC-ClinicDB | QENAS vs Attention U-Net | 612 | 79.64 | 70.26 | 9.383 | [+7.67, +11.14] | 2.794e-13 | 5.587e-13 | 342 / 265 |
| Kvasir-SEG → CVC-ClinicDB | QENAS vs nnU-Net | 612 | 79.64 | 81.54 | -1.898 | [-3.01, -0.79] | 8.424e-11 | 8.424e-11 | 221 / 391 |
