### RQ4 — Paired comparison on the 612 CVC-ClinicDB frames

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG → CVC-ClinicDB | QENAS vs U-Net | 612 | 75.92 | 68.52 | 7.408 | [+5.98, +8.89] | 3.064e-12 | 6.128e-12 | 340 / 266 |
| Kvasir-SEG → CVC-ClinicDB | QENAS vs Attention U-Net | 612 | 75.92 | 70.94 | 4.989 | [+3.62, +6.42] | 0.0117 | 0.0117 | 274 / 332 |
