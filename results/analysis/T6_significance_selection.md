### RQ3 — QENAS vs each selection ablation on the same test images

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS vs QENAS, random candidate | 129 | 79.3 | 80.43 | -1.131 | [-2.29, +0.12] | 0.0009557 | 0.001911 | 47 / 81 |
| BUSI | QENAS vs QENAS, zero-shot pick (max ZiCo) | 129 | 79.3 | 79.97 | -0.67 | [-2.07, +0.79] | 0.04408 | 0.04408 | 51 / 78 |
| Kvasir-SEG | QENAS vs QENAS, random candidate | 120 | 85.39 | 84.04 | 1.345 | [-0.19, +2.90] | 0.01908 | 0.03817 | 73 / 47 |
| Kvasir-SEG | QENAS vs QENAS, zero-shot pick (max ZiCo) | 120 | 85.39 | 84.9 | 0.4828 | [-0.86, +1.82] | 0.5009 | 0.5009 | 61 / 59 |
