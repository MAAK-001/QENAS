### RQ2 — QENAS vs each ablation on the same test images

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS vs QENAS w/o ZiCo | 129 | 79.87 | 78.48 | 1.396 | [+0.12, +2.78] | 0.1602 | 0.3718 | 67 / 62 |
| BUSI | QENAS vs QENAS w/o SynFlow | 129 | 79.87 | 79.06 | 0.8133 | [-0.42, +2.11] | 0.1239 | 0.3718 | 69 / 60 |
| BUSI | QENAS vs QENAS, random Pareto candidate | 129 | 79.87 | 79.66 | 0.2157 | [-0.94, +1.47] | 0.8884 | 0.8884 | 63 / 65 |
| BUSI | QENAS vs QENAS, zero-shot pick (max ZiCo) | 129 | 79.87 | 78.95 | 0.925 | [-0.13, +2.04] | 0.008067 | 0.03227 | 76 / 52 |
| Kvasir-SEG | QENAS vs QENAS w/o ZiCo | 120 | 82.33 | 84.12 | -1.788 | [-3.11, -0.54] | 0.0168 | 0.0336 | 49 / 71 |
| Kvasir-SEG | QENAS vs QENAS w/o SynFlow | 120 | 82.33 | 85.43 | -3.104 | [-4.75, -1.61] | 5.693e-05 | 0.0002277 | 43 / 77 |
| Kvasir-SEG | QENAS vs QENAS, random Pareto candidate | 120 | 82.33 | 82.33 | 0 | [+0.00, +0.00] | 1 | 1 | 0 / 0 |
| Kvasir-SEG | QENAS vs QENAS, zero-shot pick (max ZiCo) | 120 | 82.33 | 85.07 | -2.745 | [-4.59, -1.04] | 0.004491 | 0.01347 | 52 / 68 |
