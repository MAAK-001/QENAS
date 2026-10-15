### RQ3 — QENAS vs each selection ablation on the same test images

| Dataset | Comparison | Images | Dice QENAS (%) | Dice other (%) | Δ Dice (pp) | 95 % CI (pp) | p (Wilcoxon) | p (Holm) | Images better / worse |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS vs QENAS, random candidate | 129 | 79.3 | 79.68 | -0.3771 | [-1.51, +0.91] | 0.02938 | 0.05875 | 50 / 78 |
| BUSI | QENAS vs QENAS, zero-shot pick (max ZiCo) | 129 | 79.3 | 79.8 | -0.4931 | [-1.66, +0.83] | 0.03329 | 0.05875 | 51 / 78 |
| Kvasir-SEG | QENAS vs QENAS, random candidate | 120 | 84.98 | 83.62 | 1.354 | [-0.03, +2.66] | 0.002736 | 0.005471 | 75 / 45 |
| Kvasir-SEG | QENAS vs QENAS, zero-shot pick (max ZiCo) | 120 | 84.98 | 84.02 | 0.9633 | [-0.51, +2.48] | 0.0432 | 0.0432 | 71 / 49 |
