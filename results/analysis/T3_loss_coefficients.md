### RQ2 — Loss-coefficient search (α grid) vs fixed α = 0.5: validation gain, test effect and cost (same architecture, paired by seed)

| Dataset | Seeds | α chosen | Val gain of the grid (pp) | Test Dice, grid (%) | Test Dice, α = 0.5 (%) | Test Δ grid − fixed (pp) | Grid time (min) | Grid share of run time (%) |
|---|---|---|---|---|---|---|---|---|
| BUSI | 2 | 0.5, 0.5 | 0.00 ± 0.00 | 78.80 ± 0.28 | 78.80 ± 0.28 | 0.00 ± 0.00 | 12.9 ± 0.4 | 20.3 ± 0.3 |
| CVC-ClinicDB | 2 | 0.2, 0.4 | 1.02 ± 0.21 | 68.31 ± 1.46 | 69.58 ± 4.97 | -1.27 ± 6.42 | 18.0 ± 7.9 | 22.5 ± 5.6 |
| IDRiD (optic disc) | 3 | 0.2, 0.6, 0.2 | 0.12 ± 0.06 | 95.70 ± 0.72 | 96.13 ± 0.15 | -0.42 ± 0.61 | 3.7 ± 0.7 | 13.3 ± 2.2 |
| Kvasir-SEG | 2 | 0.4, 0.5 | 0.10 ± 0.15 | 85.39 ± 0.55 | 85.20 ± 0.82 | 0.19 ± 0.27 | 28.0 ± 3.7 | 18.4 ± 0.3 |
