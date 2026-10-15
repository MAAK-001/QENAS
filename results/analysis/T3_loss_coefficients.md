### RQ2 — Loss-coefficient search (α grid) vs fixed α = 0.5: validation gain, test effect and cost (same architecture, paired by seed)

| Dataset | Seeds | α chosen | Val gain of the grid (pp) | Test Dice, grid (%) | Test Dice, α = 0.5 (%) | Test Δ grid − fixed (pp) | Grid time (min) | Grid share of run time (%) |
|---|---|---|---|---|---|---|---|---|
| BUSI | 3 | 0.5, 0.5, 0.2 | 0.01 ± 0.02 | 79.30 ± 0.90 | 78.83 ± 0.20 | 0.47 ± 0.82 | 16.5 ± 6.3 | 20.8 ± 0.8 |
| CVC-ClinicDB | 3 | 0.2, 0.4, 0.4 | 1.79 ± 1.34 | 69.59 ± 2.45 | 68.29 ± 4.15 | 1.30 ± 6.35 | 20.7 ± 7.3 | 23.0 ± 4.0 |
| IDRiD (optic disc) | 3 | 0.2, 0.6, 0.2 | 0.12 ± 0.06 | 95.70 ± 0.72 | 96.13 ± 0.15 | -0.42 ± 0.61 | 3.7 ± 0.7 | 13.3 ± 2.2 |
| Kvasir-SEG | 3 | 0.4, 0.5, 0.5 | 0.07 ± 0.12 | 84.98 ± 0.81 | 84.85 ± 0.83 | 0.13 ± 0.22 | 28.7 ± 2.8 | 19.2 ± 1.4 |
