### Flip TTA decided on validation — DESIGN_DECISIONS D27

| Dataset | Method | Runs | TTA chosen | Val Dice without TTA (%) | Val Dice with TTA (%) |
|---|---|---|---|---|---|
| BUSI | QENAS | 3 | 1/3 | 79.65 ± 0.77 | 79.41 ± 1.12 |
| BUSI | QENAS w/o ZiCo | 3 | 1/3 | 79.89 ± 0.25 | 79.80 ± 0.53 |
| BUSI | QENAS w/o SynFlow | 3 | 2/3 | 80.74 ± 0.12 | 80.66 ± 0.25 |
| BUSI | QENAS, random Pareto candidate | 3 | 0/3 | 80.04 ± 0.29 | 79.57 ± 0.08 |
| BUSI | QENAS, zero-shot pick (max ZiCo) | 3 | 1/3 | 79.85 ± 0.43 | 79.63 ± 0.32 |
| BUSI | U-Net | 3 | 2/3 | 74.72 ± 0.49 | 74.86 ± 0.47 |
| BUSI | Attention U-Net | 3 | 3/3 | 74.76 ± 0.20 | 75.11 ± 0.47 |
| CVC-ClinicDB | QENAS | 3 | 1/3 | 74.22 ± 2.14 | 74.54 ± 0.87 |
| CVC-ClinicDB | U-Net | 3 | 1/3 | 63.95 ± 2.03 | 63.56 ± 1.39 |
| CVC-ClinicDB | Attention U-Net | 3 | 1/3 | 68.23 ± 5.38 | 67.67 ± 3.90 |
| IDRiD (optic disc) | QENAS | 3 | 2/3 | 95.84 ± 0.97 | 95.82 ± 0.85 |
| IDRiD (optic disc) | U-Net | 3 | 3/3 | 92.82 ± 0.90 | 92.96 ± 1.01 |
| IDRiD (optic disc) | Attention U-Net | 3 | 2/3 | 92.47 ± 0.94 | 92.46 ± 1.32 |
| Kvasir-SEG | QENAS | 3 | 2/3 | 87.79 ± 0.37 | 88.22 ± 0.85 |
| Kvasir-SEG | QENAS w/o ZiCo | 3 | 2/3 | 88.55 ± 0.68 | 89.01 ± 0.39 |
| Kvasir-SEG | QENAS w/o SynFlow | 3 | 3/3 | 88.82 ± 0.38 | 89.14 ± 0.48 |
| Kvasir-SEG | QENAS, random Pareto candidate | 3 | 2/3 | 87.79 ± 0.37 | 88.22 ± 0.85 |
| Kvasir-SEG | QENAS, zero-shot pick (max ZiCo) | 3 | 3/3 | 88.56 ± 0.85 | 88.95 ± 0.75 |
| Kvasir-SEG | U-Net | 3 | 3/3 | 86.66 ± 0.69 | 87.07 ± 0.67 |
| Kvasir-SEG | Attention U-Net | 3 | 3/3 | 88.22 ± 0.49 | 88.85 ± 0.42 |
| CVC-ClinicDB, frame-level split | QENAS | 3 | 2/3 | 90.76 ± 0.24 | 90.81 ± 0.30 |
| CVC-ClinicDB, frame-level split | U-Net | 3 | 3/3 | 78.58 ± 1.00 | 79.21 ± 1.07 |
