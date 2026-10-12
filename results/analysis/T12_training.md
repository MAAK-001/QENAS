### Final training: best epoch, early stopping, generalisation — D25, D29

| Dataset | Method | Runs | Best epoch | Epochs run | Stopped early (overfitting) | Train Dice (%) | Val Dice (%) |
|---|---|---|---|---|---|---|---|
| BUSI | QENAS | 3 | 65.0 ± 7.5 | 92.7 ± 3.1 | 3/3 | 89.95 ± 1.73 | 79.65 ± 0.77 |
| BUSI | QENAS w/o ZiCo | 3 | 64.3 ± 10.7 | 87.7 ± 14.3 | 2/3 | 90.15 ± 1.47 | 79.89 ± 0.25 |
| BUSI | QENAS w/o SynFlow | 3 | 57.7 ± 7.5 | 85.3 ± 15.0 | 2/3 | 87.40 ± 0.89 | 80.74 ± 0.12 |
| BUSI | QENAS, random Pareto candidate | 3 | 79.0 ± 11.4 | 98.3 ± 2.9 | 1/3 | 91.11 ± 1.61 | 80.04 ± 0.29 |
| BUSI | QENAS, zero-shot pick (max ZiCo) | 3 | 56.3 ± 11.9 | 79.7 ± 16.5 | 3/3 | 87.74 ± 1.76 | 79.85 ± 0.43 |
| BUSI | U-Net | 3 | 82.0 ± 14.9 | 96.7 ± 5.8 | 1/3 | 89.92 ± 2.45 | 74.72 ± 0.49 |
| BUSI | Attention U-Net | 3 | 72.0 ± 10.6 | 94.3 ± 7.4 | 2/3 | 87.29 ± 0.90 | 74.76 ± 0.20 |
| CVC-ClinicDB | QENAS | 3 | 66.3 ± 12.7 | 99.3 ± 1.2 | 1/3 | 94.33 ± 1.22 | 74.22 ± 2.14 |
| CVC-ClinicDB | U-Net | 3 | 58.3 ± 9.3 | 100.0 ± 0.0 | 0/3 | 86.15 ± 0.83 | 63.95 ± 2.03 |
| CVC-ClinicDB | Attention U-Net | 3 | 56.3 ± 15.6 | 87.7 ± 11.6 | 2/3 | 86.25 ± 5.93 | 68.23 ± 5.38 |
| IDRiD (optic disc) | QENAS | 3 | 69.3 ± 42.7 | 98.0 ± 3.5 | 1/3 | 96.78 ± 1.63 | 95.84 ± 0.97 |
| IDRiD (optic disc) | U-Net | 3 | 85.3 ± 6.7 | 100.0 ± 0.0 | 0/3 | 93.46 ± 0.33 | 92.82 ± 0.90 |
| IDRiD (optic disc) | Attention U-Net | 3 | 98.0 ± 3.5 | 100.0 ± 0.0 | 0/3 | 94.19 ± 0.20 | 92.47 ± 0.94 |
| Kvasir-SEG | QENAS | 3 | 74.7 ± 17.2 | 94.3 ± 9.8 | 1/3 | 93.29 ± 0.94 | 87.79 ± 0.37 |
| Kvasir-SEG | QENAS w/o ZiCo | 3 | 88.7 ± 4.2 | 100.0 ± 0.0 | 0/3 | 94.37 ± 1.08 | 88.55 ± 0.68 |
| Kvasir-SEG | QENAS w/o SynFlow | 3 | 98.3 ± 1.2 | 100.0 ± 0.0 | 0/3 | 94.84 ± 0.53 | 88.82 ± 0.38 |
| Kvasir-SEG | QENAS, random Pareto candidate | 3 | 74.7 ± 17.2 | 94.3 ± 9.8 | 1/3 | 93.29 ± 0.94 | 87.79 ± 0.37 |
| Kvasir-SEG | QENAS, zero-shot pick (max ZiCo) | 3 | 85.0 ± 14.8 | 99.7 ± 0.6 | 1/3 | 94.16 ± 0.77 | 88.56 ± 0.85 |
| Kvasir-SEG | U-Net | 3 | 93.0 ± 3.5 | 100.0 ± 0.0 | 0/3 | 93.45 ± 1.08 | 86.66 ± 0.69 |
| Kvasir-SEG | Attention U-Net | 3 | 91.7 ± 7.4 | 100.0 ± 0.0 | 0/3 | 94.22 ± 1.21 | 88.22 ± 0.49 |
| CVC-ClinicDB, frame-level split | QENAS | 3 | 89.3 ± 16.8 | 100.0 ± 0.0 | 0/3 | 94.73 ± 0.23 | 90.76 ± 0.24 |
| CVC-ClinicDB, frame-level split | U-Net | 3 | 95.0 ± 4.4 | 100.0 ± 0.0 | 0/3 | 84.23 ± 0.51 | 78.58 ± 1.00 |
