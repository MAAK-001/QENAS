### Compute time per experiment (minutes) and hardware

| Dataset | Method | Runs | search (min) | candidates (min) | loss_weights (min) | final (min) | total (min) | GPU |
|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS | 3 | 6.9 ± 0.3 | 32.0 ± 2.4 | 20.2 ± 3.6 | 23.5 ± 3.8 | 83.3 ± 9.9 | Tesla T4 |
| BUSI | QENAS w/o ZiCo | 3 | 5.6 ± 0.7 | 30.8 ± 3.9 | 21.6 ± 8.5 | 23.1 ± 8.3 | 81.5 ± 20.9 | Tesla T4 |
| BUSI | QENAS w/o SynFlow | 3 | 6.0 ± 0.7 | 34.5 ± 6.2 | 19.8 ± 6.5 | 22.1 ± 10.7 | 82.9 ± 15.9 | Tesla T4 |
| BUSI | QENAS, random Pareto candidate | 3 |  |  | 15.3 ± 0.8 | 18.5 ± 1.0 | 34.3 ± 1.7 | Tesla T4 |
| BUSI | QENAS, zero-shot pick (max ZiCo) | 3 |  |  | 18.3 ± 2.1 | 18.2 ± 5.3 | 37.1 ± 7.4 | Tesla T4 |
| BUSI | U-Net | 3 |  |  | 29.5 ± 0.3 | 28.3 ± 1.5 | 58.4 ± 1.3 | Tesla T4 |
| BUSI | Attention U-Net | 3 |  |  | 16.6 ± 0.4 | 15.7 ± 1.6 | 32.9 ± 1.5 | Tesla T4 |
| BUSI | Random search | 3 | 5.8 ± 0.2 |  |  |  | 6.1 ± 0.2 | Tesla T4 |
| CVC-ClinicDB | QENAS | 3 | 5.2 ± 0.4 | 33.5 ± 2.7 | 15.1 ± 6.0 | 18.8 ± 7.6 | 73.2 ± 16.7 | Tesla T4 |
| CVC-ClinicDB | U-Net | 3 |  |  | 28.1 ± 0.3 | 29.9 ± 0.8 | 58.5 ± 0.4 | Tesla T4 |
| CVC-ClinicDB | Attention U-Net | 3 |  |  | 15.8 ± 0.3 | 14.2 ± 2.0 | 30.5 ± 2.0 | Tesla T4 |
| CVC-ClinicDB | Random search | 3 | 5.4 ± 0.2 |  |  |  | 5.7 ± 0.1 | Tesla T4 |
| IDRiD (optic disc) | QENAS | 3 | 10.2 ± 0.6 | 9.4 ± 0.6 | 5.4 ± 1.3 | 6.8 ± 1.8 | 32.7 ± 3.8 | Tesla T4 |
| IDRiD (optic disc) | U-Net | 3 |  |  | 14.9 ± 0.1 | 13.8 ± 0.5 | 29.7 ± 0.4 | Tesla T4 |
| IDRiD (optic disc) | Attention U-Net | 3 |  |  | 6.0 ± 0.3 | 5.8 ± 0.3 | 12.5 ± 0.6 | Tesla T4 |
| IDRiD (optic disc) | Random search | 3 | 9.2 ± 0.2 |  |  |  | 9.7 ± 0.2 | Tesla T4 |
| Kvasir-SEG | QENAS | 3 | 5.1 ± 0.1 | 46.2 ± 2.4 | 24.3 ± 3.3 | 29.2 ± 5.1 | 105.6 ± 10.2 | Tesla T4 |
| Kvasir-SEG | QENAS w/o ZiCo | 3 | 5.3 ± 0.2 | 54.3 ± 10.3 | 31.6 ± 13.6 | 39.6 ± 16.7 | 131.5 ± 40.6 | Tesla T4 |
| Kvasir-SEG | QENAS w/o SynFlow | 3 | 5.4 ± 0.7 | 66.6 ± 11.3 | 30.6 ± 15.9 | 38.5 ± 20.0 | 141.7 ± 46.6 | Tesla T4 |
| Kvasir-SEG | QENAS, random Pareto candidate | 3 |  |  | 23.7 ± 2.3 | 28.3 ± 4.1 | 52.7 ± 5.6 | Tesla T4 |
| Kvasir-SEG | QENAS, zero-shot pick (max ZiCo) | 3 |  |  | 26.7 ± 3.5 | 33.5 ± 4.4 | 61.0 ± 7.6 | Tesla T4 |
| Kvasir-SEG | U-Net | 3 |  |  | 46.7 ± 1.7 | 46.8 ± 1.9 | 94.6 ± 3.4 | Tesla T4 |
| Kvasir-SEG | Attention U-Net | 3 |  |  | 26.2 ± 0.9 | 26.4 ± 1.0 | 53.2 ± 1.7 | Tesla T4 |
| Kvasir-SEG | Random search | 3 | 5.4 ± 0.1 |  |  |  | 5.8 ± 0.1 | Tesla T4 |
| CVC-ClinicDB, frame-level split | QENAS | 3 | 5.6 ± 0.4 | 35.6 ± 4.5 | 21.3 ± 6.7 | 26.7 ± 8.3 | 89.8 ± 19.6 | Tesla T4 |
| CVC-ClinicDB, frame-level split | U-Net | 3 |  |  | 28.5 ± 1.2 | 28.9 ± 1.0 | 57.8 ± 2.2 | Tesla T4 |
