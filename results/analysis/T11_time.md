### Compute time per experiment (minutes) and hardware

| Dataset | Method | Runs | search (min) | candidates (min) | loss_weights (min) | final (min) | total (min) | GPU | nnunet_train (min) |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS | 3 | 5.5 ± 0.3 | 34.8 ± 8.1 | 16.5 ± 6.3 | 21.4 ± 12.3 | 78.7 ± 26.6 | Tesla T4 |  |
| BUSI | QENAS, fixed α = 0.5 | 2 |  |  |  | 14.2 ± 1.2 | 14.6 ± 1.2 | Tesla T4 |  |
| BUSI | QENAS, random candidate | 2 |  |  | 26.1 ± 17.2 | 29.3 ± 12.5 | 55.9 ± 29.7 | Tesla T4 |  |
| BUSI | QENAS, zero-shot pick (max ZiCo) | 2 |  |  | 31.6 ± 9.1 | 31.4 ± 9.5 | 63.5 ± 18.5 | Tesla T4 |  |
| BUSI | U-Net | 2 |  |  | 28.5 ± 0.1 | 34.2 ± 0.3 | 63.1 ± 0.4 | Tesla T4 |  |
| BUSI | Attention U-Net | 2 |  |  | 17.0 ± 0.4 | 19.8 ± 0.3 | 37.2 ± 0.1 | Tesla T4 |  |
| BUSI | nnU-Net | 2 |  |  |  |  | 180.5 ± 5.4 | Tesla T4 | 178.1 ± 5.4 |
| BUSI | Random search | 3 | 4.6 ± 0.1 |  |  |  | 4.8 ± 0.1 | Tesla T4 |  |
| CVC-ClinicDB | QENAS | 3 | 4.7 ± 0.7 | 39.6 ± 7.7 | 20.7 ± 7.3 | 23.0 ± 7.7 | 88.4 ± 21.3 | Tesla T4 |  |
| CVC-ClinicDB | QENAS, fixed α = 0.5 | 2 |  |  |  | 17.5 ± 1.7 | 18.0 ± 1.8 | Tesla T4 |  |
| CVC-ClinicDB | U-Net | 2 |  |  | 27.3 ± 0.3 | 19.0 ± 7.5 | 46.7 ± 7.2 | Tesla T4 |  |
| CVC-ClinicDB | Attention U-Net | 2 |  |  | 15.5 ± 0.2 | 15.2 ± 0.7 | 31.1 ± 0.5 | Tesla T4 |  |
| CVC-ClinicDB | nnU-Net | 2 |  |  |  |  | 165.5 ± 3.4 | Tesla T4 | 163.6 ± 3.6 |
| CVC-ClinicDB | Random search | 3 | 4.3 ± 0.1 |  |  |  | 4.5 ± 0.1 | Tesla T4 |  |
| IDRiD (optic disc) | QENAS | 3 | 9.4 ± 1.4 | 9.4 ± 0.3 | 3.7 ± 0.7 | 4.8 ± 1.4 | 28.1 ± 0.6 | Tesla T4 |  |
| IDRiD (optic disc) | QENAS, fixed α = 0.5 | 3 |  |  |  | 5.1 ± 1.8 | 5.8 ± 1.8 | Tesla T4 |  |
| IDRiD (optic disc) | U-Net | 2 |  |  | 14.0 ± 1.3 | 15.6 ± 0.6 | 30.3 ± 1.9 | Tesla T4 |  |
| IDRiD (optic disc) | Attention U-Net | 3 |  |  | 5.9 ± 0.3 | 6.9 ± 0.3 | 13.4 ± 0.6 | Tesla T4 |  |
| IDRiD (optic disc) | nnU-Net | 2 |  |  |  |  | 72.1 ± 1.2 | Tesla T4 | 70.1 ± 1.4 |
| IDRiD (optic disc) | Random search | 3 | 7.1 ± 0.3 |  |  |  | 7.6 ± 0.4 | Tesla T4 |  |
| Kvasir-SEG | QENAS | 2 | 4.7 ± 0.6 | 76.7 ± 6.4 | 28.0 ± 3.7 | 41.7 ± 6.3 | 151.9 ± 17.3 | Tesla T4 |  |
| Kvasir-SEG | QENAS, fixed α = 0.5 | 2 |  |  |  | 41.8 ± 6.3 | 42.3 ± 6.3 | Tesla T4 |  |
| Kvasir-SEG | QENAS, random candidate | 2 |  |  | 60.8 ± 0.5 | 73.2 ± 0.7 | 134.8 ± 1.1 | Tesla T4 |  |
| Kvasir-SEG | QENAS, zero-shot pick (max ZiCo) | 1 |  |  | 50.9 | 74.1 | 125.7 | Tesla T4 |  |
| Kvasir-SEG | U-Net | 2 |  |  | 44.9 ± 1.2 | 54.6 ± 1.9 | 100.1 ± 3.1 | Tesla T4 |  |
| Kvasir-SEG | Attention U-Net | 2 |  |  | 27.3 ± 0.6 | 33.1 ± 0.2 | 60.9 ± 0.4 | Tesla T4 |  |
| Kvasir-SEG | nnU-Net | 2 |  |  |  |  | 272.4 ± 2.6 | Tesla T4 | 270.0 ± 2.5 |
| Kvasir-SEG | Random search | 3 | 4.3 ± 0.1 |  |  |  | 4.5 ± 0.2 | Tesla T4 |  |
