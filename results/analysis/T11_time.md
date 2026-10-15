### Compute time per experiment (minutes) and hardware

| Dataset | Method | Runs | search (min) | candidates (min) | loss_weights (min) | final (min) | total (min) | GPU | nnunet_train (min) |
|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS | 3 | 5.5 ± 0.3 | 34.8 ± 8.1 | 16.5 ± 6.3 | 21.4 ± 12.3 | 78.7 ± 26.6 | Tesla T4 |  |
| BUSI | QENAS, fixed α = 0.5 | 3 |  |  |  | 19.2 ± 8.7 | 19.8 ± 9.0 | Tesla T4 |  |
| BUSI | QENAS, random candidate | 3 |  |  | 30.9 ± 14.7 | 29.8 ± 8.9 | 61.3 ± 23.0 | Tesla T4 |  |
| BUSI | QENAS, zero-shot pick (max ZiCo) | 3 |  |  | 33.1 ± 6.9 | 34.6 ± 8.7 | 68.3 ± 15.5 | Tesla T4 |  |
| BUSI | U-Net | 3 |  |  | 28.9 ± 0.7 | 34.8 ± 1.1 | 64.1 ± 1.8 | Tesla T4 |  |
| BUSI | Attention U-Net | 3 |  |  | 16.9 ± 0.3 | 19.9 ± 0.3 | 37.2 ± 0.1 | Tesla T4 |  |
| BUSI | nnU-Net | 3 |  |  |  |  | 181.9 ± 4.5 | Tesla T4 | 179.7 ± 4.7 |
| BUSI | Random search | 3 | 4.6 ± 0.1 |  |  |  | 4.8 ± 0.1 | Tesla T4 |  |
| CVC-ClinicDB | QENAS | 3 | 4.7 ± 0.7 | 39.6 ± 7.7 | 20.7 ± 7.3 | 23.0 ± 7.7 | 88.4 ± 21.3 | Tesla T4 |  |
| CVC-ClinicDB | QENAS, fixed α = 0.5 | 3 |  |  |  | 20.2 ± 4.8 | 20.7 ± 4.8 | Tesla T4 |  |
| CVC-ClinicDB | U-Net | 3 |  |  | 28.8 ± 2.5 | 20.6 ± 6.0 | 49.8 ± 7.3 | Tesla T4 |  |
| CVC-ClinicDB | Attention U-Net | 3 |  |  | 15.8 ± 0.6 | 16.8 ± 2.9 | 33.0 ± 3.4 | Tesla T4 |  |
| CVC-ClinicDB | nnU-Net | 3 |  |  |  |  | 168.8 ± 6.2 | Tesla T4 | 166.9 ± 6.3 |
| CVC-ClinicDB | Random search | 3 | 4.3 ± 0.1 |  |  |  | 4.5 ± 0.1 | Tesla T4 |  |
| IDRiD (optic disc) | QENAS | 3 | 9.4 ± 1.4 | 9.4 ± 0.3 | 3.7 ± 0.7 | 4.8 ± 1.4 | 28.1 ± 0.6 | Tesla T4 |  |
| IDRiD (optic disc) | QENAS, fixed α = 0.5 | 3 |  |  |  | 5.1 ± 1.8 | 5.8 ± 1.8 | Tesla T4 |  |
| IDRiD (optic disc) | U-Net | 3 |  |  | 14.4 ± 1.2 | 15.8 ± 0.7 | 31.2 ± 2.0 | Tesla T4 |  |
| IDRiD (optic disc) | Attention U-Net | 3 |  |  | 5.9 ± 0.3 | 6.9 ± 0.3 | 13.4 ± 0.6 | Tesla T4 |  |
| IDRiD (optic disc) | nnU-Net | 3 |  |  |  |  | 70.2 ± 3.3 | Tesla T4 | 68.5 ± 2.9 |
| IDRiD (optic disc) | Random search | 3 | 7.1 ± 0.3 |  |  |  | 7.6 ± 0.4 | Tesla T4 |  |
| Kvasir-SEG | QENAS | 3 | 5.0 ± 0.7 | 77.0 ± 4.5 | 28.7 ± 2.8 | 37.8 ± 8.1 | 149.4 ± 13.0 | Tesla T4 |  |
| Kvasir-SEG | QENAS, fixed α = 0.5 | 3 |  |  |  | 38.2 ± 7.6 | 38.7 ± 7.7 | Tesla T4 |  |
| Kvasir-SEG | QENAS, random candidate | 3 |  |  | 48.8 ± 20.9 | 58.8 ± 25.0 | 108.3 ± 45.9 | Tesla T4 |  |
| Kvasir-SEG | QENAS, zero-shot pick (max ZiCo) | 3 |  |  | 58.9 ± 7.0 | 75.0 ± 1.5 | 134.8 ± 8.1 | Tesla T4 |  |
| Kvasir-SEG | U-Net | 3 |  |  | 46.3 ± 2.5 | 56.1 ± 2.9 | 103.0 ± 5.4 | Tesla T4 |  |
| Kvasir-SEG | Attention U-Net | 3 |  |  | 27.3 ± 0.4 | 33.1 ± 0.2 | 60.9 ± 0.3 | Tesla T4 |  |
| Kvasir-SEG | nnU-Net | 3 |  |  |  |  | 277.0 ± 8.0 | Tesla T4 | 274.5 ± 8.0 |
| Kvasir-SEG | Random search | 3 | 4.3 ± 0.1 |  |  |  | 4.5 ± 0.2 | Tesla T4 |  |
