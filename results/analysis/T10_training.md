### Final training (120 epochs): best epoch, early stopping, generalisation — D25, D29

| Dataset | Method | Runs | Best epoch | Epochs run | Stopped early (overfitting) | Train Dice (%) | Val Dice (%) |
|---|---|---|---|---|---|---|---|
| BUSI | QENAS | 3 | 62.0 ± 23.6 | 98.7 ± 18.9 | 2/3 | 87.91 ± 3.73 | 80.02 ± 0.47 |
| BUSI | QENAS, fixed α = 0.5 | 2 | 53.5 ± 26.2 | 88.0 ± 5.7 | 2/2 | 86.28 ± 3.46 | 79.94 ± 0.64 |
| BUSI | QENAS, random candidate | 2 | 77.0 ± 15.6 | 110.5 ± 13.4 | 1/2 | 90.10 ± 1.54 | 80.38 ± 0.57 |
| BUSI | QENAS, zero-shot pick (max ZiCo) | 2 | 70.0 ± 5.7 | 99.5 ± 2.1 | 2/2 | 89.22 ± 0.30 | 80.43 ± 0.64 |
| BUSI | U-Net | 2 | 91.5 ± 3.5 | 120.0 ± 0.0 | 0/2 | 90.36 ± 2.38 | 74.42 ± 0.31 |
| BUSI | Attention U-Net | 2 | 106.5 ± 9.2 | 120.0 ± 0.0 | 1/2 | 90.88 ± 1.35 | 75.40 ± 1.04 |
| BUSI | nnU-Net | 2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0/2 | – | 78.91 ± 0.91 |
| CVC-ClinicDB | QENAS | 3 | 44.3 ± 20.5 | 93.3 ± 29.3 | 2/3 | 92.52 ± 1.73 | 76.35 ± 0.46 |
| CVC-ClinicDB | QENAS, fixed α = 0.5 | 2 | 53.5 ± 38.9 | 87.0 ± 46.7 | 1/2 | 92.80 ± 3.47 | 76.24 ± 2.47 |
| CVC-ClinicDB | U-Net | 2 | 40.5 ± 26.2 | 67.5 ± 29.0 | 2/2 | 73.21 ± 19.20 | 57.93 ± 6.29 |
| CVC-ClinicDB | Attention U-Net | 2 | 68.5 ± 13.4 | 96.5 ± 4.9 | 2/2 | 87.63 ± 2.59 | 70.08 ± 0.90 |
| CVC-ClinicDB | nnU-Net | 2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0/2 | – | 66.78 ± 0.53 |
| IDRiD (optic disc) | QENAS | 3 | 56.7 ± 28.0 | 101.3 ± 17.6 | 2/3 | 96.37 ± 1.20 | 95.83 ± 0.27 |
| IDRiD (optic disc) | QENAS, fixed α = 0.5 | 3 | 61.7 ± 7.1 | 104.7 ± 23.2 | 2/3 | 96.92 ± 0.30 | 96.03 ± 0.19 |
| IDRiD (optic disc) | U-Net | 2 | 115.0 ± 1.4 | 120.0 ± 0.0 | 0/2 | 94.41 ± 1.72 | 92.77 ± 0.79 |
| IDRiD (optic disc) | Attention U-Net | 3 | 103.3 ± 17.8 | 120.0 ± 0.0 | 0/3 | 94.43 ± 1.77 | 94.49 ± 1.17 |
| IDRiD (optic disc) | nnU-Net | 2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0/2 | – | 95.25 ± 0.22 |
| Kvasir-SEG | QENAS | 2 | 94.5 ± 2.1 | 118.0 ± 2.8 | 1/2 | 94.88 ± 0.26 | 89.14 ± 0.32 |
| Kvasir-SEG | QENAS, fixed α = 0.5 | 2 | 99.0 ± 8.5 | 118.0 ± 2.8 | 1/2 | 95.11 ± 0.58 | 88.58 ± 0.47 |
| Kvasir-SEG | QENAS, random candidate | 2 | 115.0 ± 5.7 | 120.0 ± 0.0 | 0/2 | 95.41 ± 0.29 | 88.71 ± 1.40 |
| Kvasir-SEG | QENAS, zero-shot pick (max ZiCo) | 1 | 120.0 | 120.0 | 0/1 | 95.84 | 88.89 |
| Kvasir-SEG | U-Net | 2 | 105.5 ± 0.7 | 120.0 ± 0.0 | 0/2 | 93.98 ± 1.24 | 87.07 ± 0.68 |
| Kvasir-SEG | Attention U-Net | 2 | 115.5 ± 4.9 | 120.0 ± 0.0 | 0/2 | 94.64 ± 0.16 | 88.21 ± 0.30 |
| Kvasir-SEG | nnU-Net | 2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0/2 | – | 90.58 ± 0.59 |
