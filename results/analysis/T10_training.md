### Final training (120 epochs): best epoch, early stopping, generalisation — D25, D29

| Dataset | Method | Runs | Best epoch | Epochs run | Stopped early (overfitting) | Train Dice (%) | Val Dice (%) |
|---|---|---|---|---|---|---|---|
| BUSI | QENAS | 3 | 62.0 ± 23.6 | 98.7 ± 18.9 | 2/3 | 87.91 ± 3.73 | 80.02 ± 0.47 |
| BUSI | QENAS, fixed α = 0.5 | 3 | 60.3 ± 22.0 | 90.0 ± 5.3 | 3/3 | 87.85 ± 3.65 | 79.68 ± 0.63 |
| BUSI | QENAS, random candidate | 3 | 70.3 ± 15.9 | 99.7 ± 21.0 | 2/3 | 88.75 ± 2.58 | 80.25 ± 0.46 |
| BUSI | QENAS, zero-shot pick (max ZiCo) | 3 | 77.0 ± 12.8 | 104.3 ± 8.5 | 3/3 | 89.97 ± 1.32 | 80.64 ± 0.58 |
| BUSI | U-Net | 3 | 91.0 ± 2.6 | 116.7 ± 5.8 | 1/3 | 90.44 ± 1.69 | 74.24 ± 0.37 |
| BUSI | Attention U-Net | 3 | 105.0 ± 7.0 | 120.0 ± 0.0 | 1/3 | 90.33 ± 1.35 | 75.29 ± 0.76 |
| BUSI | nnU-Net | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0/3 | – | 79.10 ± 0.72 |
| CVC-ClinicDB | QENAS | 3 | 44.3 ± 20.5 | 93.3 ± 29.3 | 2/3 | 92.52 ± 1.73 | 76.35 ± 0.46 |
| CVC-ClinicDB | QENAS, fixed α = 0.5 | 3 | 49.7 ± 28.3 | 84.0 ± 33.4 | 2/3 | 92.67 ± 2.47 | 77.84 ± 3.27 |
| CVC-ClinicDB | U-Net | 3 | 44.7 ± 19.9 | 69.3 ± 20.7 | 3/3 | 77.17 ± 15.21 | 59.02 ± 4.83 |
| CVC-ClinicDB | Attention U-Net | 3 | 70.7 ± 10.2 | 104.3 ± 14.0 | 2/3 | 88.65 ± 2.55 | 72.19 ± 3.72 |
| CVC-ClinicDB | nnU-Net | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0/3 | – | 66.78 ± 0.38 |
| IDRiD (optic disc) | QENAS | 3 | 56.7 ± 28.0 | 101.3 ± 17.6 | 2/3 | 96.37 ± 1.20 | 95.83 ± 0.27 |
| IDRiD (optic disc) | QENAS, fixed α = 0.5 | 3 | 61.7 ± 7.1 | 104.7 ± 23.2 | 2/3 | 96.92 ± 0.30 | 96.03 ± 0.19 |
| IDRiD (optic disc) | U-Net | 3 | 115.7 ± 1.5 | 120.0 ± 0.0 | 0/3 | 94.11 ± 1.33 | 92.68 ± 0.58 |
| IDRiD (optic disc) | Attention U-Net | 3 | 103.3 ± 17.8 | 120.0 ± 0.0 | 0/3 | 94.43 ± 1.77 | 94.49 ± 1.17 |
| IDRiD (optic disc) | nnU-Net | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0/3 | – | 95.36 ± 0.24 |
| Kvasir-SEG | QENAS | 3 | 79.7 ± 25.7 | 105.0 ± 22.6 | 2/3 | 93.70 ± 2.05 | 88.89 ± 0.48 |
| Kvasir-SEG | QENAS, fixed α = 0.5 | 3 | 82.7 ± 28.9 | 105.0 ± 22.6 | 2/3 | 93.85 ± 2.21 | 88.52 ± 0.34 |
| Kvasir-SEG | QENAS, random candidate | 3 | 112.0 ± 6.6 | 120.0 ± 0.0 | 0/3 | 95.14 ± 0.50 | 88.50 ± 1.06 |
| Kvasir-SEG | QENAS, zero-shot pick (max ZiCo) | 3 | 113.0 ± 6.2 | 120.0 ± 0.0 | 0/3 | 95.45 ± 0.37 | 89.01 ± 0.13 |
| Kvasir-SEG | U-Net | 3 | 108.0 ± 4.4 | 120.0 ± 0.0 | 0/3 | 94.21 ± 0.96 | 86.97 ± 0.51 |
| Kvasir-SEG | Attention U-Net | 3 | 112.7 ± 6.0 | 120.0 ± 0.0 | 0/3 | 94.99 ± 0.61 | 88.27 ± 0.24 |
| Kvasir-SEG | nnU-Net | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0/3 | – | 90.17 ± 0.83 |
