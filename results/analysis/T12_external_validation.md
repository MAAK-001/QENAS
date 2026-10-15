### RQ5 — External validation: trained on Kvasir-SEG, tested on all CVC-ClinicDB frames (mean ± SD over seeds)

| Train → test | Method | Seeds | Dice / DSC (%) | mIoU (%) | HD95 (px) | Sensitivity (%) | IoU (%) |
|---|---|---|---|---|---|---|---|
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | QENAS | 2 | 79.64 ± 1.24 | 83.62 ± 0.81 | 34.51 ± 2.62 | 82.38 ± 0.95 | 70.92 ± 1.30 |
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | U-Net | 2 | 69.24 ± 0.98 | 78.39 ± 0.67 | 64.12 ± 6.85 | 69.69 ± 0.08 | 61.18 ± 1.21 |
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | Attention U-Net | 2 | 70.26 ± 1.24 | 79.14 ± 0.67 | 67.85 ± 1.35 | 71.40 ± 1.37 | 62.54 ± 1.22 |
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | nnU-Net | 2 | 81.54 ± 0.29 | 84.77 ± 0.28 | 32.94 ± 0.11 | 83.96 ± 0.36 | 73.41 ± 0.33 |
