### RQ5 — External validation: trained on Kvasir-SEG, tested on all CVC-ClinicDB frames (mean ± SD over seeds)

| Train → test | Method | Seeds | Dice / DSC (%) | mIoU (%) | HD95 (px) | Sensitivity (%) | IoU (%) |
|---|---|---|---|---|---|---|---|
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | QENAS | 3 | 78.63 ± 1.97 | 82.95 ± 1.31 | 36.06 ± 3.26 | 82.13 ± 0.80 | 69.79 ± 2.16 |
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | U-Net | 3 | 69.31 ± 0.70 | 78.41 ± 0.47 | 63.10 ± 5.16 | 69.85 ± 0.29 | 61.21 ± 0.86 |
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | Attention U-Net | 3 | 70.72 ± 1.18 | 79.44 ± 0.71 | 65.28 ± 4.56 | 71.47 ± 0.98 | 63.08 ± 1.27 |
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | nnU-Net | 3 | 81.32 ± 0.43 | 84.57 ± 0.40 | 33.68 ± 1.27 | 84.50 ± 0.96 | 73.06 ± 0.64 |
