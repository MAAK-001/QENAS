### RQ4 — External validation: trained on Kvasir-SEG, tested on all CVC-ClinicDB frames (mean ± SD over seeds)

| Train → test | Method | Seeds | Dice / DSC (%) | mIoU (%) | HD95 (px) | Sensitivity (%) | IoU (%) |
|---|---|---|---|---|---|---|---|
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | QENAS | 3 | 75.92 ± 5.66 | 81.70 ± 2.71 | 48.83 ± 22.03 | 78.67 ± 6.13 | 67.42 ± 5.05 |
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | U-Net | 3 | 68.52 ± 2.41 | 78.23 ± 1.27 | 68.10 ± 5.71 | 68.00 ± 2.75 | 60.79 ± 2.39 |
| Kvasir-SEG → CVC-ClinicDB (all 612 frames) | Attention U-Net | 3 | 70.94 ± 0.59 | 79.72 ± 0.18 | 66.18 ± 4.74 | 71.40 ± 1.00 | 63.64 ± 0.31 |
