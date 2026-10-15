### RQ3 — Selection rule: learning capability vs random candidate vs zero-shot (max ZiCo)

| Dataset | Method | Seeds | Dice / DSC (%) | mIoU (%) | HD95 (px) | Sensitivity (%) | IoU (%) | Params (M) | FLOPs (G, batch) | GFLOPs / image | ms / image |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BUSI | QENAS | 3 | 79.30 ± 0.90 | 82.57 ± 0.53 | 28.78 ± 1.43 | 79.50 ± 1.50 | 70.32 ± 0.73 | 1.04 ± 0.03 | 20.4 ± 2.3 | 2.55 ± 0.29 | 21.4 ± 4.2 |
| BUSI | QENAS, random candidate | 2 | 80.43 ± 0.46 | 83.32 ± 0.15 | 26.69 ± 1.76 | 81.39 ± 2.12 | 71.67 ± 0.27 | 1.94 ± 1.15 | 32.3 ± 9.6 | 4.04 ± 1.20 | 32.5 ± 0.1 |
| BUSI | QENAS, zero-shot pick (max ZiCo) | 2 | 79.97 ± 0.19 | 83.16 ± 0.08 | 29.04 ± 1.56 | 80.73 ± 1.18 | 71.37 ± 0.15 | 2.76 ± 0.00 | 40.2 ± 1.6 | 5.02 ± 0.19 | 32.3 ± 1.9 |
| Kvasir-SEG | QENAS | 2 | 85.39 ± 0.55 | 86.27 ± 0.59 | 28.73 ± 2.25 | 87.30 ± 0.83 | 77.83 ± 1.02 | 2.51 ± 0.05 | 34.0 ± 11.7 | 4.25 ± 1.46 | 18.7 ± 2.1 |
| Kvasir-SEG | QENAS, random candidate | 2 | 84.04 ± 0.81 | 84.81 ± 0.76 | 32.44 ± 1.86 | 86.05 ± 0.81 | 76.00 ± 1.21 | 1.57 ± 0.99 | 35.7 ± 2.9 | 4.46 ± 0.36 | 20.6 ± 1.4 |
| Kvasir-SEG | QENAS, zero-shot pick (max ZiCo) | 1 | 84.90 | 85.75 | 30.46 | 86.21 | 77.13 | 2.54 | 40.1 | 5.01 | 18.5 |
