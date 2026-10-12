### Effect of one gene of each block type (vs Residual), least squares on random architectures — DESIGN_DECISIONS D14

| Dataset | Criterion | All-Residual network | +1 Dense | +1 Inception | +1 ConvNeXt | +1 DARTS |
|---|---|---|---|---|---|---|
| BUSI | params (M) | 1.630 | -0.015 | 0.232 | 0.419 | 0.041 |
| BUSI | ln(1+SynFlow) | 56.191 | 0.456 | 3.118 | 3.874 | 2.788 |
| BUSI | ZiCo | 411.869 | 21.004 | 66.514 | 25.718 | 96.287 |
| CVC-ClinicDB | params (M) | 1.552 | -0.002 | 0.234 | 0.429 | -0.122 |
| CVC-ClinicDB | ln(1+SynFlow) | 56.291 | 0.415 | 3.030 | 3.890 | -1.261 |
| CVC-ClinicDB | ZiCo | 410.148 | 20.793 | 65.518 | 27.128 | 29.707 |
| IDRiD (optic disc) | params (M) | 1.616 | -0.013 | 0.235 | 0.421 | -0.006 |
| IDRiD (optic disc) | ln(1+SynFlow) | 57.105 | 0.483 | 3.196 | 3.866 | -0.306 |
| IDRiD (optic disc) | ZiCo | 393.679 | 22.486 | 68.400 | 27.636 | 69.243 |
| Kvasir-SEG | params (M) | 1.552 | -0.002 | 0.234 | 0.429 | -0.122 |
| Kvasir-SEG | ln(1+SynFlow) | 56.291 | 0.415 | 3.030 | 3.890 | -1.261 |
| Kvasir-SEG | ZiCo | 412.594 | 21.311 | 66.086 | 27.348 | 30.575 |
