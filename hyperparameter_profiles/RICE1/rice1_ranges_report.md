# 🛰️ RICE1 Dataset — Optimal Hyperparameter Profiling Report

**Dataset Nature**: Thin Cloud / Hazy Translucent Layer across full frame.
**Primary Loss**: Atmospheric Scattering Model (ASM) Inversion ($I = J \cdot t + A(1-t)$) + Angstrom RTE.

**Total Samples Analyzed**: `500`

## 📊 Recommended Parameter Scales & Ranges

| Hyperparameter | Min Observed | Max Observed | Mean | Recommended Best Default |
| :--- | :---: | :---: | :---: | :---: |
| **`lambda_asm`** | `0.5093` | `0.6064` | `0.5453` | **`0.5311`** |
| **`lambda_rte`** | `0.02` | `0.035` | `0.0218` | **`0.02`** |
| **`noise_std`** | `0.012` | `0.017` | `0.0148` | **`0.0153`** |
| **`lambda_tv`** | `9e-05` | `0.00012` | `0.00012` | **`0.00012`** |
| **`lambda_ndvi`** | `0.0502` | `0.0571` | `0.0532` | **`0.0533`** |
| **`lambda_t_prior`** | `0.1` | `0.15` | `0.1285` | **`0.133`** |
| **`learning_rate`** | `0.005` | `0.005` | `0.005` | **`0.005`** |


---
*Generated automatically by Autonomous Hyperparameter Profiler.*
