# 🛰️ RICE2 Dataset — Optimal Hyperparameter Profiling Report

**Dataset Nature**: Thick Cumulus / Dense Opaque Cloud Patches.
**Primary Loss**: Masked Deep Image Prior Inpainting on Cloud Mask $M$ + Clear Boundary Continuity.

**Total Samples Analyzed**: `736`

## 📊 Recommended Parameter Scales & Ranges

| Hyperparameter | Min Observed | Max Observed | Mean | Recommended Best Default |
| :--- | :---: | :---: | :---: | :---: |
| **`lambda_asm`** | `0.05` | `0.08` | `0.0579` | **`0.055`** |
| **`lambda_rte`** | `0.025` | `0.0417` | `0.0302` | **`0.0294`** |
| **`noise_std`** | `0.018` | `0.028` | `0.0204` | **`0.0194`** |
| **`lambda_tv`** | `1.6e-05` | `2.4e-05` | `2.3e-05` | **`2.4e-05`** |
| **`lambda_ndvi`** | `0.042` | `0.0625` | `0.0465` | **`0.0448`** |
| **`lambda_t_prior`** | `0.6` | `1.234` | `0.7503` | **`0.691`** |
| **`learning_rate`** | `0.003` | `0.003` | `0.003` | **`0.003`** |


---
*Generated automatically by Autonomous Hyperparameter Profiler.*
