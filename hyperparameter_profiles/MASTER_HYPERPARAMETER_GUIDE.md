# 🛰️ Master Hyperparameter & Calibration Guide for DIP-PINN

This comprehensive guide details the calculated optimal physical scales, ranges, and sample-level distributions across both **RICE1 (Thin Cloud)** and **RICE2 (Thick/Dense Cloud)** datasets.

## 🔬 Physical Distinction Between RICE1 and RICE2

| Characteristic | RICE-1 (Thin Cloud) | RICE-2 (Thick Cloud) |
| :--- | :--- | :--- |
| **Cloud Optics** | Translucent thin veil / haze ($t \in [0.35, 0.95]$) | Opaque cumulus clouds ($t \to 0$ in cloud core) |
| **Primary Model** | Full-frame ASM Dehazing Inversion | Masked Context DIP Inpainting |
| **$\lambda_{\text{ASM}}$ (Atmospheric Scattering)** | **`0.50 - 0.65`** (Primary dehazing constraint) | **`0.04 - 0.08`** (Soft edge regularizer) |
| **$\lambda_{\text{RTE}}$ (Radiative Transfer)** | **`0.020 - 0.035`** (Multi-spectral Angstrom) | **`0.025 - 0.045`** (Mie boundary scattering) |
| **$\lambda_{\text{T-Prior}}$ (Transmission Prior)** | **`0.08 - 0.15`** (Targets $t \approx 0.45$) | **`0.60 - 1.20`** (Forces $t \to 0$ in clouds) |
| **Noise $\sigma$ (DIP Regularization)** | **`0.010 - 0.015`** | **`0.018 - 0.028`** (Prevents mask boundary artifacts) |
| **$\lambda_{\text{TV}}$ (Total Variation)** | **`5e-5 - 1.5e-4`** | **`1e-5 - 4e-5`** |
| **Learning Rate** | **`0.005`** | **`0.003`** |

## 📊 RICE1 Dataset Aggregated Summary

| Parameter | Min Range | Max Range | Dataset Mean | Recommended Best Default |
| :--- | :---: | :---: | :---: | :---: |
| **`lambda_asm`** | `0.5093` | `0.6064` | `0.5453` | **`0.5311`** |
| **`lambda_rte`** | `0.02` | `0.035` | `0.0218` | **`0.02`** |
| **`noise_std`** | `0.012` | `0.017` | `0.0148` | **`0.0153`** |
| **`lambda_tv`** | `9e-05` | `0.00012` | `0.00012` | **`0.00012`** |
| **`lambda_ndvi`** | `0.0502` | `0.0571` | `0.0532` | **`0.0533`** |
| **`lambda_t_prior`** | `0.1` | `0.15` | `0.1285` | **`0.133`** |
| **`learning_rate`** | `0.005` | `0.005` | `0.005` | **`0.005`** |

## 📊 RICE2 Dataset Aggregated Summary

| Parameter | Min Range | Max Range | Dataset Mean | Recommended Best Default |
| :--- | :---: | :---: | :---: | :---: |
| **`lambda_asm`** | `0.05` | `0.08` | `0.0579` | **`0.055`** |
| **`lambda_rte`** | `0.025` | `0.0417` | `0.0302` | **`0.0294`** |
| **`noise_std`** | `0.018` | `0.028` | `0.0204` | **`0.0194`** |
| **`lambda_tv`** | `1.6e-05` | `2.4e-05` | `2.3e-05` | **`2.4e-05`** |
| **`lambda_ndvi`** | `0.042` | `0.0625` | `0.0465` | **`0.0448`** |
| **`lambda_t_prior`** | `0.6` | `1.234` | `0.7503` | **`0.691`** |
| **`learning_rate`** | `0.003` | `0.003` | `0.003` | **`0.003`** |

## 📁 Directory Structure

```text
hyperparameter_profiles/
├── MASTER_HYPERPARAMETER_GUIDE.md
├── RICE1/
│   ├── rice1_dataset_summary.csv
│   ├── rice1_dataset_summary.json
│   ├── rice1_ranges_report.md
│   └── sample_0.json ... sample_499.json
└── RICE2/
    ├── rice2_dataset_summary.csv
    ├── rice2_dataset_summary.json
    ├── rice2_ranges_report.md
    └── sample_0.json ... sample_735.json
```
