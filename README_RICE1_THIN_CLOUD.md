# RICE-I Thin-Cloud DIP + PINN

## Files
- `rice1_thin_cloud_dip.py` — inference/optimization pipeline.
- `rice1_thin_cloud_losses.py` — RICE-I loss functions.

## Expected inputs
- `cloudy`: RGB synthetic cloudy image from RICE-I.
- `mask`: binary cloud mask, `1 = cloud`, `0 = clear`.
- `gt`: optional cloud-free RICE-I reference. **Used only for evaluation, not optimization.**

## Key changes from the RICE-II thick-cloud version
1. RGB-only: no NIR/NDVI loss.
2. No `remove_clouds_advanced()` inpainting initializer. Thin clouds retain useful scene information, so the cloudy RGB image is used directly as fixed DIP input `z`.
3. Transmission prior is softened: cloud target `t=0.65`, clear target `t=0.95`; thin clouds are not forced toward opaque-cloud transmission.
4. ASM weight = `0.5` and RTE weight = `0.05`.
5. Edge-aware TV weight = `1e-4`.
6. Transmission-prior weight = `0.10`.
7. GT is not passed into the loss, preventing supervised leakage in zero-shot DIP inference.
8. Output files are prefixed with `rice1_thin_cloud_` to keep RICE-I and RICE-II experiments separate.

## Example
```python
from PIL import Image
from rice1_thin_cloud_dip import optimize_rice1_thin_cloud

cloudy = Image.open("RICE1/cloudy/sample_001.png")
mask = Image.open("RICE1/mask/sample_001.png")
gt = Image.open("RICE1/ground_truth/sample_001.png")

optimize_rice1_thin_cloud(
    cloudy=cloudy,
    mask=mask,
    gt=gt,
    iterations=2000,
    lr=0.005,
    image_size=256,
    output_dir="rice1_thin_cloud_outputs/sample_001",
)
```
