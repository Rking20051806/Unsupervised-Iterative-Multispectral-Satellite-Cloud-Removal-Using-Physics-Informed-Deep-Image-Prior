from __future__ import annotations

from pathlib import Path
import torch
from PIL import Image
import numpy as np
from models.segmentation.unet import UNet


class ModelManager:
    """Manages model loading and inference."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.segmentation_model = None
        self.removal_model = None

    def load_segmentation_model(self, model_path: Path | None = None, weights_only: bool = False) -> UNet:
        """Load or create U-Net segmentation model."""
        if self.segmentation_model is None:
            model = UNet(in_channels=3, out_channels=1).to(self.device)
            if model_path is None:
                default_path = Path(__file__).resolve().parents[1] / "models" / "segmentation" / "unet_rice2.pth"
                if default_path.exists():
                    model_path = default_path
            if model_path and model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=weights_only)
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"Warning: failed to load checkpoint from {model_path}: {e}")
            model.eval()
            self.segmentation_model = model
        return self.segmentation_model

    def infer_segmentation(self, image: Image.Image, threshold: float = 0.5) -> tuple[Image.Image, np.ndarray]:
        """Run cloud segmentation on image and return mask + logits."""
        model = self.load_segmentation_model()

        # Preprocess
        img_arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Infer
        with torch.no_grad():
            logits = model(img_tensor)

        # Postprocess
        pred_mask = torch.sigmoid(logits).squeeze().cpu().numpy()
        binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
        mask_image = Image.fromarray(binary_mask, mode="L")

        return mask_image, pred_mask

    def classify_cloud(self, mask: np.ndarray) -> dict[str, float]:
        """Classify cloud coverage and type from mask."""
        cloud_ratio = mask.mean()
        
        classification = {
            "cloud_free": 0.0,
            "thin_cloud": 0.0,
            "thick_cloud": 0.0,
            "cirrus": 0.0,
        }

        if cloud_ratio < 0.1:
            classification["cloud_free"] = 1.0
        elif cloud_ratio < 0.3:
            classification["thin_cloud"] = 0.7
            classification["cirrus"] = 0.3
        elif cloud_ratio < 0.6:
            classification["thin_cloud"] = 0.5
            classification["thick_cloud"] = 0.5
        else:
            classification["thick_cloud"] = 1.0

        return classification

    def apply_removal_filter(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply cloud removal filter (placeholder until real model is available)."""
        img_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0

        # For non-cloudy regions, enhance; for cloudy regions, apply inpainting-like effect
        output = img_arr.copy()
        cloudy_region = mask_arr > 0.5

        if cloudy_region.any():
            # Enhance non-cloudy regions
            non_cloudy = ~cloudy_region
            output[non_cloudy] = np.clip(output[non_cloudy] * 1.15, 0, 1)

            # For cloudy regions, use simple interpolation from neighbors
            output[cloudy_region] = output[cloudy_region] * 0.85

        output = (output * 255).astype(np.uint8)
        return Image.fromarray(output, mode="RGB")
