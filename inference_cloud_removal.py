"""
CloudRemovalInference — Inference pipeline for the supervised CloudRemovalUNet.

Loads a trained checkpoint and runs single-image inference:
  input  (cloudy RGB image + cloud mask) → output (reconstructed cloud-free RGB)

Features:
  - Single-image inference with automatic preprocessing
  - 4-panel visualization (cloudy / mask / ground truth / predicted)
  - Side-by-side "Before | After" comparison
  - PSNR / SSIM computation when ground truth is available
  - Batch inference support

Based on the inference pipeline described in Nikita Zade's ISRO Internship Report.

Usage:
    python inference_cloud_removal.py --image path/to/cloudy.png --mask path/to/mask.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cloud_removal.cloud_removal_unet import CloudRemovalUNet


class CloudRemovalInference:
    """Inference pipeline for the supervised cloud removal U-Net.

    Attributes:
        model:      Loaded CloudRemovalUNet in eval mode
        device:     Inference device
        image_size: Spatial resolution for inference (default: 256)
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        image_size: int = 256,
    ):
        """Initialize inference engine.

        Args:
            checkpoint_path: Path to trained model checkpoint (.pth).
                             If None, searches for default location.
            device:          Target device. Auto-detects GPU if available.
            image_size:      Input resolution for the model.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size

        # Load model
        self.model = CloudRemovalUNet(in_channels=4, out_channels=3).to(self.device)

        # Find checkpoint
        if checkpoint_path is None:
            default_path = PROJECT_ROOT / "models" / "cloud_removal" / "best_model.pth"
            if default_path.exists():
                checkpoint_path = default_path
            else:
                print("Warning: No checkpoint found. Model is untrained (random weights).")
                self.model.eval()
                return

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                epoch = checkpoint.get("epoch", "?")
                best_loss = checkpoint.get("best_val_loss", "?")
                print(f"Loaded checkpoint from epoch {epoch} (best_val_loss={best_loss})")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model weights from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")

        self.model.eval()
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"CloudRemovalUNet ({param_count:,} params) on {self.device}")

    def preprocess(
        self,
        cloud_image: Image.Image,
        mask_image: Image.Image,
    ) -> torch.Tensor:
        """Preprocess input images for the model.

        Args:
            cloud_image: RGB cloudy satellite image.
            mask_image:  Grayscale binary cloud mask.

        Returns:
            Input tensor [1, 4, H, W] on the target device.
        """
        sz = self.image_size

        cloud_img = cloud_image.convert("RGB").resize((sz, sz), Image.Resampling.LANCZOS)
        mask_img = mask_image.convert("L").resize((sz, sz), Image.Resampling.NEAREST)

        cloud_tensor = TF.to_tensor(cloud_img)   # [3, H, W]
        mask_tensor = TF.to_tensor(mask_img)      # [1, H, W]

        input_tensor = torch.cat([cloud_tensor, mask_tensor], dim=0)  # [4, H, W]
        return input_tensor.unsqueeze(0).to(self.device)  # [1, 4, H, W]

    @torch.no_grad()
    def infer(
        self,
        cloud_image: Image.Image,
        mask_image: Image.Image,
    ) -> Image.Image:
        """Run inference on a single image.

        Args:
            cloud_image: RGB cloudy satellite image.
            mask_image:  Binary cloud mask.

        Returns:
            Reconstructed cloud-free RGB image as PIL Image.
        """
        input_tensor = self.preprocess(cloud_image, mask_image)

        t0 = time.time()
        output = self.model(input_tensor)  # [1, 3, H, W]
        elapsed = time.time() - t0

        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        output_np = (np.clip(output_np, 0.0, 1.0) * 255).astype(np.uint8)

        print(f"Inference time: {elapsed:.3f}s")
        return Image.fromarray(output_np, mode="RGB")

    @torch.no_grad()
    def infer_with_metrics(
        self,
        cloud_image: Image.Image,
        mask_image: Image.Image,
        ground_truth: Optional[Image.Image] = None,
    ) -> dict:
        """Run inference and compute quality metrics.

        Args:
            cloud_image:  RGB cloudy input.
            mask_image:   Binary cloud mask.
            ground_truth: Optional cloud-free ground truth for PSNR/SSIM.

        Returns:
            Dictionary with:
                - "predicted":  PIL Image of reconstructed output
                - "elapsed":    Inference time in seconds
                - "psnr":       PSNR (if ground_truth provided)
                - "ssim":       SSIM (if ground_truth provided)
                - "mse":        MSE loss value
                - "l1":         L1 loss value
        """
        input_tensor = self.preprocess(cloud_image, mask_image)

        t0 = time.time()
        output = self.model(input_tensor)  # [1, 3, H, W]
        elapsed = time.time() - t0

        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        output_np = np.clip(output_np, 0.0, 1.0)

        result: dict = {
            "predicted": Image.fromarray((output_np * 255).astype(np.uint8), mode="RGB"),
            "elapsed": elapsed,
        }

        if ground_truth is not None:
            sz = self.image_size
            gt_img = ground_truth.convert("RGB").resize((sz, sz), Image.Resampling.LANCZOS)
            gt_np = np.array(gt_img, dtype=np.float32) / 255.0

            # MSE and L1
            result["mse"] = float(np.mean((output_np - gt_np) ** 2))
            result["l1"] = float(np.mean(np.abs(output_np - gt_np)))

            # PSNR
            if result["mse"] < 1e-10:
                result["psnr"] = 99.0
            else:
                result["psnr"] = float(10.0 * np.log10(1.0 / result["mse"]))

            # SSIM
            try:
                from skimage.metrics import structural_similarity
                result["ssim"] = float(structural_similarity(
                    output_np, gt_np, channel_axis=2, data_range=1.0
                ))
            except ImportError:
                result["ssim"] = 0.0

        return result

    def create_comparison_panel(
        self,
        cloud_image: Image.Image,
        mask_image: Image.Image,
        predicted: Image.Image,
        ground_truth: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Create a 4-panel comparison visualization.

        Panels: Cloudy Input | Cloud Mask | Ground Truth (or blank) | Predicted

        Args:
            cloud_image:  Original cloudy image.
            mask_image:   Cloud mask.
            predicted:    Model prediction.
            ground_truth: Ground truth (optional).

        Returns:
            PIL Image containing the 4-panel layout.
        """
        sz = self.image_size
        cloud_rgb = cloud_image.convert("RGB").resize((sz, sz), Image.Resampling.LANCZOS)
        mask_l = mask_image.convert("L").resize((sz, sz), Image.Resampling.NEAREST)
        pred_rgb = predicted.convert("RGB").resize((sz, sz), Image.Resampling.LANCZOS)

        # Colorize mask (red channel)
        mask_np = np.array(mask_l, dtype=np.uint8)
        mask_rgb = np.stack([mask_np, np.zeros_like(mask_np), np.zeros_like(mask_np)], axis=-1)
        mask_pil = Image.fromarray(mask_rgb, mode="RGB")

        if ground_truth is not None:
            gt_rgb = ground_truth.convert("RGB").resize((sz, sz), Image.Resampling.LANCZOS)
        else:
            gt_rgb = Image.new("RGB", (sz, sz), (30, 30, 30))

        # Compose panel
        panel = Image.new("RGB", (sz * 4, sz), (0, 0, 0))
        panel.paste(cloud_rgb, (0, 0))
        panel.paste(mask_pil, (sz, 0))
        panel.paste(gt_rgb, (sz * 2, 0))
        panel.paste(pred_rgb, (sz * 3, 0))

        return panel


def main():
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(description="Run supervised cloud removal inference")
    parser.add_argument("--image", type=str, required=True, help="Path to cloudy image")
    parser.add_argument("--mask", type=str, required=True, help="Path to cloud mask")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground truth (optional)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="predicted.png", help="Output image path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    engine = CloudRemovalInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    cloud_img = Image.open(args.image).convert("RGB")
    mask_img = Image.open(args.mask).convert("L")
    gt_img = Image.open(args.gt).convert("RGB") if args.gt else None

    result = engine.infer_with_metrics(cloud_img, mask_img, ground_truth=gt_img)

    # Save predicted image
    result["predicted"].save(args.output)
    print(f"Saved predicted image to: {args.output}")

    # Print metrics
    if "psnr" in result:
        print(f"PSNR: {result['psnr']:.2f} dB")
        print(f"SSIM: {result['ssim']:.4f}")
        print(f"MSE:  {result['mse']:.6f}")
        print(f"L1:   {result['l1']:.6f}")

    # Save comparison panel
    panel = engine.create_comparison_panel(cloud_img, mask_img, result["predicted"], gt_img)
    panel_path = Path(args.output).with_name("comparison_panel.png")
    panel.save(str(panel_path))
    print(f"Saved comparison panel to: {panel_path}")


if __name__ == "__main__":
    main()
