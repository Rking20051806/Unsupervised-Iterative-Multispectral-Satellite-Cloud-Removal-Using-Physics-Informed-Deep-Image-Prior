"""
CloudRemovalTrainer — Supervised training pipeline for the CloudRemovalUNet.

Implements the training methodology from Nikita Zade's ISRO Internship Report:
  - Combined loss: 0.7 × MSE + 0.3 × L1
  - Optimizer: Adam (lr=1e-4)
  - Scheduler: ReduceLROnPlateau (patience=2)
  - Metrics: Training/Validation loss, PSNR, SSIM per epoch
  - Checkpointing: Best model + per-epoch checkpoints
  - Visualization: 4-panel comparison plots saved each epoch
  - TensorBoard logging via SummaryWriter

Usage:
    python training/train_cloud_removal.py [--epochs 50] [--batch_size 8] [--lr 0.0001]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cloud_removal.cloud_removal_unet import CloudRemovalUNet
from datasets.cloud_removal_dataset import get_dataloaders


class CombinedLoss(nn.Module):
    """Combined loss: 0.7 × MSE + 0.3 × L1.

    MSE ensures overall radiometric accuracy;
    L1 preserves sharp edges and fine details.
    """

    def __init__(self, mse_weight: float = 0.7, l1_weight: float = 0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(pred, target) + self.l1_weight * self.l1(pred, target)


def calculate_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def calculate_ssim_simple(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate SSIM (simplified implementation using skimage if available)."""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(pred, target, channel_axis=2, data_range=1.0))
    except ImportError:
        # Simplified SSIM fallback
        mu_p = pred.mean()
        mu_t = target.mean()
        sigma_p = pred.std()
        sigma_t = target.std()
        sigma_pt = ((pred - mu_p) * (target - mu_t)).mean()
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / \
               ((mu_p ** 2 + mu_t ** 2 + c1) * (sigma_p ** 2 + sigma_t ** 2 + c2))
        return float(ssim)


class CloudRemovalTrainer:
    """Complete training pipeline for the supervised cloud removal U-Net.

    Attributes:
        model:     CloudRemovalUNet model instance
        device:    Target device (cuda/cpu)
        criterion: Combined 0.7×MSE + 0.3×L1 loss
        optimizer: Adam optimizer
        scheduler: ReduceLROnPlateau scheduler
    """

    def __init__(
        self,
        dataset_root: str | Path,
        epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-4,
        image_size: int = 256,
        device: str | None = None,
        checkpoint_dir: str | Path | None = None,
        viz_dir: str | Path | None = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Training device: {self.device}")

        # Directories
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else PROJECT_ROOT / "models" / "cloud_removal"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.viz_dir = Path(viz_dir) if viz_dir else PROJECT_ROOT / "outputs" / "training_viz"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Data loaders
        self.train_loader, self.val_loader = get_dataloaders(
            root_dir=dataset_root,
            image_size=image_size,
            batch_size=batch_size,
            val_split=0.2,
            num_workers=0,  # Windows compatibility
            seed=42,
        )

        # Model
        self.model = CloudRemovalUNet(in_channels=4, out_channels=3).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"CloudRemovalUNet parameters: {param_count:,}")

        # Loss, optimizer, scheduler (from the report)
        self.criterion = CombinedLoss(mse_weight=0.7, l1_weight=0.3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.5, verbose=True,
        )

        # TensorBoard
        self.writer: "SummaryWriter | None" = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(PROJECT_ROOT / "runs" / "cloud_removal"))
            print(f"TensorBoard logging to: {PROJECT_ROOT / 'runs' / 'cloud_removal'}")
        except Exception:
            self.writer = None
            print("TensorBoard not available — skipping logging.")

        # Training history
        self.history: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_val_loss = float("inf")

    def train_one_epoch(self, epoch: int) -> dict:
        """Train for a single epoch.

        Returns:
            Dictionary with train_loss, mse_loss, l1_loss.
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0
        num_batches = 0

        for batch in self.train_loader:
            inputs = batch["input"].to(self.device)   # [B, 4, H, W]
            targets = batch["target"].to(self.device)  # [B, 3, H, W]

            self.optimizer.zero_grad()
            outputs = self.model(inputs)               # [B, 3, H, W]

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_mse += nn.functional.mse_loss(outputs, targets).item()
                total_l1 += nn.functional.l1_loss(outputs, targets).item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "train_loss": total_loss / n,
            "mse_loss": total_mse / n,
            "l1_loss": total_l1 / n,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate the model on the validation set.

        Returns:
            Dictionary with val_loss, psnr, ssim.
        """
        self.model.eval()
        total_loss = 0.0
        psnr_list: list[float] = []
        ssim_list: list[float] = []
        num_batches = 0

        for batch in self.val_loader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()

            # Compute PSNR and SSIM per sample
            for i in range(outputs.shape[0]):
                pred_np = outputs[i].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
                target_np = targets[i].cpu().numpy().transpose(1, 2, 0)
                pred_np = np.clip(pred_np, 0.0, 1.0)
                target_np = np.clip(target_np, 0.0, 1.0)

                psnr_list.append(calculate_psnr(pred_np, target_np))
                ssim_list.append(calculate_ssim_simple(pred_np, target_np))

            num_batches += 1

        n = max(num_batches, 1)
        return {
            "val_loss": total_loss / n,
            "psnr": np.mean(psnr_list) if psnr_list else 0.0,
            "ssim": np.mean(ssim_list) if ssim_list else 0.0,
        }

    @torch.no_grad()
    def save_visualization(self, epoch: int):
        """Save 4-panel comparison visualization (cloudy / mask / ground truth / predicted).

        Saves to outputs/training_viz/epoch_XX.png
        """
        self.model.eval()
        batch = next(iter(self.val_loader))
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        outputs = self.model(inputs)

        # Pick first 4 samples (or fewer)
        n_samples = min(4, inputs.shape[0])
        fig_height = n_samples * 256
        fig_width = 4 * 256

        canvas = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)

        for i in range(n_samples):
            row_start = i * 256

            # Column 0: Cloudy input (first 3 channels)
            cloudy_np = inputs[i, :3].cpu().numpy().transpose(1, 2, 0)
            cloudy_np = (np.clip(cloudy_np, 0, 1) * 255).astype(np.uint8)
            canvas[row_start:row_start + 256, 0:256] = cloudy_np

            # Column 1: Cloud mask (4th channel, colorized)
            mask_np = inputs[i, 3].cpu().numpy()
            mask_rgb = np.stack([
                (mask_np * 255).astype(np.uint8),
                np.zeros_like(mask_np, dtype=np.uint8),
                np.zeros_like(mask_np, dtype=np.uint8),
            ], axis=-1)
            canvas[row_start:row_start + 256, 256:512] = mask_rgb

            # Column 2: Ground truth
            gt_np = targets[i].cpu().numpy().transpose(1, 2, 0)
            gt_np = (np.clip(gt_np, 0, 1) * 255).astype(np.uint8)
            canvas[row_start:row_start + 256, 512:768] = gt_np

            # Column 3: Predicted output
            pred_np = outputs[i].cpu().numpy().transpose(1, 2, 0)
            pred_np = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
            canvas[row_start:row_start + 256, 768:1024] = pred_np

        viz_path = self.viz_dir / f"epoch_{epoch:03d}.png"
        Image.fromarray(canvas, mode="RGB").save(str(viz_path))

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.

        Saves:
            - Per-epoch checkpoint: checkpoint_epoch_XX.pth
            - Best model: best_model.pth (when is_best=True)
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }

        # Per-epoch checkpoint
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, str(epoch_path))

        # Best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, str(best_path))
            print(f"  ★ Best model saved (val_loss={self.best_val_loss:.6f})")

    def train(self):
        """Run the full training loop.

        Trains for self.epochs epochs, logging metrics, saving checkpoints,
        and generating visualizations each epoch.
        """
        print(f"\n{'='*70}")
        print(f"  Starting Training: {self.epochs} epochs, batch_size={self.batch_size}, lr={self.lr}")
        print(f"  Loss: 0.7 × MSE + 0.3 × L1")
        print(f"  Optimizer: Adam | Scheduler: ReduceLROnPlateau(patience=2)")
        print(f"{'='*70}\n")

        total_start = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Scheduler step
            self.scheduler.step(val_metrics["val_loss"])

            # Check for best model
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            # Record history
            epoch_record = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
                "time": time.time() - epoch_start,
            }
            self.history.append(epoch_record)

            # Print summary
            elapsed = epoch_record["time"]
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.6f} | "
                f"Val Loss: {val_metrics['val_loss']:.6f} | "
                f"PSNR: {val_metrics['psnr']:.2f} dB | "
                f"SSIM: {val_metrics['ssim']:.4f} | "
                f"LR: {epoch_record['lr']:.2e} | "
                f"{elapsed:.1f}s"
                f"{' ★' if is_best else ''}"
            )

            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
                self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
                self.writer.add_scalar("Loss/mse", train_metrics["mse_loss"], epoch)
                self.writer.add_scalar("Loss/l1", train_metrics["l1_loss"], epoch)
                self.writer.add_scalar("Metrics/PSNR", val_metrics["psnr"], epoch)
                self.writer.add_scalar("Metrics/SSIM", val_metrics["ssim"], epoch)
                self.writer.add_scalar("LR", epoch_record["lr"], epoch)

            # Save checkpoint and visualization
            self.save_checkpoint(epoch, is_best=is_best)
            self.save_visualization(epoch)

        total_time = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"  Training complete in {total_time / 3600:.2f} hours ({total_time:.1f}s)")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")
        if self.history:
            best = max(self.history, key=lambda x: x["psnr"])
            print(f"  Best PSNR: {best['psnr']:.2f} dB (epoch {best['epoch']})")
            best_ssim = max(self.history, key=lambda x: x["ssim"])
            print(f"  Best SSIM: {best_ssim['ssim']:.4f} (epoch {best_ssim['epoch']})")
        print(f"{'='*70}")

        if self.writer is not None:
            self.writer.close()


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train CloudRemovalUNet on RICE-II dataset")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "RICE_DATASET" / "RICE2"),
                        help="Path to RICE-II dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=256, help="Image resize dimension")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    trainer = CloudRemovalTrainer(
        dataset_root=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        device=args.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
