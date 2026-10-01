import argparse
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image

# Setup sys path relative to project root
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.cloud_removal_dataset import get_dataloaders
from models.cloud_removal_advanced.cloud_removal_unet_advanced import CloudRemovalUNetAdvanced
from models.cloud_removal_advanced.advanced_losses import PerceptualLoss, SSIMLoss, EdgeLoss
from training.train_cloud_removal import CombinedLoss  # reuse the original pixel loss

class AdvancedCloudRemovalTrainer:
    def __init__(self, dataset_dir: str, epochs: int = 50, batch_size: int = 8, lr: float = 1e-4, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"Training device: {self.device}")

        # Initialize loaders
        self.train_loader, self.val_loader = get_dataloaders(
            root_dir=dataset_dir,
            batch_size=batch_size,
            image_size=256,
            num_workers=4
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # Model setup
        self.model = CloudRemovalUNetAdvanced(in_channels=4, out_channels=3).to(self.device)
        print(f"CloudRemovalUNetAdvanced parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss and optimization setups
        self.pixel_loss_fn = CombinedLoss(mse_weight=0.7, l1_weight=0.3)
        self.ssim_loss_fn = SSIMLoss().to(self.device)
        self.edge_loss_fn = EdgeLoss().to(self.device)
        
        # Load VGG dynamically, handle CPU/GPU cleanly
        self.perceptual_loss_fn: "PerceptualLoss | None" = None
        try:
            self.perceptual_loss_fn = PerceptualLoss().to(self.device)
            print("Loaded VGG Perceptual Loss successfully.")
        except Exception as e:
            self.perceptual_loss_fn = None
            print(f"VGG Perceptual Loss loading failed: {e}. Skipping perceptual loss.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.5
        )

        # Output Directories
        self.checkpoint_dir = PROJECT_ROOT / "models" / "cloud_removal_advanced"
        self.viz_dir = PROJECT_ROOT / "outputs" / "training_viz_advanced"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        self.history = []
        self.best_val_loss = float("inf")

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_pixel = 0.0
        total_ssim = 0.0
        total_perceptual = 0.0
        total_edge = 0.0

        from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch:3d}", leave=False)
        for i, batch in pbar:
            # Input shape: (B, 4, 256, 256) -> [RGB + Cloud Mask]
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)

            # Calculate individual loss terms
            l_pix = self.pixel_loss_fn(pred, y)
            l_ssim = self.ssim_loss_fn(pred, y)
            l_edge = self.edge_loss_fn(pred, y)

            # Combined total objective:
            # loss = 0.5 * pixel + 0.2 * ssim + 0.1 * edge + 0.2 * perceptual
            loss = 0.5 * l_pix + 0.2 * l_ssim + 0.1 * l_edge

            l_percep_val = 0.0
            if self.perceptual_loss_fn is not None:
                l_percep = self.perceptual_loss_fn(pred, y)
                loss += 0.2 * l_percep
                l_percep_val = l_percep.item()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_pixel += l_pix.item()
            total_ssim += l_ssim.item()
            total_edge += l_edge.item()
            total_perceptual += l_percep_val

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        n = len(self.train_loader)
        return {
            "train_loss": total_loss / n,
            "pixel_loss": total_pixel / n,
            "ssim_loss": total_ssim / n,
            "edge_loss": total_edge / n,
            "perceptual_loss": total_perceptual / n
        }

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        psnr_acc = 0.0
        ssim_acc = 0.0

        for batch in self.val_loader:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)
            pred = self.model(x)

            l_pix = self.pixel_loss_fn(pred, y)
            l_ssim = self.ssim_loss_fn(pred, y)
            l_edge = self.edge_loss_fn(pred, y)
            loss = 0.5 * l_pix + 0.2 * l_ssim + 0.1 * l_edge
            if self.perceptual_loss_fn is not None:
                loss += 0.2 * self.perceptual_loss_fn(pred, y)

            total_loss += loss.item()

            # Metric Calculations
            for p, t in zip(pred, y):
                mse = F.mse_loss(p, t).item()
                psnr_acc += 20 * torch.log10(1.0 / torch.tensor(mse).sqrt()).item() if mse > 1e-10 else 100.0
                
                # Standard batch structural similarity
                ssim_acc += (1.0 - self.ssim_loss_fn(p.unsqueeze(0), t.unsqueeze(0)).item())

        # Safely count samples matching Sized protocol
        from collections.abc import Sized
        dataset = self.val_loader.dataset
        n_samples = len(dataset) if isinstance(dataset, Sized) else 1
        return {
            "val_loss": total_loss / len(self.val_loader),
            "psnr": psnr_acc / max(1, n_samples),
            "ssim": ssim_acc / max(1, n_samples)
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        
        # Save standard checkpoints
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, str(epoch_path))

        if is_best:
            best_path = self.checkpoint_dir / "best_model_advanced.pth"
            torch.save(checkpoint, str(best_path))
            print(f"  ★ Best model saved (val_loss={self.best_val_loss:.6f})")

    def save_visualization(self, epoch: int):
        self.model.eval()
        batch = next(iter(self.val_loader))
        x = batch["input"][:4].to(self.device)
        y = batch["target"][:4].to(self.device)

        with torch.no_grad():
            pred = self.model(x)

        # 4-panel visual comparison compilation
        cloudy = x[:, :3]
        mask = x[:, 3:4].expand(-1, 3, -1, -1)
        grid = torch.cat([cloudy, mask, y, pred], dim=0)
        grid_img = make_grid(grid, nrow=4, normalize=True, scale_each=True)

        ndarr = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(self.viz_dir / f"visual_epoch_{epoch:03d}.png")

    def train(self):
        print(f"\n{'='*80}")
        print(f"  Starting Advanced Training: {self.epochs} epochs | batch_size={self.batch_size}")
        print(f"  Loss structure: 0.5×Pixel + 0.2×SSIM + 0.1×Sobel + 0.2×VGG")
        print(f"{'='*80}")

        total_start = time.time()
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            elapsed = time.time() - t0

            # Learning rate decay step
            self.scheduler.step(val_metrics["val_loss"])
            curr_lr = self.optimizer.param_groups[0]["lr"]

            # Save stats
            epoch_record = {
                "epoch": epoch,
                "lr": curr_lr,
                "elapsed": elapsed,
                **train_metrics,
                **val_metrics
            }
            self.history.append(epoch_record)

            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.6f} | "
                f"Val Loss: {val_metrics['val_loss']:.6f} | "
                f"PSNR: {val_metrics['psnr']:.2f} dB | "
                f"SSIM: {val_metrics['ssim']:.4f} | "
                f"LR: {curr_lr:.2e} | {elapsed:.1f}s"
                f"{' ★' if is_best else ''}",
                flush=True
            )

            self.save_checkpoint(epoch, is_best=is_best)
            self.save_visualization(epoch)

        total_time = time.time() - total_start
        print(f"\n{'='*80}")
        print(f"  Advanced Training complete in {total_time / 3600:.2f} hours")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")
        print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Train Feature-guided Advanced CloudRemovalUNet")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "RICE_DATASET" / "RICE2"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    trainer = AdvancedCloudRemovalTrainer(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
    trainer.train()

if __name__ == "__main__":
    main()
