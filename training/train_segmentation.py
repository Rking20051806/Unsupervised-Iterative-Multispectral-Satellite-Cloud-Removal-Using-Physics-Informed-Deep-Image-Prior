from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rice import RiceDataset
from metrics.segmentation_metrics import dice_score, iou_score
from models.segmentation.unet import UNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        if not bool(batch["has_mask"].all()):
            continue
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    model.eval()
    dices = []
    ious = []
    with torch.no_grad():
        for batch in loader:
            if not bool(batch["has_mask"].all()):
                continue
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()
            dices.append(dice_score(preds, masks).item())
            ious.append(iou_score(preds, masks).item())
    return {
        "dice": sum(dices) / max(len(dices), 1),
        "iou": sum(ious) / max(len(ious), 1),
    }


def main():
    dataset_root = PROJECT_ROOT / "RICE_DATASET"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RiceDataset(dataset_root, split="RICE2", image_size=256)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Starting training on {device} for 3 epochs...")
    for epoch in range(1, 4):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, train_loader, device)
        print(f"Epoch {epoch:02d}/10 | loss: {loss:.4f} | dice: {metrics['dice']:.4f} | iou: {metrics['iou']:.4f}")

    # Save model weights
    save_path = PROJECT_ROOT / "models" / "segmentation" / "unet_rice2.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Trained model saved successfully to: {save_path}")


if __name__ == "__main__":
    main()

