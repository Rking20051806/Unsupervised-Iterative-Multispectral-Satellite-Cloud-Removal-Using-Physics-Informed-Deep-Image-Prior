from __future__ import annotations

import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (pred * target).sum(dim=1)
    return ((2.0 * intersection + epsilon) / (pred.sum(dim=1) + target.sum(dim=1) + epsilon)).mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + epsilon) / (union + epsilon)).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, epsilon: float = 1e-7) -> torch.Tensor:
    mse = torch.mean((pred.float() - target.float()) ** 2)
    return 20.0 * torch.log10(torch.tensor(max_val)) - 10.0 * torch.log10(mse + epsilon)
