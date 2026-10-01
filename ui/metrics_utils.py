from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref: Image.Image, pred: Image.Image) -> float:
    """Calculate PSNR between reference and predicted images."""
    ref_arr = np.asarray(ref, dtype=np.float32) / 255.0
    pred_arr = np.asarray(pred, dtype=np.float32) / 255.0
    return float(peak_signal_noise_ratio(ref_arr, pred_arr, data_range=1.0))


def ssim(ref: Image.Image, pred: Image.Image) -> float:
    """Calculate SSIM between reference and predicted images."""
    ref_arr = np.asarray(ref, dtype=np.float32) / 255.0
    pred_arr = np.asarray(pred, dtype=np.float32) / 255.0
    if len(ref_arr.shape) == 3:
        return float(structural_similarity(ref_arr, pred_arr, channel_axis=2, data_range=1.0))
    else:
        return float(structural_similarity(ref_arr, pred_arr, data_range=1.0))


def dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-7) -> float:
    """Calculate Dice coefficient between prediction and target masks."""
    pred_flat = pred.flatten().astype(np.float32)
    target_flat = target.flatten().astype(np.float32)
    intersection = np.sum(pred_flat * target_flat)
    return float((2.0 * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth))


def iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-7) -> float:
    """Calculate Intersection over Union between prediction and target masks."""
    pred_flat = pred.flatten().astype(np.float32)
    target_flat = target.flatten().astype(np.float32)
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat) - intersection
    return float((intersection + smooth) / (union + smooth))


def calc_metrics(
    pred_image: Image.Image,
    ref_image: Image.Image,
    pred_mask: Image.Image | None = None,
    ref_mask: Image.Image | None = None,
) -> dict[str, float]:
    """Calculate all available metrics."""
    metrics: dict[str, float] = {}

    metrics["psnr"] = psnr(ref_image, pred_image)
    metrics["ssim"] = ssim(ref_image, pred_image)

    if pred_mask is not None and ref_mask is not None:
        pred_arr = np.asarray(pred_mask.convert("L"), dtype=np.float32) / 255.0 > 0.5
        ref_arr = np.asarray(ref_mask.convert("L"), dtype=np.float32) / 255.0 > 0.5
        metrics["dice"] = dice(pred_arr, ref_arr)
        metrics["iou"] = iou(pred_arr, ref_arr)

    return metrics
