from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

from rice1_thin_cloud_losses import RICE1ThinCloudLosses


def estimate_atmospheric_light_dcp(I: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    """Estimates neutral atmospheric light vector A from top brightest pixels."""
    if I.ndim == 4:
        I_sample = I[0]
    else:
        I_sample = I
    c, h, w = I_sample.shape
    flat = I_sample.reshape(c, -1).transpose(0, 1)
    brightness = flat.mean(dim=1)
    k = min(500, flat.shape[0])
    _, idx = torch.topk(brightness, k=k)
    A_val = flat[idx].mean(dim=0).mean().item()
    A_val = float(np.clip(A_val, 0.75, 0.90))
    dev = I.device if hasattr(I, 'device') else torch.device('cpu')
    return torch.full((1, c, 1, 1), A_val, device=dev)


class RICE1ThinCloudDIP(nn.Module):
    """Enhanced DIP backbone with Physical Radiative Transfer Inversion for RICE-I."""

    def __init__(self, channels: int = 3, init_A: torch.Tensor | None = None):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.out_J = nn.Sequential(nn.Conv2d(64, channels, 1), nn.Sigmoid())
        self.out_t_raw = nn.Conv2d(64, 1, 1)

        if init_A is not None:
            init_A_safe = torch.clamp(init_A, 0.001, 0.999)
            self.A_param = nn.Parameter(torch.log(init_A_safe / (1.0 - init_A_safe)), requires_grad=True)
        else:
            init_val = torch.full((1, channels, 1, 1), 0.75)
            self.A_param = nn.Parameter(torch.log(init_val / (1.0 - init_val)), requires_grad=True)

    def forward(self, x):
        # x is cloudy image tensor I
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.dec1(e2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d1, e1], dim=1))
        
        # Neural direct clear radiance prediction J
        J = self.out_J(d2)
        
        # 1-channel spatial transmission map [B, 1, H, W] bounded in [0.10, 0.95]
        eps = 0.05
        t_1ch = eps + (1.0 - 2.0 * eps) * torch.sigmoid(self.out_t_raw(d2))
        t = t_1ch.expand(-1, x.shape[1], -1, -1)
        
        A = torch.sigmoid(self.A_param)

        self.last_features = {
            'early': e1.detach().cpu(),
            'middle': e2.detach().cpu(),
            'deep': d2.detach().cpu(),
            'magnitude': d1.detach().cpu()
        }

        return J, t, A


def _to_tensor(image: Image.Image, size: int | None = None) -> torch.Tensor:
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _mask_tensor(mask: Image.Image, size: int | None = None) -> torch.Tensor:
    if size is not None:
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    arr = (arr >= 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def _save_rgb(tensor: torch.Tensor, path: str | Path) -> None:
    arr = tensor.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0).round().astype(np.uint8), "RGB").save(path)


def _metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    p = pred.detach().cpu().clamp(0, 1)
    g = gt.detach().cpu().clamp(0, 1)
    mse = torch.mean((p - g) ** 2).item()
    mae = torch.mean(torch.abs(p - g)).item()
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
    return {"MSE": mse, "MAE": mae, "PSNR": float(psnr)}


def optimize_rice1_thin_cloud(
    cloudy: Image.Image,
    mask: Image.Image,
    gt: Image.Image | None = None,
    iterations: int = 2000,
    lr: float = 0.005,
    image_size: int = 256,
    device: str | None = None,
    output_dir: str | Path = "rice1_thin_cloud_outputs",
):
    """Zero-shot DIP inference. GT is used only after optimization for evaluation."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    I = _to_tensor(cloudy, image_size).unsqueeze(0).to(dev)
    M = _mask_tensor(mask, image_size).unsqueeze(0).to(dev)

    z = I.detach().clone()
    A_init = estimate_atmospheric_light_dcp(I).to(dev)

    model = RICE1ThinCloudDIP(channels=3, init_A=A_init).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = RICE1ThinCloudLosses()

    best_loss = float("inf")
    best_J = None
    best_t = None
    J, t, A = None, None, None

    start = time.perf_counter()
    pbar = tqdm(range(iterations), desc="RICE-I Thin Cloud DIP")
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        z_noisy = z + torch.randn_like(z) * 0.015
        J, t, A = model(z_noisy)

        loss_dict = losses.combined_loss(
            I=I,
            J=J,
            t=t,
            A=A,
            M=M,
            lambda_asm=0.5,
            lambda_rte=0.05,
            lambda_tv=1e-4,
            lambda_t_prior=0.10,
            lambda_sam=0.1,
            lambda_ndvi=0.05,
            lambda_ssim=0.1,
            alpha=1.3,
        )

        tot_l = loss_dict["total_loss"]
        tot_l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if tot_l.item() < best_loss:
            best_loss = tot_l.item()
            best_J = J.detach().clone()
            best_t = t.detach().clone()

        if step % 50 == 0 or step == iterations - 1:
            pbar.set_postfix(
                loss=f"{loss_dict['total_loss'].item():.5f}",
                recon=f"{loss_dict['l_recon'].item():.5f}",
                asm=f"{loss_dict['l_asm'].item():.5f}",
                rte=f"{loss_dict['l_rte'].item():.5f}",
            )

    elapsed = time.perf_counter() - start
    if best_J is not None:
        J_res = best_J
    elif J is not None:
        J_res = J
    else:
        J_res = I

    if best_t is not None:
        t_res = best_t
    elif t is not None:
        t_res = t
    else:
        t_res = torch.ones_like(I)

    if A is None:
        A = A_init

    _save_rgb(J_res, out_dir / "rice1_thin_cloud_restored.png")
    _save_rgb(I, out_dir / "rice1_thin_cloud_input.png")

    mask_arr = (M.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_arr, "L").save(out_dir / "rice1_thin_cloud_mask.png")

    result = {
        "device": str(dev),
        "iterations": iterations,
        "runtime_seconds": elapsed,
        "output": str(out_dir / "rice1_thin_cloud_restored.png"),
    }

    if gt is not None:
        GT = _to_tensor(gt, image_size).unsqueeze(0).to(dev)
        result.update(_metrics(J_res, GT))

    print(f"Saved: {result['output']}")
    print(f"Runtime: {elapsed:.2f} s")
    if gt is not None:
        print(f"PSNR: {result['PSNR']:.4f} dB | MAE: {result['MAE']:.6f} | MSE: {result['MSE']:.6f}")

    return J_res.cpu(), t_res.cpu(), A.cpu(), result


def optimize_rice1_thin_cloud_api(
    cloudy_image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    gt_image_tensor: torch.Tensor | None = None,
    num_iters: int = 2000,
    lr: float = 0.005,
    use_gpu: bool = True,
    callback=None,
    status_check=None,
    lambda_asm: float = 0.5,
    lambda_rte: float = 0.05,
    lambda_tv: float = 1e-4,
    lambda_t_prior: float = 0.10,
    lambda_sam: float = 0.1,
    lambda_ndvi: float = 0.05,
    lambda_ssim: float = 0.1,
    lambda_edge: float = 0.05,
    alpha: float = 1.3,
    **kwargs
):
    """
    API version of optimize_rice1_thin_cloud.
    """
    dev = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    I = cloudy_image_tensor.to(dev)
    if I.ndim == 2:
        I = I.unsqueeze(0).unsqueeze(0)
    elif I.ndim == 3:
        I = I.unsqueeze(0)
    
    M = mask_tensor.to(dev)
    if M.ndim == 2:
        M = M.unsqueeze(0).unsqueeze(0)
    elif M.ndim == 3:
        M = M.unsqueeze(0)

    if I.shape[1] > 3:
        I = I[:, :3, :, :]
    
    if M.shape[1] > 1:
        M = M[:, :1, :, :]

    z = I.detach().clone()
    A_init = estimate_atmospheric_light_dcp(I).to(dev)

    # Transmission guide from Dark Channel Prior for clean thin cloud dehazing
    min_c = torch.min(I, dim=1, keepdim=True)[0]
    dark_channel = -F.max_pool2d(-min_c, kernel_size=15, stride=1, padding=7)
    A_val = A_init.mean().item()
    t_dcp = torch.clamp(1.0 - 0.85 * (dark_channel / (A_val + 1e-4)), 0.40, 0.98)

    model = RICE1ThinCloudDIP(channels=3, init_A=A_init).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = RICE1ThinCloudLosses()

    best_loss = float("inf")
    best_J = None
    best_t = None
    J, t, A = None, None, None

    for step in range(1, num_iters + 1):
        if status_check:
            status_check()

        optimizer.zero_grad(set_to_none=True)
        z_noisy = z + torch.randn_like(z) * 0.015
        J, t, A = model(z_noisy)

        loss_dict = losses.combined_loss(
            I=I,
            J=J,
            t=t,
            A=A,
            M=M,
            t_dcp=t_dcp,
            lambda_asm=lambda_asm,
            lambda_rte=lambda_rte,
            lambda_tv=lambda_tv,
            lambda_t_prior=lambda_t_prior,
            lambda_sam=lambda_sam,
            lambda_ndvi=lambda_ndvi,
            lambda_ssim=lambda_ssim,
            lambda_edge=lambda_edge,
            alpha=alpha,
        )

        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_J = J.detach().clone()
            best_t = t.detach().clone()

        if callback:
            with torch.no_grad():
                callback(
                    step,
                    total_loss.item(),
                    {k: v.item() for k, v in loss_dict.items()},
                    J.detach().cpu()[0],
                    t.detach().cpu()[0],
                    A.detach().cpu()[0],
                    features=getattr(model, 'last_features', None)
                )

    if best_J is not None:
        J_final = best_J
    elif J is not None:
        J_final = J
    else:
        J_final = I

    if best_t is not None:
        t_final = best_t
    elif t is not None:
        t_final = t
    else:
        t_final = torch.ones_like(I)

    return J_final.detach().cpu()[0], t_final.detach().cpu()[0]

