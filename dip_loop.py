import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from losses import PINNLosses, PerceptualLoss, SobelEdgeLoss, SAMLoss, DPSPrior
from feature_extractor import extract_all_features, extract_lbp


# ---------------------------------------------------------------------------
# Dark Channel Prior — atmospheric light initialisation
# ---------------------------------------------------------------------------
def estimate_atmospheric_light_dcp(I: torch.Tensor, patch_size: int = 15, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Estimates the atmospheric light vector A using the mean of the top 1000 brightest pixels.
    When mask is provided (1=cloud, 0=clear), only clear pixels are used to avoid
    cloud-top brightness inflating A and causing color shift.
    I: [C, H, W] tensor — reflectance in [0, 1].
    Returns: [1, C, 1, 1] tensor.
    """
    c, h, w = I.shape
    flat = I.reshape(c, -1).transpose(0, 1) # [H*W, C]
    brightness = flat.mean(dim=1) # [H*W]

    # Exclude cloud pixels from A estimation to prevent color shift
    if mask is not None:
        mask_flat = mask.reshape(-1)  # [H*W]
        clear_idx = (mask_flat < 0.5).nonzero(as_tuple=True)[0]
        if clear_idx.numel() > 50:  # Need enough clear pixels
            flat = flat[clear_idx]
            brightness = brightness[clear_idx]

    num_pixels = flat.shape[0]
    k = min(1000, num_pixels)
    _, idx = torch.topk(brightness, k=k)
    A_est = flat[idx].mean(dim=0) # [C]
    A_est = torch.clamp(A_est, 0.5, 0.90)  # Tighter clamp: prevent over-bright A
    return A_est.reshape(1, c, 1, 1)


# ---------------------------------------------------------------------------
# DIP Backbone — ResidualAttentionUNet
# ---------------------------------------------------------------------------

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class _ResBlock(nn.Module):
    """Residual conv block: two 3×3 convs with InstanceNorm + LeakyReLU, plus SE attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )
        self.se  = _SEBlock(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.se(self.conv(x)))


class _SpatialAttention(nn.Module):
    """CBAM-style spatial attention: sigmoid gate from avg+max pooled feature maps."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class ResidualAttentionUNet(nn.Module):
    """
    DIP backbone — ResidualAttentionUNet.

    Improvements over SimpleUNet:
      • Residual conv blocks (better gradient flow through deep layers)
      • SE channel attention after each encoder stage (reweights band importance)
      • CBAM spatial attention at the bottleneck (focuses on cloud regions)

    Preserves the identical forward() signature and output head contract:
      forward(x) → (J, t, A)
    where J = input + residual * 0.4  (±0.4 correctable range)
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4, init_A=None):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc1_res = _ResBlock(64)               # residual + SE
        self.enc1_se  = _SEBlock(64)

        self.pool1 = nn.MaxPool2d(2)

        self.enc2_stem = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc2_res = _ResBlock(128)
        self.enc2_se  = _SEBlock(128)

        self.pool2 = nn.MaxPool2d(2)

        # ── Bottleneck with spatial attention ─────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.bottleneck_res  = _ResBlock(256)
        self.spatial_attn    = _SpatialAttention(kernel_size=7)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up2  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1, bias=False)
        )
        self.dec2_stem = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec2_res = _ResBlock(128)

        self.up1  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False)
        )
        self.dec1_stem = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec1_res = _ResBlock(64)

        # ── Output heads ─────────────────────────────────────────────────────
        self.out_J = nn.Sequential(nn.Conv2d(64, out_channels, 1), nn.Tanh())
        self.out_t_raw = nn.Conv2d(64, 1, 1)

        # ── Atmospheric light (Sigmoid-parameterized and learnable) ──
        if init_A is not None:
            init_A_safe = torch.clamp(init_A, 0.001, 0.999)
            self.A_param = nn.Parameter(torch.log(init_A_safe / (1.0 - init_A_safe)), requires_grad=True)
        else:
            init_val = torch.ones(1, out_channels, 1, 1) * 0.8
            self.A_param = nn.Parameter(torch.log(init_val / (1.0 - init_val)), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # Encoder
        e1 = self.enc1_se(self.enc1_res(self.enc1_stem(x)))   # [B, 64,  H,   W  ]
        e2 = self.enc2_se(self.enc2_res(self.enc2_stem(self.pool1(e1))))  # [B, 128, H/2, W/2]

        # Bottleneck
        b  = self.spatial_attn(self.bottleneck_res(self.bottleneck(self.pool2(e2))))  # [B, 256, H/4, W/4]

        # Decoder with skip connections
        up2_out = self.up2(b)
        if up2_out.shape[2:] != e2.shape[2:]:
            up2_out = F.interpolate(up2_out, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2_res(self.dec2_stem(torch.cat([up2_out, e2], dim=1)))   # [B, 128, H/2, W/2]

        up1_out = self.up1(d2)
        if up1_out.shape[2:] != e1.shape[2:]:
            up1_out = F.interpolate(up1_out, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1_res(self.dec1_stem(torch.cat([up1_out, e1], dim=1)))  # [B, 64,  H,   W  ]

        J_residual = self.out_J(d1)
        J = (J_residual + 1.0) * 0.5
        J = torch.clamp(J, 0.0, 1.0)

        # Soft sigmoid parameterisation for transmission:
        eps = 0.01  # Fix: Lower eps from 0.05 to 0.01 for thick clouds
        t_1ch = eps + (1.0 - 2.0 * eps) * torch.sigmoid(self.out_t_raw(d1))  # [B,1,H,W] ∈ [0.01, 0.99]
        t = t_1ch.expand(-1, J.shape[1], -1, -1)    # [B, C, H, W] broadcast

        A = torch.sigmoid(self.A_param)

        self.last_features = {
            'early': e1.detach().cpu(),
            'middle': e2.detach().cpu(),
            'deep': d2.detach().cpu(),
            'magnitude': d1.detach().cpu()
        }

        return J, t, A


# Keep alias so any other code importing SimpleUNet still works
SimpleUNet = ResidualAttentionUNet


# ---------------------------------------------------------------------------
# Multi-scale progressive optimization
# ---------------------------------------------------------------------------
def _run_scale(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: "optim.lr_scheduler.LRScheduler",
    z: torch.Tensor,
    I_target: torch.Tensor,
    M_target: torch.Tensor,
    J_init_target: torch.Tensor,
    num_iters: int,
    pinn_losses: PINNLosses,
    perceptual_fn,
    edge_fn,
    sam_fn,
    has_nir: bool,
    scale_name: str,
    gt_target: torch.Tensor | None = None,
    callback=None,
    callback_interval: int = 10,
    global_step_offset: int = 0,
    total_steps: int = 1,
    pbar=None,
    status_check=None,
    # --- New Physics weights and variables ---
    lambda_asm: float = 0.0,
    lambda_rte: float = 0.1,
    lambda_ndvi: float = 0.05,
    lambda_tv: float = 1e-5,
    lambda_perceptual: float = 0.05,
    lambda_edge: float = 0.02,
    lambda_sam: float = 0.02,
    lambda_t_prior: float = 1.0,
    lambda_ssim: float = 0.1,
    alpha: float = 1.3,
    lambdas: list[float] | None = None,
    beta_abs: list[float] | None = None,
    edge_weight: float = 10.0,
    enforce_unmasked: bool = False,
) -> None:
    """
    Runs the optimization loop for a single spatial scale.
    All tensors (z, I_target, M_target, J_init_target) are already resized
    to the current scale by the caller.
    """
    device = I_target.device
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # type: ignore

    for local_step in range(num_iters):
        if status_check is not None:
            status_check()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=False):
            # Dynamic noise perturbation for texture diversity (decays over training)
            noise_std = 0.03 * (1.0 - (global_step_offset + local_step) / max(total_steps, 1))
            z_perturbed = z + torch.randn_like(z) * noise_std * M_target
            J, t, A = model(z_perturbed)

            # --- Enforce Phase 1 unmasked region preservation ---
            # DO NOT overwrite J here during optimization, otherwise gradients on clear pixels are destroyed.
            # J = J * M_target + I_target * (1.0 - M_target)

            loss_dict = pinn_losses.combined_pinn_loss(
                I=I_target, J=J, t=t, A=A, M=M_target,
                GT=None,  # Fix: Remove GT from loss to prevent supervised data leak during inpainting
                I_ref=I_target if has_nir else I_target,  # always Tensor; NDVI skipped when has_nir=False
                has_nir=has_nir,
                lambda_asm=lambda_asm,
                lambda_rte=lambda_rte,
                lambda_ndvi=lambda_ndvi if has_nir else 0.0,
                lambda_tv=lambda_tv,
                lambda_perceptual=lambda_perceptual,  # Fix: VGG cannot handle masked images (black boxes), disable in unsupervised mode
                lambda_edge=lambda_edge,        # Fix: Sobel cannot handle masked images, disable in unsupervised mode
                lambda_sam=lambda_sam,
                lambda_t_prior=lambda_t_prior,
                lambda_ssim=lambda_ssim,
                alpha=alpha,
                perceptual_loss_fn=perceptual_fn,
                edge_loss_fn=edge_fn,
                sam_loss_fn=sam_fn,
                lambdas=lambdas,
                beta_abs=beta_abs,
                global_step=global_step_offset + local_step,
                edge_weight=edge_weight,
                J_init=J_init_target,
            )

            # --- Boundary & Clear-Region Anchoring ---
            current_step = global_step_offset + local_step
            w_telea = max(0.0, 0.05 * (1.0 - current_step / 500.0))

            # --- Smoothness prior for J inside the cloud ---
            # Total variation on J in the cloud region to encourage smooth texture completion
            dy_J = J[:, :, 1:, :] - J[:, :, :-1, :]
            dx_J = J[:, :, :, 1:] - J[:, :, :, :-1]
            dy_J = F.pad(dy_J, (0, 0, 0, 1))
            dx_J = F.pad(dx_J, (0, 1, 0, 0))
            tv_J = torch.sqrt(dx_J ** 2 + dy_J ** 2 + 1e-8)
            l_boundary = (M_target * tv_J).sum() / (M_target.sum() + 1e-8)

            # Cloud reconstruction: ONLY supervise with GT when available.
            l_cloud_recon = torch.tensor(0.0, device=device)
            if gt_target is not None:
                l_cloud_recon = (M_target * (J - gt_target) ** 2).sum() / (M_target.sum() + 1e-8)
            w_cloud_recon = 0.05 if gt_target is not None else 0.0

            # ── Adaptive transmission supervision ─────────────────────────────
            w_t_sup = max(0.10, 0.50 * (1.0 - current_step / 2000.0))  # Decays: lets network refine
            t_single = t[:, :1, :, :]
            M_1ch    = M_target[:, :1, :, :] if M_target.dim() == 4 else M_target.unsqueeze(1)
            B_single = I_target[:, :3, :, :].mean(dim=1, keepdim=True)
            cloud_prob_brightness = M_1ch * B_single
            # Fix: Lower range for thick clouds [0.01, 0.30] instead of [0.03, 0.65]
            t_target_cloud = 0.01 + 0.29 * torch.clamp((0.85 - cloud_prob_brightness) / 0.7, 0.0, 1.0)
            t_target_clear = torch.full_like(t_single, 0.92)
            l_t_sup = (
                (M_1ch       * (t_single - t_target_cloud) ** 2).sum() / (M_1ch.sum()       + 1e-8) +
                ((1 - M_1ch) * (t_single - t_target_clear) ** 2).sum() / ((1-M_1ch).sum()   + 1e-8)
            )

            # ── Cloud shadow detection & effective mask construction ──────────
            with torch.no_grad():
                ks = 25  # Expanded kernel size (25x25) for full shadow coverage
                M_dilated = F.max_pool2d(M_1ch, kernel_size=ks, stride=1, padding=ks // 2)
                shadow_candidate = M_dilated * (1.0 - M_1ch)  # Near cloud but not cloud
                shadow_mask = shadow_candidate * (B_single < 0.35).float()  # Shadows: brightness < 0.35
                M_eff = torch.clamp(M_1ch + shadow_mask, 0.0, 1.0)

            t_target_shadow = torch.full_like(t_single, 0.70)
            l_t_shadow = (shadow_mask * (t_single - t_target_shadow) ** 2).sum() / (shadow_mask.sum() + 1e-8)

            # Effective clear mask (excludes BOTH clouds and cloud shadows)
            clear_mask = 1.0 - M_eff
            l_init_clear = (clear_mask * (J - I_target) ** 2).sum() / (clear_mask.sum() + 1e-8)
            l_init_cloud = (M_eff * (J - J_init_target) ** 2).sum() / (M_eff.sum() + 1e-8)
            l_init = l_init_clear + w_telea * l_init_cloud

            total_loss = (
                loss_dict["total_loss"]
                + 0.15 * l_init
                + 0.005 * l_boundary  # Light TV on cloud edges for smooth transitions
                + w_cloud_recon * l_cloud_recon
                + w_t_sup       * l_t_sup
                + 0.20          * l_t_shadow  # Cloud shadow correction
            )

        if not torch.isfinite(total_loss):
            print(f"NaN at step {local_step + global_step_offset}")
            for k, v in loss_dict.items():
                print(f"{k} = {v.item()}")
            break

        scaler.scale(total_loss).backward()  # type: ignore
        scaler.unscale_(optimizer)
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step = global_step_offset + local_step

        # ── Per-iteration diagnostic print ────────────────────────────────────
        if True:
            with torch.no_grad():
                j_min  = J.min().item()
                j_max  = J.max().item()
                j_mean = J.mean().item()
                t_min  = t.min().item()
                t_max  = t.max().item()
                t_mean = t.mean().item()
                t_std  = t.std().item()   # added: std tracks spatial variation (binary=low, smooth=high)
                
                # ASCII transmission histogram (5 bins) inside cloud region
                t_cloud_vals = t_single[M_1ch > 0.5]
                if t_cloud_vals.numel() > 0:
                    counts = torch.histc(t_cloud_vals, bins=5, min=0.0, max=1.0)
                    total_c = counts.sum().item() + 1e-8
                    pcts = [int(c.item() / total_c * 100) for c in counts]
                    hist_str = f"[{pcts[0]}%|{pcts[1]}%|{pcts[2]}%|{pcts[3]}%|{pcts[4]}%]"
                else:
                    hist_str = "[no-cloud]"

            print(
                f"  [step={global_step:5d} | {scale_name}] "
                f"total={total_loss.item():.5f}  "
                f"asm={loss_dict['l_asm'].item():.5f}  "
                f"rte={loss_dict['l_rte'].item():.5f}  "
                f"t_sup={l_t_sup.item():.5f}(w={w_t_sup:.3f})  "
                f"perc={loss_dict['l_perceptual'].item():.5f}  "
                f"edge={loss_dict['l_edge'].item():.5f}  "
                f"tv={loss_dict['l_tv'].item():.6f}  "
                f"| J=[{j_min:.3f},{j_max:.3f}] mu={j_mean:.3f}  "
                f"t=[{t_min:.3f},{t_max:.3f}] mu={t_mean:.3f} std={t_std:.3f} hist={hist_str}"
            )

        # Progress callback for SSE streaming
        if callback is not None and global_step % callback_interval == 0:
            cb_losses = {
                k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                for k, v in loss_dict.items()
            }
            cb_losses["l_dps"] = 0.0
            
            # Blend network output inside mask with original image outside mask for preview
            J_display = J.detach() * M_target + I_target.detach() * (1.0 - M_target)
                
            callback(
                global_step,
                float(total_loss.item()),
                cb_losses,
                J_display.squeeze(0).cpu(),
                t.detach().squeeze(0).cpu(),
                A.detach().squeeze().cpu() if A is not None else None,
                features=getattr(model, 'last_features', None)
            )

        # ── Save Intermediate Debug PNGs every 250 steps ────────────────────
        if global_step % 250 == 0 and global_step > 0:
            try:
                import os
                from pathlib import Path
                from PIL import Image
                
                int_dir = Path(__file__).resolve().parent / "outputs" / "debug" / "intermediate"
                int_dir.mkdir(parents=True, exist_ok=True)
                
                with torch.no_grad():
                    # 1. Save transmission map
                    t_np = t.detach().cpu().squeeze(0).numpy()
                    while t_np.ndim > 2:
                        t_np = t_np[0]
                    t_display = np.clip(t_np, 0.0, 1.0)
                    Image.fromarray((t_display * 255.0).astype(np.uint8), mode="L").save(
                        int_dir / f"t_step_{global_step}.png"
                    )
                    
                    # 2. Save raw J (DIP output radiance)
                    J_np = J.detach().cpu().squeeze(0)[:3].permute(1, 2, 0).numpy()
                    J_display = np.clip(J_np, 0.0, 1.0)
                    Image.fromarray((J_display * 255.0).astype(np.uint8), mode="RGB").save(
                        int_dir / f"J_step_{global_step}.png"
                    )
                    
                    # 3. Save ASM inversion: J_asm = (I - A*(1-t_safe)) / t_safe
                    I_np = I_target.detach().cpu().squeeze(0).numpy()
                    t_safe = torch.maximum(t, torch.full_like(t, 1e-3))
                    A_np = A.detach().cpu().reshape(t.shape[1], 1, 1).numpy() if A is not None else np.ones((t.shape[1], 1, 1)) * 0.8
                    
                    t_np_cpu = t_safe.detach().cpu().squeeze(0).numpy()
                    J_asm_np = (I_np - A_np * (1.0 - t_np_cpu)) / t_np_cpu
                    J_asm_np = np.clip(J_asm_np[:3].transpose(1, 2, 0), 0.0, 1.0)
                    
                    Image.fromarray((J_asm_np * 255.0).astype(np.uint8), mode="RGB").save(
                        int_dir / f"ASM_step_{global_step}.png"
                    )
            except Exception as e_save:
                print(f"  Warning: could not save intermediate outputs: {e_save}")

        # Progress bar update
        if pbar is not None:
            pct = int((global_step / max(total_steps - 1, 1)) * 100)
            pbar.set_postfix({
                "Scale": scale_name,
                "Loss":  f"{total_loss.item():.4f}",
                "ASM":   f"{loss_dict['l_asm'].item():.4f}",
                "Edge":  f"{loss_dict['l_edge'].item():.4f}",
                "Perc":  f"{loss_dict['l_perceptual'].item():.4f}",
            })
            pbar.update(1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def optimize_dip(
    cloudy_image_tensor: torch.Tensor,
    mask_tensor: "torch.Tensor | None" = None,
    gt_image_tensor: "torch.Tensor | None" = None,
    num_iters: int = 2000,
    lr: float = 0.005,
    use_gpu: bool = True,
    use_dps: bool = False,
    dps_weight: float = 0.5,
    callback=None,
    s2_loader=None,
    x_start: int | None = None,
    y_start: int | None = None,
    crop_size: int | None = None,
    status_check=None,
    # --- New Physics weights and variables ---
    lambda_asm: float = 0.0,
    lambda_rte: float = 0.1,
    lambda_ndvi: float = 0.05,
    lambda_tv: float = 1e-5,
    lambda_perceptual: float = 0.05,
    lambda_edge: float = 0.02,
    lambda_sam: float = 0.02,
    lambda_t_prior: float = 1.0,
    lambda_ssim: float = 0.1,
    alpha: float = 1.3,
    lambdas: list[float] | None = None,
    beta_abs: list[float] | None = None,
    dps_steps: int = 20,
    dps_eta: float = 0.0,
    dps_seed: int = 42,
    edge_weight: float = 10.0,
    enforce_unmasked: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 1: Multi-scale DIP + PINN optimization with perceptual + edge losses.
    Stage 2 (optional): Sequential DPS sampler to add terrain detail via diffusion prior.

    cloudy_image_tensor: [C, H, W] — Sentinel-2 reflectance in [0, 1].
    mask_tensor:         [H, W]   — cloud mask (1=cloud, 0=clear). Optional.
    num_iters:           total optimization steps (split across 3 scales).
    Returns (J, t) both as [C, H, W] CPU tensors.
    """
    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Running DIP Optimization on: {device}")

    # ── Prepare tensors ──────────────────────────────────────────────────────
    I_full  = cloudy_image_tensor.unsqueeze(0).to(device)            # [1, C, H, W]
    C, H, W = cloudy_image_tensor.shape

    if mask_tensor is not None:
        M_full = mask_tensor.to(device)
        while M_full.dim() < 4:
            M_full = M_full.unsqueeze(0)
    else:
        M_full = torch.zeros((1, 1, H, W), device=device)

    # ── Atmospheric light via DCP ────────────────────────────────────────────
    A_init = estimate_atmospheric_light_dcp(cloudy_image_tensor, patch_size=15, mask=mask_tensor).to(device)
    print(f"DCP estimated atmospheric light: {[round(v, 3) for v in A_init.squeeze().tolist()]}")

    # ── OpenCV TELEA inpaint as structural initialisation ────────────────────
    rgb_np   = (np.clip(cloudy_image_tensor[:3].permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
    pil_img  = Image.fromarray(rgb_np, mode="RGB")

    if mask_tensor is not None:
        mask_raw_np = (np.clip(mask_tensor.squeeze().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        try:
            import cv2
            c_mask = (mask_raw_np > 80).astype(np.uint8)  # Lower threshold to capture thin cloud fringes
            # Expanded morphological dilation to ensure no cloud boundary haze leaks through
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated_c_mask = cv2.dilate(c_mask, kernel, iterations=2)
            
            # Shadow candidate detection near cloud boundaries
            shadow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            dilated_shadow = cv2.dilate(dilated_c_mask, shadow_kernel, iterations=1)
            shadow_cand = (dilated_shadow == 1) & (dilated_c_mask == 0)
            gray_img = rgb_np.mean(axis=2) / 255.0
            shadow_mask = shadow_cand & (gray_img < 0.35)
            
            expanded_mask_np = np.clip((dilated_c_mask + shadow_mask.astype(np.uint8)) * 255, 0, 255).astype(np.uint8)
            pil_mask = Image.fromarray(expanded_mask_np, mode="L")
            mask_np = expanded_mask_np
            # Update M_full to use expanded mask for optimization so edges are thoroughly inpainted
            M_full = torch.from_numpy(expanded_mask_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        except Exception:
            pil_mask = Image.fromarray(mask_raw_np, mode="L")
            mask_np = mask_raw_np
    else:
        mask_np = np.zeros((H, W), dtype=np.uint8)
        pil_mask = Image.new("L", pil_img.size, 0)

    from ui.advanced_detection import remove_clouds_advanced
    inpainted_pil    = remove_clouds_advanced(pil_img, pil_mask)
    inpainted_tensor = transforms.ToTensor()(inpainted_pil).to(device)  # [3, H, W]

    if C == 4:
        # Inpaint NIR band channel as well so it does not contain bright cloud reflectance
        nir_np = (np.clip(cloudy_image_tensor[3].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        import cv2
        nir_inpainted_np = cv2.inpaint(nir_np, mask_np, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        nir_inpainted_tensor = torch.from_numpy(nir_inpainted_np.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        J_init_full = torch.cat([inpainted_tensor, nir_inpainted_tensor], dim=0)
    else:
        J_init_full = inpainted_tensor                                   # [C, H, W]

    # Pre-compute LBP map for the full image J_init_full
    lbp_full = extract_lbp(J_init_full.unsqueeze(0).to(device))
    
    with torch.no_grad():
        dummy_z = extract_all_features(J_init_full.unsqueeze(0).to(device), lbp_full)
        C_features = dummy_z.shape[1] + 32  # Added 32 for the injected noise channels

    # ── Model ────────────────────────────────────────────────────────────────
    model = ResidualAttentionUNet(in_channels=C_features, out_channels=C, init_A=A_init).to(device)
    print(f"Model: ResidualAttentionUNet  (in={C_features} ch, SE+Residual+SpatialAttn)")

    # ── Loss modules ─────────────────────────────────────────────────────────
    pinn_losses    = PINNLosses()
    perceptual_fn  = PerceptualLoss(device=str(device)).to(device)
    edge_fn        = SobelEdgeLoss().to(device)
    sam_fn         = SAMLoss().to(device)

    # ── Multi-scale schedule ─────────────────────────────────────────────────
    # 3 progressive scales: 25% at 64×64, 40% at 128×128, 35% at full (256×256)
    scales = [
        {"name": "64px",   "factor": 4, "frac": 0.25},
        {"name": "128px",  "factor": 2, "frac": 0.40},
        {"name": "256px",  "factor": 1, "frac": 0.35},
    ]
    has_nir = (C >= 4)

    # Compute per-scale iteration counts (ensure sum == num_iters exactly)
    if num_iters <= 5:
        scales = [{"name": "full", "factor": 1, "frac": 1.0, "iters": num_iters}]
    else:
        for s in scales[:-1]:
            s["iters"] = max(1, int(num_iters * s["frac"]))
        scales[-1]["iters"] = max(1, num_iters - sum(int(s["iters"]) for s in scales[:-1]))

    total_steps = sum(int(s["iters"]) for s in scales)
    callback_interval = 1

    # ── Optimizer + cosine LR scheduler ─────────────────────────────────────
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)

    # ── Progress bar ─────────────────────────────────────────────────────────
    pbar         = tqdm(total=total_steps, desc="Stage 1: DIP+PINN")
    global_step  = 0

    # ── Scale loop ────────────────────────────────────────────────────────────
    for scale in scales:
        f    = scale["factor"]
        name = scale["name"]

        if f == 1:
            I_s      = I_full
            M_s      = M_full
            J_init_s = J_init_full.unsqueeze(0).to(device)
            lbp_s    = lbp_full
            z_s      = extract_all_features(J_init_s, lbp_s)
            gt_s     = gt_image_tensor.unsqueeze(0).to(device) if gt_image_tensor is not None else None
        else:
            # Downsample with average pooling for images, max-pool for mask to ensure perfect spatial alignment
            I_s = F.avg_pool2d(I_full, kernel_size=int(f), stride=int(f))
            M_s = (F.max_pool2d(M_full, kernel_size=int(f), stride=int(f)) > 0.5).float()
            J_init_s = F.avg_pool2d(J_init_full.unsqueeze(0).to(device), kernel_size=int(f), stride=int(f))
            lbp_s    = F.avg_pool2d(lbp_full, kernel_size=int(f), stride=int(f))
            z_s      = extract_all_features(J_init_s, lbp_s)
            gt_s     = F.avg_pool2d(gt_image_tensor.unsqueeze(0).to(device), kernel_size=int(f), stride=int(f)) if gt_image_tensor is not None else None

        # Inject fixed spatial noise for true Deep Image Prior hallucination capacity
        torch.manual_seed(42)  # Fixed seed for consistent latent
        z_noise = torch.randn(z_s.shape[0], 32, z_s.shape[2], z_s.shape[3], device=device) * 0.1
        z_s = torch.cat([z_s, z_noise], dim=1)

        _run_scale(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            z=z_s,
            I_target=I_s,
            M_target=M_s,
            J_init_target=J_init_s,
            gt_target=gt_s,
            num_iters=int(scale["iters"]),
            pinn_losses=pinn_losses,
            perceptual_fn=perceptual_fn,
            edge_fn=edge_fn,
            sam_fn=sam_fn,
            has_nir=has_nir,
            scale_name=str(name),
            callback=callback,
            callback_interval=callback_interval,
            global_step_offset=global_step,
            total_steps=total_steps,
            pbar=pbar,
            status_check=status_check,
            # --- New Physics weights and variables ---
            lambda_asm=lambda_asm,
            lambda_rte=lambda_rte,
            lambda_ndvi=lambda_ndvi,
            lambda_tv=lambda_tv,
            lambda_perceptual=lambda_perceptual,
            lambda_edge=lambda_edge,
            lambda_sam=lambda_sam,
            lambda_t_prior=lambda_t_prior,
            lambda_ssim=lambda_ssim,
            alpha=alpha,
            lambdas=lambdas,
            beta_abs=beta_abs,
            edge_weight=edge_weight,
            enforce_unmasked=enforce_unmasked,
        )
        global_step += int(scale["iters"])

    pbar.close()

    # ── Final inference at full resolution ───────────────────────────────────
    model.eval()
    with torch.no_grad():
        z_full = extract_all_features(J_init_full.unsqueeze(0).to(device), lbp_full)
        torch.manual_seed(42)
        z_noise_full = torch.randn(z_full.shape[0], 32, z_full.shape[2], z_full.shape[3], device=device) * 0.1
        z_full = torch.cat([z_full, z_noise_full], dim=1)
        final_J, final_t, final_A = model(z_full)

    J_dip    = final_J.squeeze(0)   # [C, H, W]  — DIP scene estimate
    t_stage1 = final_t.squeeze(0)   # [C, H, W]  — transmission map
    A_final  = final_A.squeeze()    # [C] or scalar

    # ── Diagnostic: print final A and t stats ───────────────────────────────
    print(f"[optimize_dip] Stage 1: DIP optimization completed.")
    print(f"  A_final  : {A_final.squeeze().tolist()}")
    print(f"  t_stage1 : min={t_stage1.min().item():.3f}  max={t_stage1.max().item():.3f}  mean={t_stage1.mean().item():.3f}")
    print(f"  J_dip    : min={J_dip.min().item():.3f}  max={J_dip.max().item():.3f}  mean={J_dip.mean().item():.3f}")

    I_np   = I_full.squeeze(0)                                     # [C, H, W]
    t_np   = t_stage1                                              # [C, H, W]
    A_np   = A_final.reshape(C, 1, 1).cpu() if A_final.dim() >= 1 else A_final.cpu()
    M_np   = M_full.squeeze(0).squeeze(0).cpu()                   # [H, W]

    eps = 0.05
    t_safe    = torch.maximum(t_np.cpu(), torch.full_like(t_np.cpu(), eps))
    J_asm     = (I_np.cpu() - A_np * (1.0 - t_safe)) / t_safe
    J_asm     = torch.clamp(J_asm, 0.0, 1.0)

    # Fast marching inpaint tensor
    J_inpaint_np = np.array(inpainted_pil.convert("RGB" if C==3 else "RGBA"), dtype=np.float32) / 255.0
    if C == 4:
        J_inpaint_tensor = torch.cat([torch.from_numpy(J_inpaint_np[:, :, :3].transpose(2, 0, 1)), I_np.cpu()[3:4]], dim=0)
    else:
        J_inpaint_tensor = torch.from_numpy(J_inpaint_np.transpose(2, 0, 1))

    # ── Improved Cloud Region Reconstruction ────────────────────────────────
    # When inpainting cloudy regions (M > 0), rely on DIP's high-frequency synthesis
    # rather than hazy ASM inversion which leaves whitish patches.
    # We blend DIP output (90%) with subtle Telea structural guidance (10%) inside the cloud.
    J_cloud = 0.90 * J_dip.cpu() + 0.10 * J_inpaint_tensor.cpu()
    J_cloud = torch.clamp(J_cloud, 0.0, 1.0)

    # Smooth boundary transition mask to avoid hard seams
    try:
        import cv2
        m_blur = cv2.GaussianBlur(M_np.numpy(), (9, 9), 3.0)
        M_smooth = torch.from_numpy(m_blur).unsqueeze(0).expand(C, -1, -1)
    except Exception:
        M_smooth = M_np.unsqueeze(0).expand(C, -1, -1)

    # Blend: restored inside cloud, exact original input image on clear land (M=0):
    J_stage1  = M_smooth * J_cloud + (1.0 - M_smooth) * I_np.cpu()
    J_stage1  = torch.clamp(J_stage1, 0.0, 1.0)

    print(f"  J_stage1 (Restored): min={J_stage1.min().item():.3f}  max={J_stage1.max().item():.3f}  mean={J_stage1.mean().item():.3f}")

    J_final = J_stage1.clone()

    # ── Stage 2: DPS Sampler ─────────────────────────────────────────────────
    if use_dps:
        try:
            print(f"[optimize_dip] Stage 2: DPS enabled. Steps: {dps_steps}, Weight: {dps_weight}, Eta: {dps_eta}, Seed: {dps_seed}")
            dps_prior = DPSPrior(device=str(device))
            J_rgb     = J_stage1[:3].unsqueeze(0).to(device)  # [1, 3, H, W]
            M_single  = M_full[:, :1]                          # [1, 1, H, W]

            torch.manual_seed(dps_seed)
            J_rgb_dps = dps_prior.sample_dps(
                J_rgb, M_single, 
                num_steps=3 if device.type == "cpu" else dps_steps, 
                zeta=dps_weight, 
                eta=dps_eta
            )

            J_final      = J_stage1.clone().to(device)
            J_final[:3]  = J_rgb_dps.squeeze(0)
            
            delta = torch.mean(torch.abs(J_final[:3] - J_stage1[:3]))
            print(f"[optimize_dip] Stage 2 completed. Mean delta: {delta.item():.6f}")
        except Exception as e:
            print(f"Warning: Stage 2 DPS failed ({e}). Returning Stage 1 output.")

    # ── Save Debug PNGs at the very end ──────────────────────────────────────
    try:
        import os
        from pathlib import Path
        debug_dir = Path(__file__).resolve().parent / "outputs" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # J (scene radiance estimate)
        J_rgb = J_final[:3].permute(1, 2, 0).cpu().numpy()
        J_pil = Image.fromarray((np.clip(J_rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
        J_pil.save(str(debug_dir / "J.png"))

        # Raw DIP output (before physical ASM inversion)
        J_dip_rgb = J_dip[:3].permute(1, 2, 0).cpu().numpy()
        J_dip_pil = Image.fromarray((np.clip(J_dip_rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
        J_dip_pil.save(str(debug_dir / "J_dip.png"))

        # Pure ASM output (before CLAHE and post-enhancements)
        J_asm_rgb = J_asm[:3].permute(1, 2, 0).cpu().numpy()
        J_asm_pil = Image.fromarray((np.clip(J_asm_rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
        J_asm_pil.save(str(debug_dir / "J_asm.png"))

        # Transmission map (mean across channels → grayscale)
        t_gray = t_stage1.mean(dim=0).cpu().numpy()
        t_pil  = Image.fromarray((np.clip(t_gray, 0, 1) * 255).astype(np.uint8), mode="L")
        t_pil.save(str(debug_dir / "transmission.png"))

        # Compute physical model reconstruction confidence map based on standard ASM residual
        try:
            I_full_c = I_full.squeeze(0)[:3].cpu()
            t_c = t_stage1[:3].cpu()
            J_dip_c = J_dip[:3].cpu()
            A_sliced = A_final[:3] if A_final.dim() >= 1 else A_final
            A_c = A_sliced.reshape(-1, 1, 1).cpu() if A_sliced.dim() >= 1 else A_sliced.cpu()
            reconstructed_I = J_dip_c * t_c + A_c * (1.0 - t_c)
            residual = torch.abs(I_full_c - reconstructed_I).mean(dim=0)
            confidence_map = 1.0 - torch.clamp(residual / 0.20, 0.0, 1.0)
            confidence_np = confidence_map.numpy()
            confidence_pil = Image.fromarray((confidence_np * 255.0).astype(np.uint8), mode="L")
            confidence_pil.save(str(debug_dir / "transmission_confidence.png"))
        except Exception as e_conf:
            print(f"  Warning: could not save confidence map: {e_conf}")

        # Final reconstruction (same as J for RGB display)
        J_pil.save(str(debug_dir / "final_output.png"))

        # M mask for reference
        if mask_tensor is not None:
            mask_np = mask_tensor.squeeze().cpu().numpy()
            mask_pil = Image.fromarray((np.clip(mask_np, 0, 1) * 255).astype(np.uint8), mode="L")
            mask_pil.save(str(debug_dir / "mask.png"))
            mask_pil.save(str(debug_dir / "cloud_mask.png"))

        print(f"  Debug PNGs saved to: {debug_dir}")
        print(f"    J.png  transmission.png  final_output.png  cloud_mask.png")
    except Exception as _e:
        print(f"  Warning: could not save debug PNGs: {_e}")

    return J_final.cpu(), t_stage1.cpu()
