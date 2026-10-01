# rice2dip_pinn_loss.py
# Add your RICE2 PINN Loss functions code here
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Perceptual Loss (VGG16 feature matching)
# ---------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 intermediate feature maps.
    Compares relu1_2, relu2_2, relu3_3 activations between predicted
    and target images to enforce texture and structural realism.

    Operates on RGB only. Input is [B, C, H, W] in [0, 1].
    If C == 4 (multispectral), only the first 3 channels (RGB) are used.
    VGG weights are frozen and loaded from torchvision once.
    """
    # Class-level cache so VGG is loaded only once per process
    _vgg_features: Optional[nn.Module] = None

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.vgg: Optional[nn.Module] = None
        self._build_vgg()

        # ImageNet normalization for VGG (mean/std of RGB in [0,1])
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def _build_vgg(self):
        """Build VGG16 up to relu3_3 (index 16). Freeze weights."""
        if PerceptualLoss._vgg_features is not None:
            self.vgg = PerceptualLoss._vgg_features
            return
        try:
            import torchvision.models as tvm
            vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
            # Extract features up to relu3_3 (layer index 16 in vgg16.features)
            self.vgg = nn.Sequential(*list(vgg.features.children())[:17]).eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            PerceptualLoss._vgg_features = self.vgg
        except Exception as e:
            print(f"Warning: VGG16 perceptual loss unavailable ({e}). Using L1 fallback.")
            self.vgg = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, C, H, W] in [0, 1].
        Returns scalar perceptual loss.
        """
        if self.vgg is None:
            # Fallback: simple L1 if VGG not available
            return F.l1_loss(pred[:, :3], target[:, :3])

        # Extract RGB channels only
        p = pred[:, :3].to(self.device)
        t = target[:, :3].to(self.device)

        # Ensure minimum spatial size for VGG (needs >= 32px per side)
        if p.shape[-1] < 32 or p.shape[-2] < 32:
            p = F.interpolate(p, size=(32, 32), mode='bilinear', align_corners=False)
            t = F.interpolate(t, size=(32, 32), mode='bilinear', align_corners=False)

        # Normalize for ImageNet
        p = (p - self.mean) / self.std
        t = (t - self.mean) / self.std

        self.vgg = self.vgg.to(self.device)
        with torch.no_grad():
            feat_t = self.vgg(t)
        feat_p = self.vgg(p)

        return F.mse_loss(feat_p, feat_t.detach())


# ---------------------------------------------------------------------------
# Sobel Edge Loss
# ---------------------------------------------------------------------------
class SobelEdgeLoss(nn.Module):
    """
    Sobel Edge Loss using fixed 3×3 Sobel kernels.
    Computes edge magnitude maps of predicted and target images,
    then applies L1 loss between them.

    Preserves terrain ridges, mountain edges, roads, and river boundaries
    that MSE loss alone cannot capture.

    Input: [B, C, H, W] in [0, 1]. Uses first 3 channels (RGB).
    """
    def __init__(self):
        super().__init__()
        # Fixed Sobel filters — never updated by optimizer
        kx = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]], requires_grad=False
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]], requires_grad=False
        ).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _sobel(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B, C, H, W]. Returns edge magnitude [B, C, H, W]."""
        B, C, H, W = img.shape
        # Process each channel independently
        img_flat = img.reshape(B * C, 1, H, W)
        gx = F.conv2d(img_flat, self.kx, padding=1)
        gy = F.conv2d(img_flat, self.ky, padding=1)
        edges = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return edges.reshape(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, C, H, W] in [0, 1].
        Returns scalar edge loss.
        """
        # Use only RGB (first 3 ch) for edge matching
        p = pred[:, :3]
        t = target[:, :3]
        edges_p = self._sobel(p)
        edges_t = self._sobel(t.detach())
        return F.l1_loss(edges_p, edges_t)


# ---------------------------------------------------------------------------
# SAM Loss (Spectral Angle Mapper)
# ---------------------------------------------------------------------------
class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) loss.
    Minimises the angular distance between predicted J and reference I spectra
    on clear pixels, preserving relative band ratios (vegetation, water, bare soil).

    Input: [B, C, H, W] in [0, 1].
    M:     [B, 1, H, W] cloud mask (1=cloud, 0=clear).
    Returns a scalar in [0, π/2].
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        M: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        pred, target: [B, C, H, W] in [0, 1].
        M:            [B, 1, H, W] or None (if None, uses all pixels).
        Returns scalar SAM loss.
        """
        # Dot product across channel dimension
        dot   = (pred * target).sum(dim=1, keepdim=True)                 # [B, 1, H, W]
        # Safe norm to avoid NaN gradients at 0
        norm_p = torch.sqrt(torch.sum(pred**2, dim=1, keepdim=True) + self.eps)
        norm_t = torch.sqrt(torch.sum(target**2, dim=1, keepdim=True) + self.eps)
        # Use 0.9999 for clamp to prevent float32 rounding to exactly 1.0, which causes acos gradient NaN
        cos_sim = torch.clamp(dot / (norm_p * norm_t), -0.9999, 0.9999)
        angles  = torch.acos(cos_sim)                                     # [B, 1, H, W]

        if M is not None:
            clear_mask = 1.0 - M                                          # 1 on clear pixels
            loss = (clear_mask * angles).sum() / (clear_mask.sum() + self.eps)
        else:
            loss = angles.mean()

        return loss


# ---------------------------------------------------------------------------
# Charbonnier Loss
# ---------------------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps))


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------
def ssim_loss_fn(img1, img2, window_size=11, size_average=True):
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1*img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2*img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1*img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return 1.0 - ssim_map.mean()
    else:
        return 1.0 - ssim_map


# ---------------------------------------------------------------------------
# PINN Losses (Physics-Informed)
# ---------------------------------------------------------------------------
class PINNLosses:
    """
    Physics-Informed Neural Network (PINN) losses for cloud removal.

    Implements:
    - ASM  : Atmospheric Scattering Model consistency
    - RTE  : Radiative Transfer Equation (Rayleigh + Mie + absorption)
    - NDVI : Vegetation index consistency on clear pixels
    - TV   : Total Variation smoothness on transmission map
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.charbonnier = CharbonnierLoss()

    def asm_loss(self, I: torch.Tensor, J: torch.Tensor, t: torch.Tensor, A: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
        """
        Atmospheric Scattering Model (ASM) loss.
        Forward model: I_hat = J * t + A * (1 - t)
        Globally enforced physical consistency to allow gradients to flow in cloud regions.
        Normalized by image size instead of mask size.
        """
        I_hat = J * t + A * (1.0 - t)
        return F.mse_loss(I_hat, I, reduction="mean")

    def masked_recon_loss(self, I: torch.Tensor, J: torch.Tensor, t: torch.Tensor, A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Masked Reconstruction Loss. Only clear pixels (M=0) contribute.
        M: Cloud mask (1 = cloud, 0 = clear). (1 - M) selects clear pixels.
        """
        I_hat = J * t + A * (1 - t)
        clear_mask = 1.0 - M
        loss = clear_mask * (I - I_hat) ** 2
        return loss.sum() / (clear_mask.sum() + self.eps)

    def rte_loss(self, t: torch.Tensor, alpha: float = 1.3, lambdas: list[float] | None = None, beta_abs: list[float] | None = None) -> torch.Tensor:
        """
        Full Radiative Transfer Equation (RTE) physical consistency.
        Models Rayleigh scattering (λ^-4), Mie scattering (λ^-alpha),
        and standard atmospheric gas absorption.
        Dynamically fits scene depth d(x) and aerosol density c_M(x)
        via pixel-wise least squares on optical depth tau = -ln(t).
        """
        channels = t.shape[1]

        # Use custom parameters if provided, else fall back to defaults
        if lambdas is not None and beta_abs is not None:
            pass
        elif channels == 4:
            # Blue (B02), Green (B03), Red (B04), NIR (B08)
            lambdas  = [0.4927, 0.5598, 0.6646, 0.8328]
            beta_abs = [0.002,  0.015,  0.010,  0.005]
        elif channels == 3:
            lambdas  = [0.4927, 0.5598, 0.6646]
            beta_abs = [0.002,  0.015,  0.010]
        else:
            lambdas  = [0.4927 + i * 0.1 for i in range(channels)]
            beta_abs = [0.01] * channels

        # Rayleigh scattering coefficient: β_R = 0.008569 · λ^(-4)
        beta_R     = [0.008569 * (l ** -4) for l in lambdas]
        beta_M_coef = [l ** -alpha for l in lambdas]
        k_coef     = [r + a for r, a in zip(beta_R, beta_abs)]

        device = t.device
        dtype  = t.dtype

        # Force float32 for numerical stability in linear algebra solver
        A_mat = torch.tensor(
            [[k, m] for k, m in zip(k_coef, beta_M_coef)],
            dtype=torch.float32, device=device
        )

        try:
            A_pinv = torch.linalg.pinv(A_mat)   # [2, C]
        except AttributeError:
            A_pinv = torch.pinverse(A_mat)

        # Optical depth: τ = -ln(t)
        tau = -torch.log(t.to(torch.float32) + self.eps)           # [B, C, H, W]

        B, C, H, W = tau.shape
        tau_flat = tau.transpose(0, 1).reshape(C, -1)

        # Least-squares solve for physical depth + aerosol parameters
        uv = torch.matmul(A_pinv, tau_flat)            # [2, B*H*W]
        uv = torch.clamp(uv, min=0.0)                  # non-negative physical params

        tau_hat_flat = torch.matmul(A_mat, uv)         # [C, B*H*W]
        tau_hat = tau_hat_flat.reshape(C, B, H, W).transpose(0, 1)

        loss = F.mse_loss(tau, tau_hat)
        return loss.to(dtype)

    def ndvi_consistency_loss(
        self, J: torch.Tensor, I_ref: torch.Tensor, M: torch.Tensor,
        gamma: float = 0.05, nir_idx: int = 3, red_idx: int = 2
    ) -> torch.Tensor:
        """
        NDVI Consistency Loss.
        - Clear pixels (1-M): match predicted NDVI to observed NDVI.
        - Cloudy pixels (M): light smoothness only (gamma=0.05, was 0.1).
        """
        def calc_ndvi(img):
            nir = img[:, nir_idx:nir_idx + 1, :, :]
            red = img[:, red_idx:red_idx + 1, :, :]
            return (nir - red) / (nir + red + self.eps)

        ndvi_J     = calc_ndvi(J)
        ndvi_I_ref = calc_ndvi(I_ref)
        clear_mask = 1.0 - M

        # Match NDVI on clear pixels
        data_loss = (clear_mask * (ndvi_J - ndvi_I_ref) ** 2).sum() / (clear_mask.sum() + self.eps)

        # Light smoothness on cloudy pixels (gamma reduced from 0.1 → 0.05)
        dy = ndvi_J[:, :, 1:, :] - ndvi_J[:, :, :-1, :]
        dx = ndvi_J[:, :, :, 1:] - ndvi_J[:, :, :, :-1]
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        grad_mag_sq = dx ** 2 + dy ** 2
        smooth_loss = (M * grad_mag_sq).sum() / (M.sum() + self.eps)

        return data_loss + gamma * smooth_loss

    def tv_loss(self, t: torch.Tensor, I: torch.Tensor | None = None, edge_weight: float = 10.0) -> torch.Tensor:
        """
        Isotropic Edge-Aware Total Variation (TV) smoothness on the transmission map t.
        Weights smoothness penalty using gradients of the original image I.
        """
        dy_t = t[:, :, 1:, :] - t[:, :, :-1, :]
        dx_t = t[:, :, :, 1:] - t[:, :, :, :-1]
        dy_t = F.pad(dy_t, (0, 0, 0, 1), mode='replicate')
        dx_t = F.pad(dx_t, (0, 1, 0, 0), mode='replicate')
        grad_t = torch.sqrt(dx_t ** 2 + dy_t ** 2 + self.eps)
        
        if I is not None:
            dy_I = I[:, :, 1:, :] - I[:, :, :-1, :]
            dx_I = I[:, :, :, 1:] - I[:, :, :, :-1]
            dy_I = F.pad(dy_I, (0, 0, 0, 1), mode='replicate')
            dx_I = F.pad(dx_I, (0, 1, 0, 0), mode='replicate')
            grad_I = torch.mean(torch.sqrt(dx_I ** 2 + dy_I ** 2 + self.eps), dim=1, keepdim=True)
            weight = torch.exp(-edge_weight * grad_I)
            return torch.mean(grad_t * weight)
        return torch.mean(grad_t)

    def combined_pinn_loss(
        self,
        I: torch.Tensor,
        J: torch.Tensor,
        t: torch.Tensor,
        A: torch.Tensor,
        M: torch.Tensor,
        GT: torch.Tensor | None = None,
        I_ref: torch.Tensor | None = None,
        has_nir: bool = False,
        # ── Physics weights ────────────────────────────────────────────────
        lambda_asm:        float = 1.0,
        lambda_rte:        float = 0.1,
        lambda_ndvi:       float = 0.05,
        lambda_tv:         float = 1e-5,   # lower — stops TV over-smoothing
        # ── Structure-preserving weights ──────────────────────────────────
        lambda_perceptual: float = 0.0,    # caller passes 0.05
        lambda_edge:       float = 0.0,    # caller passes 0.02
        lambda_sam:        float = 0.0,    # caller passes 0.02
        lambda_t_prior:    float = 0.5,    # penalizes high transmission in clouds
        lambda_ssim:       float = 0.1,    # structural similarity loss
        # ── Misc ──────────────────────────────────────────────────────────
        alpha:   float = 1.3,
        nir_idx: int   = 3,
        red_idx: int   = 2,
        perceptual_loss_fn = None,
        edge_loss_fn       = None,
        sam_loss_fn        = None,
        lambdas: list[float] | None = None,
        beta_abs: list[float] | None = None,
        global_step: int = 0,
        edge_weight: float = 10.0,
        J_init: torch.Tensor | None = None,
    ) -> dict:
        """
        Combined PINN Loss.
        Returns a dict of all scalar loss components plus total_loss.
        """
        # Clamp transmission before log/prior computations
        t = torch.clamp(t, 1e-3, 0.999)

        # Reconstruction Loss using Charbonnier
        if GT is not None:
            # Masked supervision: target GT in cloud region, Input in clear region
            diff_gt = torch.sqrt((J - GT) ** 2 + 1e-6)
            diff_in = torch.sqrt((J - I) ** 2 + 1e-6)
            l_recon = torch.mean(M * diff_gt + (1.0 - M) * 0.1 * diff_in)
        else:
            # Unsupervised: target Input on clear region only
            diff_in = torch.sqrt((J - I) ** 2 + 1e-6)
            clear_mask = 1.0 - M
            l_recon = (clear_mask * diff_in).sum() / (clear_mask.sum() + self.eps)

        # Color Constancy Loss on clear pixels to preserve original ground chromaticity
        l_color = torch.tensor(0.0, device=I.device)
        clear_mask = 1.0 - M
        clear_sum = clear_mask.sum() + self.eps
        J_mean = (J * clear_mask).sum(dim=[2,3], keepdim=True) / clear_sum
        I_mean = (I * clear_mask).sum(dim=[2,3], keepdim=True) / clear_sum
        J_norm = J / (J_mean + 1e-8)
        I_norm = I / (I_mean + 1e-8)
        l_color = (clear_mask * torch.abs(J_norm - I_norm)).sum() / clear_sum

        # Adaptive ramp + decay for physics losses:
        # Ramp up over first 400 steps, decay after step 1000 (was 600) so physics constraints stay longer for thick clouds
        if global_step < 400:
            physics_weight_scale = global_step / 400.0
        elif global_step > 1000:
            physics_weight_scale = max(0.15, 1.0 - (global_step - 1000) / 500.0)  # Slower decay
        else:
            physics_weight_scale = 1.0

        effective_lambda_asm = lambda_asm * physics_weight_scale
        effective_lambda_rte = lambda_rte * physics_weight_scale

        if effective_lambda_asm > 0:
            I_hat = J * t + A * (1.0 - t)
            diff_asm = (I_hat - I) ** 2
            # Weight ASM loss by transmission t so we don't force J to match opaque white clouds
            l_asm = (t * diff_asm).sum() / (t.sum() + self.eps)
        else:
            l_asm = torch.tensor(0.0, device=I.device)

        l_rte   = self.rte_loss(t * M + (1.0 - M) * 0.95, alpha=alpha, lambdas=lambdas, beta_abs=beta_abs) if effective_lambda_rte > 0 else torch.tensor(0.0, device=I.device)
        l_tv    = self.tv_loss(t, I=I, edge_weight=edge_weight)

        l_ndvi = torch.tensor(0.0, device=I.device)
        if has_nir and I_ref is not None:
            l_ndvi = self.ndvi_consistency_loss(
                J, I_ref, M, nir_idx=nir_idx, red_idx=red_idx
            )

        # Perceptual loss (VGG16 feature matching)
        l_perceptual = torch.tensor(0.0, device=I.device)
        if lambda_perceptual > 0 and perceptual_loss_fn is not None:
            if GT is not None:
                l_perceptual = perceptual_loss_fn(J, GT)
            else:
                l_perceptual = perceptual_loss_fn(J * clear_mask, I * clear_mask)

        # Sobel edge loss
        l_edge = torch.tensor(0.0, device=I.device)
        if lambda_edge > 0 and edge_loss_fn is not None:
            if GT is not None:
                l_edge = edge_loss_fn(J, GT)
            else:
                l_edge = edge_loss_fn(J * clear_mask, I * clear_mask)

        # SAM loss — spectral angle mapping
        l_sam = torch.tensor(0.0, device=I.device)
        if lambda_sam > 0 and sam_loss_fn is not None:
            if GT is not None:
                l_sam = sam_loss_fn(J, GT, None)
            else:
                l_sam = sam_loss_fn(J, I, M)

        # Transmission prior — delayed to match ASM warmup (active after step 50)
        l_t_prior = torch.tensor(0.0, device=I.device)
        if lambda_t_prior > 0 and global_step >= 50:
            # Soft prior — only penalise extreme values to complement explicit t-supervision
            # Fix: Adjust thresholds for thick clouds - penalise t > 0.3 in clouds (was 0.5)
            l_t_cloud = (M * torch.clamp(t - 0.3, min=0.0) ** 2).mean()  # Only penalise t > 0.3 in clouds
            l_t_clear = ((1.0 - M) * torch.clamp(0.7 - t, min=0.0) ** 2).mean()  # Only penalise t < 0.7 in clear
            l_t_prior = l_t_cloud + l_t_clear

        # Differentiable SSIM loss
        l_ssim = torch.tensor(0.0, device=I.device)
        if lambda_ssim > 0:
            if GT is not None:
                l_ssim = ssim_loss_fn(J, GT)
            else:
                l_ssim = (clear_mask * ssim_loss_fn(J, I, size_average=False)).sum() / (clear_mask.sum() + self.eps)

        recon_weight = 2.0 if GT is None else 1.0
        total_loss = (
            recon_weight * l_recon
            + effective_lambda_asm * l_asm   # ramped then decayed
            + effective_lambda_rte * l_rte   # ramped then decayed
            + lambda_tv         * l_tv
            + lambda_ndvi       * l_ndvi
            + lambda_perceptual * l_perceptual
            + lambda_edge       * l_edge
            + lambda_sam        * l_sam
            + lambda_t_prior    * l_t_prior
            + lambda_ssim       * l_ssim
            + 0.05              * l_color    # Enforce clear ground color constancy
        )

        return {
            "total_loss":   total_loss,
            "l_recon":      l_recon,
            "l_asm":        l_asm,
            "l_rte":        l_rte,
            "l_ndvi":       l_ndvi,
            "l_tv":         l_tv,
            "l_perceptual": l_perceptual,
            "l_edge":       l_edge,
            "l_sam":        l_sam,
            "l_t_prior":    l_t_prior,
            "l_ssim":       l_ssim,
        }


# ---------------------------------------------------------------------------
# DPS Prior (Stage 2 — Diffusion Posterior Sampling)
# ---------------------------------------------------------------------------
class DPSPrior:
    """
    Diffusion Posterior Sampling (DPS) Prior.
    Acts as a sequential second-stage sampler to hallucinate detail in
    heavily cloud-obscured areas. Conditions on the Stage 1 DIP output J.

    Uses google/ddpm-church-256 (256×256, 3-channel landscape model).
    Memory-efficient: uses gradient detachment and normalized step sizes.
    """
    def __init__(self, model_id: str = "google/ddpm-church-256", device: str = "cuda"):
        try:
            from diffusers import DDPMPipeline
        except ImportError:
            raise ImportError(
                "Please install diffusers: pip install diffusers transformers accelerate"
            )

        print(f"Loading pretrained diffusion prior: {model_id}…")
        self.device = device
        pipeline = DDPMPipeline.from_pretrained(model_id)
        assert pipeline is not None, f"Failed to load pipeline from {model_id}"
        self.pipeline  = pipeline.to(device)
        self.unet      = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        self.unet.eval()

        for param in self.unet.parameters():
            param.requires_grad = False

        self.eps = 1e-8

    def compute_loss(self, J: torch.Tensor) -> torch.Tensor:
        """Legacy SDS/score-matching loss. Kept for backwards compatibility."""
        J_resized = J
        if J.shape[-1] != 256 or J.shape[-2] != 256:
            J_resized = F.interpolate(J, size=(256, 256), mode='bilinear', align_corners=False)

        batch_size = J_resized.shape[0]
        t   = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
        eps = torch.randn_like(J_resized)
        J_t = self.scheduler.add_noise(J_resized, eps, t)

        with torch.no_grad():
            eps_theta = self.unet(J_t, t).sample

        return F.mse_loss(eps_theta.detach(), eps)

    def sample_dps(
        self, J_dip: torch.Tensor, M: torch.Tensor,
        num_steps: int = 20, zeta: float = 0.5, eta: float = 0.0
    ) -> torch.Tensor:
        """
        Sequential Diffusion Posterior Sampling (DPS) conditioned on J_dip.
        Tweedie formula for clean estimate + posterior gradient steps.

        J_dip: [1, 3, H, W] in [0, 1]
        M:     [1, 1, H, W]  cloud mask (1=cloud)
        Returns [1, 3, H, W] in [0, 1]
        """
        device   = self.device
        orig_h, orig_w = J_dip.shape[-2], J_dip.shape[-1]

        # Resize to 256×256 for the diffusion model
        J_dip_r = J_dip if (orig_w == 256 and orig_h == 256) else \
                  F.interpolate(J_dip, size=(256, 256), mode='bilinear', align_corners=False)
        M_r     = M if (M.shape[-1] == 256 and M.shape[-2] == 256) else \
                  F.interpolate(M, size=(256, 256), mode='bilinear', align_corners=False)

        # Measurement: clear-pixel observations in [-1, 1]
        y = (1.0 - M_r) * (J_dip_r * 2.0 - 1.0)

        self.scheduler.set_timesteps(num_steps)
        x_t = torch.randn((1, 3, 256, 256), device=device)

        import inspect
        step_params = inspect.signature(self.scheduler.step).parameters

        for t in self.scheduler.timesteps:
            t_tensor = torch.tensor([t], device=device).long()
            x_t      = x_t.detach().requires_grad_(True)

            # Predict noise
            noise_pred = self.unet(x_t, t_tensor).sample

            # Tweedie clean estimate x̂₀
            alpha_bar = self.scheduler.alphas_cumprod[t]
            x0_hat    = (x_t - torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

            # Measurement consistency loss on clear pixels
            loss = torch.sum(((1.0 - M_r) * (x0_hat - y)) ** 2)
            grad = torch.autograd.grad(loss, x_t)[0]

            # Normalised posterior step size
            grad_norm = torch.linalg.norm(grad) + self.eps
            zeta_t    = zeta / grad_norm

            with torch.no_grad():
                # Dynamically pass eta if supported by the scheduler step method
                if "eta" in step_params:
                    step_out = self.scheduler.step(noise_pred, t, x_t, eta=eta)
                else:
                    step_out = self.scheduler.step(noise_pred, t, x_t)
                x_t = step_out.prev_sample - zeta_t * grad

        with torch.no_grad():
            x_t     = torch.clamp(x_t, -1.0, 1.0)
            J_final = (x_t + 1.0) / 2.0
            
            # Apply Gaussian blurred mask compositing (confidence blending)
            # to prevent seams between clear and cloudy regions
            import torchvision.transforms.functional as TVF
            M_blurred = TVF.gaussian_blur(M_r, kernel_size=[7, 7], sigma=[2.0, 2.0])
            J_final = (1.0 - M_blurred) * J_dip_r + M_blurred * J_final

            if orig_w != 256 or orig_h != 256:
                J_final = F.interpolate(J_final, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return J_final
