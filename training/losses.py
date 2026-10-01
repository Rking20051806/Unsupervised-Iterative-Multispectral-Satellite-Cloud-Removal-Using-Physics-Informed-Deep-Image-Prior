import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNLosses:
    """
    Physics-Informed Neural Network (PINN) losses for cloud removal.
    Implements ASM, RTE, NDVI, Masked Recon, and TV losses.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def asm_loss(self, I: torch.Tensor, J: torch.Tensor, t: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Atmospheric Scattering Model (ASM) loss.
        I_b = J_b * t_b + A_b * (1 - t_b)
        """
        I_hat = J * t + A * (1 - t)
        return F.mse_loss(I_hat, I)

    def masked_recon_loss(self, I: torch.Tensor, J: torch.Tensor, t: torch.Tensor, A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Masked Reconstruction Loss. Only clear pixels contribute.
        M: Cloud mask (1 = cloud, 0 = clear). (1 - M) selects clear pixels.
        """
        I_hat = J * t + A * (1 - t)
        clear_mask = 1.0 - M
        # Ensure mask broadcasts if necessary
        loss = clear_mask * (I - I_hat) ** 2
        return loss.sum() / (clear_mask.sum() + self.eps)

    def rte_loss(self, t: torch.Tensor, ref_band_idx: int = 0, alpha: float = 1.3, lambdas: list[float] = None) -> torch.Tensor:
        """
        Radiative Transfer Equation (RTE) residual.
        tau_b = -ln(t_b + eps)
        tau_b = tau_ref * (lambda_ref / lambda_b)^alpha
        """
        if lambdas is None:
            # Default to some standard visible wavelengths if not provided (e.g., RGB)
            lambdas = [450.0, 550.0, 650.0]  # B, G, R approximate wavelengths
        
        tau = -torch.log(t + self.eps)
        tau_ref = tau[:, ref_band_idx:ref_band_idx+1, :, :]
        lambda_ref = lambdas[ref_band_idx]
        
        loss = 0.0
        channels = t.shape[1]
        for b in range(channels):
            if b == ref_band_idx:
                continue
            ratio = (lambda_ref / lambdas[b]) ** alpha
            expected_tau_b = tau_ref * ratio
            loss += F.mse_loss(tau[:, b:b+1, :, :], expected_tau_b)
            
        return loss / max(1, channels - 1)

    def ndvi_consistency_loss(self, J: torch.Tensor, I_ref: torch.Tensor, M: torch.Tensor, gamma: float = 0.1, nir_idx: int = 3, red_idx: int = 0) -> torch.Tensor:
        """
        NDVI Consistency Loss. Requires NIR band.
        J, I_ref expected to have [B, C, H, W] where C >= 4 if NIR is included.
        If C=3, this function assumes it can't be used or Red/NIR indices need adjusting.
        """
        def calc_ndvi(img):
            nir = img[:, nir_idx:nir_idx+1, :, :]
            red = img[:, red_idx:red_idx+1, :, :]
            return (nir - red) / (nir + red + self.eps)

        ndvi_J = calc_ndvi(J)
        ndvi_I_ref = calc_ndvi(I_ref)
        
        clear_mask = 1.0 - M
        
        # Data term
        data_loss = (clear_mask * (ndvi_J - ndvi_I_ref) ** 2).sum() / (clear_mask.sum() + self.eps)
        
        # Smoothness term
        cloud_mask = M
        dy = ndvi_J[:, :, 1:, :] - ndvi_J[:, :, :-1, :]
        dx = ndvi_J[:, :, :, 1:] - ndvi_J[:, :, :, :-1]
        
        # Pad to match sizes
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        
        grad_mag_sq = dx**2 + dy**2
        smooth_loss = (cloud_mask * grad_mag_sq).sum() / (cloud_mask.sum() + self.eps)
        
        return data_loss + gamma * smooth_loss

    def tv_loss(self, t: torch.Tensor) -> torch.Tensor:
        """
        Isotropic Total Variation (TV) smoothness for transmission map.
        """
        dy = t[:, :, 1:, :] - t[:, :, :-1, :]
        dx = t[:, :, :, 1:] - t[:, :, :, :-1]
        
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        
        return torch.mean(torch.sqrt(dx**2 + dy**2 + self.eps))

    def combined_pinn_loss(self, I: torch.Tensor, J: torch.Tensor, t: torch.Tensor, A: torch.Tensor, M: torch.Tensor, 
                           I_ref: torch.Tensor = None, has_nir: bool = False,
                           lambda_asm: float = 1.0, lambda_rte: float = 0.5, 
                           lambda_ndvi: float = 0.3, lambda_tv: float = 0.1) -> dict:
        """
        Combined PINN Loss calculation.
        """
        l_recon = self.masked_recon_loss(I, J, t, A, M)
        l_asm = self.asm_loss(I, J, t, A)
        l_rte = self.rte_loss(t)
        l_tv = self.tv_loss(t)
        
        l_ndvi = torch.tensor(0.0, device=I.device)
        if has_nir and I_ref is not None:
            l_ndvi = self.ndvi_consistency_loss(J, I_ref, M)
            
        total_loss = l_recon + lambda_asm * l_asm + lambda_rte * l_rte + lambda_tv * l_tv + lambda_ndvi * l_ndvi
        
        return {
            "total_loss": total_loss,
            "l_recon": l_recon,
            "l_asm": l_asm,
            "l_rte": l_rte,
            "l_ndvi": l_ndvi,
            "l_tv": l_tv
        }

class DPSPrior:
    """
    Diffusion Posterior Sampling (DPS) Prior implementation.
    """
    def __init__(self, diffusion_model, alphas_cumprod: torch.Tensor):
        """
        diffusion_model: Pretrained DDPM score network epsilon_theta
        alphas_cumprod: \bar{\alpha}_t from DDPM schedule
        """
        self.diffusion_model = diffusion_model
        self.alphas_cumprod = alphas_cumprod

    def dps_guidance_loss(self, J_t: torch.Tensor, t_step: torch.Tensor, I: torch.Tensor, t_trans: torch.Tensor, A: torch.Tensor, zeta: float = 1.0) -> torch.Tensor:
        """
        Computes the DPS measurement guidance component.
        
        J_t: Noisy DIP prediction
        t_step: Diffusion timestep
        I: Cloudy image measurement
        t_trans: Transmission map
        A: Atmospheric light
        """
        # 1. Score matching / Denoising
        alpha_bar = self.alphas_cumprod[t_step].view(-1, 1, 1, 1)
        eps_theta = self.diffusion_model(J_t, t_step)
        
        # 2. Tweedie's formula for x0_hat (clean image estimate)
        J_0_hat = (J_t - torch.sqrt(1 - alpha_bar) * eps_theta) / torch.sqrt(alpha_bar)
        
        # 3. Measurement forward model A_ASM(x)
        I_hat = J_0_hat * t_trans + A * (1 - t_trans)
        
        # 4. DPS Loss: || I - A_ASM(J_0_hat) ||_2^2
        meas_loss = torch.norm(I - I_hat, p=2) ** 2
        
        return meas_loss, eps_theta
