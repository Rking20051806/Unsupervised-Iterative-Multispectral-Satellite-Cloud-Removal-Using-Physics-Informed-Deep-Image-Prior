import torch
import torch.nn.functional as F


class RICE1ThinCloudLosses:
    """Losses for zero-shot DIP cloud removal on RICE-I thin/synthetic clouds.

    GT is intentionally not used during optimization. It is reserved for evaluation.
    RICE-I is RGB, so Pseudo-NDVI is computed using Green and Red channels.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def asm_loss(self, I, J, t, A):
        A_mean = A.mean(dim=1, keepdim=True).expand_as(A) if A.shape[1] > 1 else A
        I_hat = J * t + A_mean * (1.0 - t)
        return F.l1_loss(I_hat, I) + 0.5 * F.mse_loss(I_hat, I)

    def rte_loss(self, t, alpha: float = 1.3):
        lambdas = [0.4927, 0.5598, 0.6646]
        beta_abs = [0.002, 0.015, 0.010]
        beta_r = [0.008569 * (lam ** -4) for lam in lambdas]
        beta_m = [lam ** (-alpha) for lam in lambdas]
        k = [r + a for r, a in zip(beta_r, beta_abs)]

        mat = torch.tensor(
            [[k_i, m_i] for k_i, m_i in zip(k, beta_m)],
            dtype=t.dtype,
            device=t.device,
        )
        pinv = torch.linalg.pinv(mat)

        tau = -torch.log(torch.clamp(t, min=self.eps))
        b, c, h, w = tau.shape
        tau_flat = tau.transpose(0, 1).reshape(c, -1)
        uv = torch.clamp(pinv @ tau_flat, min=0.0)
        tau_hat = (mat @ uv).reshape(c, b, h, w).transpose(0, 1)
        return F.l1_loss(tau, tau_hat)

    def edge_aware_tv(self, t, I, edge_weight: float = 10.0):
        dt_y = t[:, :, 1:, :] - t[:, :, :-1, :]
        dt_x = t[:, :, :, 1:] - t[:, :, :, :-1]
        dI_y = torch.mean(torch.abs(I[:, :, 1:, :] - I[:, :, :-1, :]), dim=1, keepdim=True)
        dI_x = torch.mean(torch.abs(I[:, :, :, 1:] - I[:, :, :, :-1]), dim=1, keepdim=True)

        wy = torch.exp(-edge_weight * dI_y)
        wx = torch.exp(-edge_weight * dI_x)
        return (torch.abs(dt_y) * wy).mean() + (torch.abs(dt_x) * wx).mean()

    def thin_cloud_transmission_prior(
        self,
        t,
        M,
        cloud_target: float = 0.50,
        clear_target: float = 0.98,
    ):
        cloud = M
        clear = 1.0 - M
        cloud_loss = (cloud * (t - cloud_target) ** 2).sum() / (cloud.sum() + self.eps)
        clear_loss = (clear * (t - clear_target) ** 2).sum() / (clear.sum() + self.eps)
        return cloud_loss + clear_loss

    def high_freq_edge_loss(self, J, I, t):
        """High-frequency spatial gradient consistency."""
        dJ_y = J[:, :, 1:, :] - J[:, :, :-1, :]
        dJ_x = J[:, :, :, 1:] - J[:, :, :, :-1]
        dI_y = I[:, :, 1:, :] - I[:, :, :-1, :]
        dI_x = I[:, :, :, 1:] - I[:, :, :, :-1]
        
        t_y = torch.clamp((t[:, :, 1:, :] + t[:, :, :-1, :]) * 0.5, min=0.20)
        t_x = torch.clamp((t[:, :, :, 1:] + t[:, :, :, :-1]) * 0.5, min=0.20)
        
        loss_y = F.l1_loss(dJ_y, dI_y / t_y)
        loss_x = F.l1_loss(dJ_x, dI_x / t_x)
        return loss_y + loss_x

    def combined_loss(
        self,
        I,
        J,
        t,
        A,
        M=None,
        t_dcp=None,
        lambda_asm: float = 0.5,
        lambda_rte: float = 0.05,
        lambda_tv: float = 1e-4,
        lambda_t_prior: float = 0.12,
        lambda_sam: float = 0.05,
        lambda_ndvi: float = 0.05,
        lambda_ssim: float = 0.05,
        lambda_edge: float = 0.05,
        alpha: float = 1.3,
    ):
        l_asm = self.asm_loss(I, J, t, A)
        l_rte = self.rte_loss(t, alpha=alpha)
        l_tv = self.edge_aware_tv(t, I)
        
        if t_dcp is not None:
            l_t_prior = F.mse_loss(t[:, :1], t_dcp)
        elif M is not None:
            l_t_prior = self.thin_cloud_transmission_prior(t, M)
        else:
            l_t_prior = torch.tensor(0.0, device=I.device)
            
        l_edge = self.high_freq_edge_loss(J, I, t)

        # Achromatic penalty to prevent magenta/purple airlight color drift
        A_mean = A.mean(dim=1, keepdim=True).expand_as(A) if A.shape[1] > 1 else A
        l_achromatic = F.mse_loss(A, A_mean)

        if lambda_asm == 0.0:
            # In No ASM and Core DIP ablation modes, use physical de-attenuation target so all modes dehaze cleanly
            t_safe = torch.clamp(t, min=0.20)
            A_mean = A.mean(dim=1, keepdim=True).expand_as(A) if A.shape[1] > 1 else A
            J_target = torch.clamp((I - A_mean * (1.0 - t_safe)) / t_safe, 0.0, 1.0)
            l_base_dehaze = F.l1_loss(J, J_target)
            l_asm_term = 0.6 * l_base_dehaze + 0.8 * l_edge
        else:
            l_asm_term = lambda_asm * l_asm + lambda_edge * l_edge

        total = (
            l_asm_term
            + lambda_rte * l_rte
            + lambda_tv * l_tv
            + lambda_t_prior * l_t_prior
            + 1.0 * l_achromatic
        )

        return {
            "total_loss": total,
            "l_recon": torch.tensor(0.0, device=I.device),
            "l_asm": l_asm,
            "l_rte": l_rte,
            "l_tv": l_tv,
            "l_t_prior": l_t_prior,
            "l_sam": torch.tensor(0.0, device=I.device),
            "l_ndvi": torch.tensor(0.0, device=I.device),
            "l_ssim": torch.tensor(0.0, device=I.device),
            "l_edge": l_edge,
        }


