import torch
import torch.nn.functional as F
import numpy as np

def calculate_sam(img1: torch.Tensor, img2: torch.Tensor, eps=1e-8) -> float:
    """
    Calculates the Spectral Angle Mapper (SAM) between two images.
    Measures the spectral similarity. Smaller is better.
    img1, img2: [C, H, W] tensors
    """
    # Flatten spatial dimensions: [C, H*W]
    v1 = img1.reshape(img1.shape[0], -1)
    v2 = img2.reshape(img2.shape[0], -1)
    
    # Dot product along channels
    dot_product = torch.sum(v1 * v2, dim=0)
    
    # Norms
    norm1 = torch.norm(v1, dim=0)
    norm2 = torch.norm(v2, dim=0)
    
    # Calculate angle
    cos_theta = dot_product / (norm1 * norm2 + eps)
    # Clamp for numerical stability before acos
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    
    # Angle in radians
    angles = torch.acos(cos_theta)
    return float(torch.mean(angles).item())

def calculate_prs(I: torch.Tensor, J: torch.Tensor, t: torch.Tensor, A: torch.Tensor, has_nir: bool = False, nir_idx: int = 3, red_idx: int = 0) -> dict:
    """
    Physical Realism Score (PRS) - No-reference metric.
    PRS = ASM residual + NDVI consistency (if NIR available) + SAM (Spectral Angle Mapper).
    Lower PRS means higher physical realism.
    
    I: Original cloudy image [C, H, W]
    J: Predicted clean image [C, H, W]
    t: Predicted transmission map [1, H, W]
    A: Atmospheric light [C, 1, 1]
    """
    # Ensure inputs have batch dimension for processing if needed
    if I.dim() == 3:
        I = I.unsqueeze(0)
    if J.dim() == 3:
        J = J.unsqueeze(0)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if A.dim() == 3:
        A = A.unsqueeze(0)

    # Ensure device consistency across all input tensors
    device = J.device
    I = I.to(device)
    t = t.to(device)
    A = A.to(device)

    # Resize I and t if spatial resolution differs during multi-scale DIP optimization
    if I.shape[-2:] != J.shape[-2:]:
        I = F.interpolate(I, size=J.shape[-2:], mode='bilinear', align_corners=False)
    if t.shape[-2:] != J.shape[-2:]:
        t = F.interpolate(t, size=J.shape[-2:], mode='bilinear', align_corners=False)

    # 1. ASM Residual (MSE of the physical forward model)
    I_hat = J * t + A * (1 - t)
    asm_residual = F.mse_loss(I_hat, I).item()
    
    # 2. SAM (Spectral Angle Mapper between predicted clean and original cloudy)
    # In cloud-free regions, SAM should be close to 0.
    sam_score = calculate_sam(I.squeeze(0), J.squeeze(0))
    
    # 3. NDVI Consistency (Smoothness)
    ndvi_smoothness = 0.0
    if has_nir and J.shape[1] >= 4:
        nir = J[:, nir_idx:nir_idx+1, :, :]
        red = J[:, red_idx:red_idx+1, :, :]
        ndvi = (nir - red) / (nir + red + 1e-8)
        
        # Calculate spatial gradients (smoothness)
        dy = ndvi[:, :, 1:, :] - ndvi[:, :, :-1, :]
        dx = ndvi[:, :, :, 1:] - ndvi[:, :, :, :-1]
        
        # Pad to match sizes
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        
        ndvi_smoothness = float(torch.mean(torch.sqrt(dx**2 + dy**2 + 1e-8)).item())

    # Combined PRS (weighted sum, these weights can be tuned)
    w_asm = 1.0
    w_sam = 0.5
    w_ndvi = 0.2 if has_nir else 0.0
    
    prs_total = (w_asm * asm_residual) + (w_sam * sam_score) + (w_ndvi * ndvi_smoothness)
    
    return {
        "PRS": round(prs_total, 5),
        "components": {
            "ASM_Residual": round(asm_residual, 5),
            "SAM": round(sam_score, 5),
            "NDVI_Smoothness": round(ndvi_smoothness, 5) if has_nir else None
        }
    }

if __name__ == "__main__":
    # Quick sanity check
    I = torch.rand((3, 256, 256))
    J = torch.rand((3, 256, 256))
    t = torch.rand((1, 256, 256))
    A = torch.rand((3, 1, 1))
    
    score = calculate_prs(I, J, t, A)
    print("Test PRS Score:", score)
