import torch
import torch.nn.functional as F
import numpy as np
from skimage.feature import local_binary_pattern

def extract_edges(x: torch.Tensor) -> torch.Tensor:
    """
    Extracts Sobel edge magnitude and Laplacian gradient maps.
    """
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device).view(1, 1, 3, 3)
    klap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device).view(1, 1, 3, 3)
    
    B, C, H, W = x.shape
    x_flat = x.reshape(B * C, 1, H, W)
    gx = F.conv2d(x_flat, kx, padding=1)
    gy = F.conv2d(x_flat, ky, padding=1)
    sobel = torch.sqrt(gx**2 + gy**2 + 1e-8)
    lap = torch.abs(F.conv2d(x_flat, klap, padding=1))
    
    edges = torch.cat([sobel, lap], dim=1)  # [B*C, 2, H, W]
    return edges.reshape(B, C * 2, H, W)

def extract_gabor(x: torch.Tensor, orientations: int = 4, frequencies: list[float] = [0.1, 0.25]) -> torch.Tensor:
    """
    Extracts Gabor filter bank magnitude features across multiple orientations and frequencies.
    """
    B, C, H, W = x.shape
    kernels = []
    size = 9
    sigma_x = 2.0
    sigma_y = 2.0
    for i in range(orientations):
        theta = i * np.pi / orientations
        for freq in frequencies:
            y_grid, x_grid = torch.meshgrid(
                torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=x.device),
                torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=x.device),
                indexing='ij'
            )
            xr = x_grid * np.cos(theta) + y_grid * np.sin(theta)
            yr = -x_grid * np.sin(theta) + y_grid * np.cos(theta)
            kernel = torch.exp(-0.5 * (xr**2 / sigma_x**2 + yr**2 / sigma_y**2)) * torch.cos(2 * np.pi * freq * xr)
            kernel = kernel - kernel.mean()
            kernel = kernel / (kernel.norm() + 1e-8)
            kernels.append(kernel.view(1, 1, size, size))
            
    g_filter = torch.cat(kernels, dim=0)  # [orientations * len(frequencies), 1, K, K]
    x_flat = x.reshape(B * C, 1, H, W)
    features = F.conv2d(x_flat, g_filter, padding=4)
    return features.reshape(B, C * len(kernels), H, W)

def extract_wavelets(x: torch.Tensor) -> torch.Tensor:
    """
    Stationary Haar wavelet transform yielding LL, LH, HL, and HH subbands.
    """
    B, C, H, W = x.shape
    h_filter = torch.tensor([
        [[[1., 1.], [1., 1.]]],     # LL
        [[[1., -1.], [1., -1.]]],   # LH
        [[[1., 1.], [-1., -1.]]],   # HL
        [[[1., -1.], [-1., 1.]]]    # HH
    ], device=x.device) * 0.5
    
    x_flat = x.reshape(B * C, 1, H, W)
    out = F.conv2d(x_flat, h_filter, padding=1)
    out = out[:, :, :H, :W]  # Crop padding to preserve spatial size H, W
    return out.reshape(B, C * 4, H, W)

def extract_fft(x: torch.Tensor) -> torch.Tensor:
    """
    Computes normalized 2D FFT magnitude spectrum features.
    """
    fft_feat = torch.fft.fft2(x)
    mag = torch.abs(fft_feat)
    mag = mag / (mag.max() + 1e-8)
    return mag

def extract_lbp(x_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes Local Binary Patterns (LBP) on the CPU and transfers back to device.
    x_tensor: [B, C, H, W]
    """
    device = x_tensor.device
    x_np = x_tensor.detach().cpu().numpy()
    B, C, H, W = x_np.shape
    lbp_batch = []
    for b in range(B):
        lbp_channels = []
        for c in range(C):
            img_2d = (np.clip(x_np[b, c], 0.0, 1.0) * 255.0).astype(np.uint8)
            lbp = local_binary_pattern(img_2d, P=8, R=1, method='uniform')
            lbp = lbp / 10.0  # Normalize to [0, 1] range (uniform LBP has 10 bins)
            lbp_channels.append(lbp)
        lbp_batch.append(np.stack(lbp_channels, axis=0))
    lbp_tensor = torch.from_numpy(np.stack(lbp_batch, axis=0)).float().to(device)
    return lbp_tensor

def extract_all_features(x: torch.Tensor, lbp_tensor: torch.Tensor | None = None) -> torch.Tensor:
    """
    Extracts and fuses edges, Gabor, Haar wavelets, FFT magnitude, and LBP.
    x: [B, C, H, W]
    lbp_tensor: [B, C, H, W] (LBP map for this scale)
    Returns: [B, C_features, H, W]
    """
    # 1. Edges: Sobel + Laplacian [B, C*2, H, W]
    edges = extract_edges(x)
    # 2. Gabor textures: [B, C*8, H, W]
    gabor = extract_gabor(x)
    # 3. Wavelets LL, LH, HL, HH: [B, C*4, H, W]
    wavelets = extract_wavelets(x)
    # 4. FFT magnitude is removed because it is in the frequency domain and
    # causes severe geometric/triangular artifacts when processed by spatial convolutions.
    
    features_list = [x, edges, gabor, wavelets]
    if lbp_tensor is not None:
        features_list.append(lbp_tensor)
    else:
        # Fallback LBP on the fly
        features_list.append(extract_lbp(x))
        
    return torch.cat(features_list, dim=1)
