from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from captum.attr import IntegratedGradients

def normalize_attribution(attribution: torch.Tensor) -> np.ndarray:
    """Normalize attribution map for visualization."""
    attr_np = attribution.squeeze().cpu().detach().numpy()
    
    # Take absolute value to show magnitude of importance
    attr_np = np.abs(attr_np)
    
    # 99th percentile clipping to remove extreme outliers
    p99 = np.percentile(attr_np, 99)
    if p99 > 0:
        attr_np = np.clip(attr_np, 0, p99) / p99
    else:
        attr_np = attr_np / (attr_np.max() + 1e-8)
        
    return attr_np

def generate_unet_explanation(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> Image.Image:
    """
    Generate an attribution heatmap for the U-Net model using Integrated Gradients.
    The heatmap highlights which pixels contributed most to the 'cloud' classification.
    """
    model.eval()
    
    # We explain the sum of the logits (overall cloudiness)
    def wrapper_func(x):
        return model(x).sum().unsqueeze(0)
    
    ig = IntegratedGradients(wrapper_func)
    
    # Calculate attributions
    # Baseline is zero (black image)
    baseline = torch.zeros_like(input_tensor).to(device)
    
    attributions, delta = ig.attribute(
        input_tensor, 
        baseline, 
        n_steps=15, 
        internal_batch_size=1,
        return_convergence_delta=True
    )
    
    # Average attributions across color channels (C, H, W) -> (H, W)
    attr_mean = attributions.mean(dim=1)
    
    # Normalize to [0, 1]
    heatmap_np = normalize_attribution(attr_mean)
    
    # Convert to heatmap color (using matplotlib inferno colormap conceptually, but here we'll map to a simple RGB)
    import matplotlib.pyplot as plt
    colormap = plt.get_cmap('inferno')
    heatmap_colored = colormap(heatmap_np)[:, :, :3] # Drop alpha
    
    # Convert to PIL Image
    heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8), mode="RGB")
    return heatmap_img
