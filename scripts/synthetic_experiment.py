import os
import sys
import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dip_loop import optimize_dip
from prs_metric import calculate_prs
from evaluate import calculate_psnr, calculate_ssim, calculate_sam

def create_synthetic_obstacle(clear_image_path, obstacle_type='mesh'):
    img = Image.open(clear_image_path).convert('RGB')
    W, H = img.size
    
    mask = Image.new('L', (W, H), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_img = ImageDraw.Draw(img)
    
    if obstacle_type == 'mesh':
        for x in range(0, W, 40):
            draw_img.line([(x, 0), (x, H)], fill=(240, 240, 240), width=6)
            draw_mask.line([(x, 0), (x, H)], fill=255, width=6)
        for y in range(0, H, 40):
            draw_img.line([(0, y), (W, y)], fill=(240, 240, 240), width=6)
            draw_mask.line([(0, y), (W, y)], fill=255, width=6)
            
    elif obstacle_type == 'shapes':
        draw_img.rectangle([W//4, H//4, 3*W//4, 3*H//4], fill=(230, 230, 230))
        draw_mask.rectangle([W//4, H//4, 3*W//4, 3*H//4], fill=255)
        draw_img.ellipse([W//8, H//8, 3*W//8, 3*H//8], fill=(255, 255, 255))
        draw_mask.ellipse([W//8, H//8, 3*W//8, 3*H//8], fill=255)
        
    elif obstacle_type == 'cloud':
        draw_img.ellipse([W//3, H//3, 2*W//3, 2*H//3], fill=(255, 255, 255))
        draw_mask.ellipse([W//3, H//3, 2*W//3, 2*H//3], fill=255)
        
    transform = transforms.ToTensor()
    clean_img_original = Image.open(clear_image_path).convert('RGB')
    
    clean_tensor = transform(clean_img_original)
    cloudy_tensor = transform(img)
    mask_tensor = transform(mask).squeeze(0)
    
    return cloudy_tensor, mask_tensor, clean_tensor

def run_experiment(clear_imge_path, obstacle_type='mesh', iters=200):
    print(f'Generating synthetic degraded image ({obstacle_type}) from {clear_imge_path}...')
    cloudy, mask, clean = create_synthetic_obstacle(clear_imge_path, obstacle_type)
    
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'synthetic_experiment')
    os.path.dirname(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print('Running DIP+PINN optimization to remove the synthetic obstacle...')
    t_start = time.time()
    pred_J, pred_t = optimize_dip(cloudy, mask, num_iters=iters, use_gpu=torch.cuda.is_available(), use_dps=False)
    elapsed = time.time() - t_start
    
    clean_np = clean.permute(1, 2, 0).numpy()
    cloudy_np = cloudy.permute(1, 2, 0).numpy()
    pred_np = pred_J.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()
    
    psnr_val = calculate_psnr(clean_np, pred_np)
    ssim_val = calculate_ssim(clean_np, pred_np)
    sam_val = calculate_sam(clean_np, pred_np)
    
    print(f'\nReconstruction Metrics against Ground Truth:')
    print(f'  PSNR: {psnr_val}') # simple print to avoid formatting issues
    print(f'  SSIM: {ssim_val}')
    print(f'  SAM:  {sam_val}')
    print(f'  Time: {elapsed}s')
    
    comparison = Image.new('RGB', (256 * 4, 256))
    
    clean_pil = Image.fromarray((clean_np * 255).astype(np.uint8)).resize((256, 256))
    cloudy_pil = Image.fromarray((cloudy_np * 255).astype(np.uint8)).resize((256, 256))
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((256, 256))
    pred_pil = Image.fromarray((pred_np * 255).astype(np.uint8)).resize((256, 256))
    
    comparison.paste(clean_pil, (0, 0))
    comparison.paste(cloudy_pil, (256, 0))
    comparison.paste(mask_pil, (256 * 2, 0))
    comparison.paste(pred_pil, (256 * 3, 0))
    
    comparison_path = os.path.join(output_dir, f'result_{obstacle_type}.png')
    comparison.save(comparison_path)
    print(f'Saved visual comparison at: {comparison_path}')

if __name__ == '__main__':
    sample_clear_path = os.path.join(PROJECT_ROOT, 'RICE_DATASET', 'RICE1', 'label', '0.png')
    if not os.path.exists(sample_clear_path):
        os.makedirs(os.path.dirname(sample_clear_path), exist_ok=True)
        dummy_img = Image.new('RGB', (256, 256), (100, 150, 50))
        draw = ImageDraw.Draw(dummy_img)
        draw.rectangle([50, 50, 150, 150], fill=(200, 100, 80))
        draw.line([0, 0, 256, 256], fill=(50, 100, 200), width=10)
        dummy_img.save(sample_clear_path)
        
    run_experiment(sample_clear_path, obstacle_type='mesh', iters=100)
    run_experiment(sample_clear_path, obstacle_type='shapes', iters=100)