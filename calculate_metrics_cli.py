import argparse
import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image
try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def pure_numpy_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Pure NumPy calculation of SSIM (Structural Similarity Index) fallback."""
    K1, K2, L = 0.01, 0.03, 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    if img1.ndim == 3:
        # Average SSIM across color channels
        scores = [pure_numpy_ssim(img1[:, :, c], img2[:, :, c]) for c in range(img1.shape[2])]
        return float(np.mean(scores))

    mu1 = float(np.mean(img1))
    mu2 = float(np.mean(img2))

    sigma1_sq = float(np.var(img1))
    sigma2_sq = float(np.var(img2))
    sigma12 = float(np.mean((img1 - mu1) * (img2 - mu2)))

    numerator = (2.0 * mu1 * mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / (denominator + 1e-8))


def compute_metrics(gt_path: str | Path, pred_path: str | Path) -> dict:
    gt_p = Path(gt_path)
    pred_p = Path(pred_path)

    if not gt_p.exists():
        raise FileNotFoundError(f"Ground Truth image not found: {gt_p}")
    if not pred_p.exists():
        raise FileNotFoundError(f"Restored image not found: {pred_p}")

    # Load images as RGB PIL Images
    gt_img = Image.open(gt_p).convert("RGB")
    pred_img = Image.open(pred_p).convert("RGB")

    # Resize pred to match GT if dimensions differ
    if gt_img.size != pred_img.size:
        print(f"⚠️  Note: Resizing restored image from {pred_img.size} to match GT {gt_img.size}...")
        pred_img = pred_img.resize(gt_img.size, Image.Resampling.LANCZOS)

    # Convert to float32 in range [0, 1]
    gt_arr = np.asarray(gt_img, dtype=np.float32) / 255.0
    pred_arr = np.asarray(pred_img, dtype=np.float32) / 255.0

    # 1. MSE & RMSE
    mse = float(np.mean((gt_arr - pred_arr) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(gt_arr - pred_arr)))

    # 2. PSNR (dB)
    if mse == 0:
        psnr_val = 100.0
    else:
        psnr_val = float(10.0 * np.log10(1.0 / mse))

    if HAS_SKIMAGE:
        sk_psnr_val = float(sk_psnr(gt_arr, pred_arr, data_range=1.0))
        ssim_rgb = float(sk_ssim(gt_arr, pred_arr, channel_axis=2, data_range=1.0))
        gt_gray = np.mean(gt_arr, axis=2)
        pred_gray = np.mean(pred_arr, axis=2)
        ssim_gray = float(sk_ssim(gt_gray, pred_gray, data_range=1.0))
    else:
        sk_psnr_val = psnr_val
        ssim_rgb = pure_numpy_ssim(gt_arr, pred_arr)
        gt_gray = np.mean(gt_arr, axis=2)
        pred_gray = np.mean(pred_arr, axis=2)
        ssim_gray = pure_numpy_ssim(gt_gray, pred_gray)

    # 4. SAM (Spectral Angle Mapper)
    gt_flat = gt_arr.reshape(-1, 3)
    pred_flat = pred_arr.reshape(-1, 3)
    dot = np.sum(gt_flat * pred_flat, axis=1)
    norms = np.linalg.norm(gt_flat, axis=1) * np.linalg.norm(pred_flat, axis=1) + 1e-8
    cos_sim = np.clip(dot / norms, -1.0, 1.0)
    sam_rad = float(np.mean(np.arccos(cos_sim)))
    sam_deg = float(np.degrees(sam_rad))

    return {
        "gt_file": gt_p.name,
        "pred_file": pred_p.name,
        "resolution": f"{gt_img.width}x{gt_img.height}",
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr_val,
        "sk_psnr": sk_psnr_val,
        "ssim_rgb": ssim_rgb,
        "ssim_gray": ssim_gray,
        "sam_rad": sam_rad,
        "sam_deg": sam_deg,
    }


def print_report(res: dict):
    print("\n" + "=" * 60)
    print("📊 IMAGE EVALUATION METRICS VERIFICATION REPORT")
    print("=" * 60)
    print(f"  • Ground Truth Image : {res['gt_file']}")
    print(f"  • Restored Image     : {res['pred_file']}")
    print(f"  • Resolution         : {res['resolution']} (RGB)")
    print("-" * 60)
    print(f"  🎯 PSNR (Pure Math)  : {res['psnr']:.4f} dB")
    print(f"  🎯 PSNR (scikit-img) : {res['sk_psnr']:.4f} dB")
    print(f"  🌟 SSIM (RGB 3-Ch)   : {res['ssim_rgb']:.4f}")
    print(f"  🌟 SSIM (Grayscale)  : {res['ssim_gray']:.4f}")
    print(f"  📉 RMSE              : {res['rmse']:.6f}")
    print(f"  📉 MAE               : {res['mae']:.6f}")
    print(f"  🌈 SAM (deg / rad)   : {res['sam_deg']:.2f}° ({res['sam_rad']:.4f} rad)")
    print("=" * 60)
    print("💡 Note: Compare 'PSNR (scikit-img)' & 'SSIM' with your Frontend UI.")
    print("=" * 60 + "\n")


def select_file(title: str) -> str:
    # 1. Try Google Colab upload if running in Colab environment
    try:
        import importlib
        colab_files = importlib.import_module("google.colab.files")
        print(f"\n📤 [Colab Mode] Please click 'Choose Files' to upload {title}:")
        uploaded = colab_files.upload()
        if uploaded:
            filename = list(uploaded.keys())[0]
            print(f"✅ Uploaded: {filename}")
            return filename
    except Exception:
        pass

    # 2. Try Tkinter GUI File Picker Dialog (Windows Desktop)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        root.update()
        print(f"\n📂 Opening Windows File Explorer popup to select: {title}...")
        file_path = filedialog.askopenfilename(
            title=f"Select {title}",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif"), ("All Files", "*.*")]
        )
        root.destroy()
        if file_path and Path(file_path).exists():
            print(f"✅ Selected: {file_path}")
            return file_path
    except Exception as e:
        print(f"⚠️ GUI dialog non-interactive: {e}")

    # 3. Terminal Drag-and-Drop / Path Input Fallback
    print(f"\n📥 Drag & Drop or paste the path to {title} below:")
    path_in = input(f"Enter path to {title}: ").strip(' "\'')
    return path_in


def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, RMSE, SAM between GT and Restored images.")
    parser.add_argument("--gt", type=str, help="Path to Ground Truth (Label/Clean) image")
    parser.add_argument("--pred", type=str, help="Path to Restored (Reconstructed/Output) image")
    args = parser.parse_args()

    gt_path = args.gt
    pred_path = args.pred

    if not gt_path:
        gt_path = select_file("Ground Truth (Label / Clean) Image")
    if not pred_path:
        pred_path = select_file("Restored (Reconstructed / Output) Image")

    try:
        results = compute_metrics(gt_path, pred_path)
        print_report(results)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
