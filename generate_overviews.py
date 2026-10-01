import os
from s2_loader import Sentinel2Loader
from PIL import Image

def generate_overview(safe_dir, output_path):
    print(f"Generating overview for {safe_dir}...")
    try:
        loader = Sentinel2Loader(safe_dir)
        tci_rgb, mask_arr = loader.get_ui_overview(max_dim=1024)
        img = Image.fromarray(tci_rgb)
        img.save(output_path)
        print(f"Saved {output_path}")
    except Exception as e:
        print(f"Error processing {safe_dir}: {e}")

def main():
    safes = [
        'China.SAFE',
        'Vidarbha_Nagpur_Maharashtra.SAFE',
        'Vidarbha_Yavatmal_Maharashtra.SAFE'
    ]
    
    out_dir = r'C:\Users\01soj\.gemini\antigravity-ide\brain\e560c5a9-042b-4aea-8fc7-30b146f3e772'
    
    for safe in safes:
        if os.path.isdir(safe):
            out_name = safe.replace('.SAFE', '_overview.png')
            generate_overview(safe, os.path.join(out_dir, out_name))

if __name__ == '__main__':
    main()
