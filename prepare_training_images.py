import os
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def collect_images(sources, dest, max_images=1000):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sources:
        src_path = Path(src).expanduser()
        if not src_path.exists():
            continue
        for root, _, files in os.walk(src_path):
            for fn in files:
                if count >= max_images:
                    return count
                ext = Path(fn).suffix.lower()
                if ext in IMAGE_EXTS:
                    src_file = Path(root) / fn
                    # Avoid name collisions
                    out_file = dest / f"{count:06d}{ext}"
                    try:
                        shutil.copy2(src_file, out_file)
                        count += 1
                    except Exception:
                        pass
    return count

if __name__ == '__main__':
    # Common Windows user folders
    user = Path.home()
    sources = [
        user / 'Pictures',
        user / 'Downloads',
        user / 'Desktop'
    ]
    dest = Path('data/train_photos')
    n = collect_images(sources, dest, max_images=1500)
    print(f"Collected {n} images into {dest}")



