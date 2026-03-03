"""Create small diagnostic thumbnails for quick viewing."""
import cv2
from pathlib import Path

IN_DIR = Path("D:/Projects/soccer-video/_tmp/diag_crops")
OUT_DIR = Path("D:/Projects/soccer-video/_tmp/diag_thumbs")
OUT_DIR.mkdir(exist_ok=True)

for img_path in sorted(IN_DIR.glob("*.jpg")):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    # Scale to max 640px wide
    scale = 640 / w
    thumb = cv2.resize(img, (int(w*scale), int(h*scale)))
    out = OUT_DIR / img_path.name
    cv2.imwrite(str(out), thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    print(f"{img_path.name}: {w}x{h} -> {thumb.shape[1]}x{thumb.shape[0]}")
