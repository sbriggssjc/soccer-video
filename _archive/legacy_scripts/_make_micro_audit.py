"""Make micro-sized individual frames for viewing"""
from PIL import Image
from pathlib import Path

base = Path(r"D:\Projects\soccer-video\out\visual_audit")
out = Path(r"D:\Projects\soccer-video\out\visual_audit\micro")
out.mkdir(exist_ok=True)

# Key frames to check
targets = [0, 60, 120, 180, 240, 300, 330, 360, 390, 420, 480]

files = sorted(base.glob("audit_f*.jpg"))
frame_map = {}
for f in files:
    fn = int(f.stem.replace("audit_f", ""))
    frame_map[fn] = f

for t in targets:
    best_fn = min(frame_map.keys(), key=lambda x: abs(x - t))
    img = Image.open(frame_map[best_fn])
    # 256x144 at quality 40 should be ~5-8KB
    img = img.resize((256, 144), Image.LANCZOS)
    fname = out / f"f{best_fn:04d}.jpg"
    img.save(fname, quality=40)
    sz = fname.stat().st_size
    print(f"f{best_fn:04d}: {sz} bytes")
