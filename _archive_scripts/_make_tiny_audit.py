"""Make tiny contact sheets (2x2, very small thumbnails) for viewing"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

base = Path(r"D:\Projects\soccer-video\out\visual_audit")
files = sorted(base.glob("audit_f*.jpg"))

# Extract frame numbers from filenames
frame_nums = []
for f in files:
    fn = f.stem.replace("audit_f", "")
    frame_nums.append((int(fn), f))

# Select key frames across clip: every ~60 frames plus critical zones
key_frames = [0, 60, 120, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480]
# Also grab nearest available frame for each target
selected = []
for target in key_frames:
    best = min(frame_nums, key=lambda x: abs(x[0] - target))
    if best not in selected:
        selected.append(best)

# Make a single small contact sheet
COLS, ROWS = 2, 2
THUMB_W, THUMB_H = 384, 216
PER_SHEET = COLS * ROWS

for si in range(0, len(selected), PER_SHEET):
    batch = selected[si:si+PER_SHEET]
    sheet = Image.new("RGB", (COLS*THUMB_W, ROWS*THUMB_H), (30,30,30))
    draw = ImageDraw.Draw(sheet)
    
    for j, (fnum, fpath) in enumerate(batch):
        col = j % COLS
        row = j // COLS
        img = Image.open(fpath).resize((THUMB_W, THUMB_H), Image.LANCZOS)
        x_off = col * THUMB_W
        y_off = row * THUMB_H
        sheet.paste(img, (x_off, y_off))
        # Label
        draw.text((x_off + 5, y_off + 2), f"f{fnum}", fill=(255, 255, 0))
    
    out = base / f"tiny_{si//PER_SHEET:02d}.jpg"
    sheet.save(out, quality=65)
    print(f"Saved {out.name}: frames {[b[0] for b in batch]}")
