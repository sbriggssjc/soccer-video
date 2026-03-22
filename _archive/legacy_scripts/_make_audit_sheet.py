"""Make small contact sheets from visual audit frames for viewing"""
from PIL import Image
from pathlib import Path
import glob

base = Path(r"D:\Projects\soccer-video\out\visual_audit")
files = sorted(base.glob("audit_f*.jpg"))
print(f"Found {len(files)} audit frames")

# Make sheets of 12 frames each (4 cols x 3 rows)
COLS, ROWS = 4, 3
PER_SHEET = COLS * ROWS
THUMB_W, THUMB_H = 320, 180

sheets = []
for i in range(0, len(files), PER_SHEET):
    batch = files[i:i+PER_SHEET]
    sheet_w = COLS * THUMB_W
    sheet_h = ROWS * THUMB_H
    sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
    
    for j, fpath in enumerate(batch):
        col = j % COLS
        row = j // COLS
        img = Image.open(fpath).resize((THUMB_W, THUMB_H), Image.LANCZOS)
        sheet.paste(img, (col * THUMB_W, row * THUMB_H))
    
    sheet_name = base / f"sheet_{i//PER_SHEET:02d}.jpg"
    sheet.save(sheet_name, quality=80)
    sheets.append(sheet_name)
    print(f"Saved {sheet_name.name} ({len(batch)} frames)")

print(f"Done: {len(sheets)} contact sheets")
