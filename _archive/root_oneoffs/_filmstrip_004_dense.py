"""Generate dense filmstrip for clip 004 frames 450-545 every 5 frames.
Labeled with frame numbers, timestamps, and percentage grid.
"""
import os, subprocess, traceback
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CLIP = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00.mp4"
OUTPUT = r"C:\Users\scott\Desktop\filmstrip_004_dense.png"
RESULT = r"D:\Projects\soccer-video\_tmp\filmstrip_004_dense_result.txt"
TEMP_DIR = r"D:\Projects\soccer-video\_tmp\filmstrip_004_dense_frames"
os.makedirs(TEMP_DIR, exist_ok=True)

FPS = 30
FRAME_START = 450
FRAME_END = 545
FRAME_STEP = 5

try:
    # Extract specific frames using ffmpeg select filter
    frames_to_extract = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    
    # Clear old frames
    for old in Path(TEMP_DIR).glob("*.jpg"):
        old.unlink()

    # Extract each frame individually
    for fr in frames_to_extract:
        out_path = os.path.join(TEMP_DIR, f"frame_{fr:04d}.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", CLIP,
            "-vf", f"select=eq(n\\,{fr})",
            "-vframes", "1", "-q:v", "2", out_path
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    # Verify extraction
    extracted = sorted(Path(TEMP_DIR).glob("frame_*.jpg"))
    n = len(extracted)
    if n == 0:
        raise FileNotFoundError("No frames extracted")

    # Tile into contact sheet
    cols = 4
    rows = (n + cols - 1) // cols
    tw = 480

    sample = Image.open(extracted[0])
    fw, fh = sample.size
    scale = tw / fw
    th = int(fh * scale)
    pad = 6

    sheet_w = cols * tw + (cols - 1) * pad
    sheet_h = rows * th + (rows - 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))

    try:
        font_label = ImageFont.truetype("arial.ttf", 16)
        font_grid = ImageFont.truetype("arial.ttf", 11)
    except:
        font_label = ImageFont.load_default()
        font_grid = ImageFont.load_default()

    for i, (fp, fr) in enumerate(zip(extracted, frames_to_extract)):
        r_idx = i // cols
        c_idx = i % cols
        x0 = c_idx * (tw + pad)
        y0 = r_idx * (th + pad)

        img = Image.open(fp).resize((tw, th), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        time_s = fr / FPS

        # Grid lines at 10% intervals
        for pct in range(0, 101, 10):
            gx = int(pct / 100.0 * tw)
            line_color = (255, 255, 0) if pct % 50 == 0 else (180, 180, 0)
            width = 2 if pct % 50 == 0 else 1
            draw.line([(gx, 0), (gx, th)], fill=line_color, width=width)
            if pct % 20 == 0:
                draw.text((gx + 2, th - 14), f"{pct}%", fill=(255, 255, 0), font=font_grid)

        # Frame label
        label = f"f{fr}  {time_s:.1f}s"
        bbox = draw.textbbox((0, 0), label, font=font_label)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        draw.rectangle([(4, 2), (8 + lw, 6 + lh)], fill=(0, 0, 0, 180))
        draw.text((6, 3), label, fill=(255, 255, 255), font=font_label)

        sheet.paste(img, (x0, y0))

    sheet.save(OUTPUT, quality=92)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)

    msg = f"SUCCESS\n{n} frames (f{FRAME_START}-f{FRAME_END} every {FRAME_STEP})\n"
    msg += f"Grid: {cols}x{rows}, Output: {OUTPUT} ({size_mb:.1f} MB)\n"
    with open(RESULT, "w") as f:
        f.write(msg)

except Exception as e:
    with open(RESULT, "w") as f:
        f.write(f"EXCEPTION: {e}\n{traceback.format_exc()}")
