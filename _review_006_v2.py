"""Generate filmstrip + review CSV for NEOFC clip 006 (v2 - denser sampling).
Extract frames at 3fps (every 10 source frames) for better anchor coverage.
"""
import os, sys, csv, traceback, subprocess, shutil, re, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CLIP_NUM = "006"
GAME = "2026-02-23__TSC_vs_NEOFC"
os.makedirs(r"D:\Projects\soccer-video\_tmp", exist_ok=True)

FPS_SRC = 30
EXTRACT_FPS = 3  # denser than before (was 2)
FRAMES_PER_EXTRACT = FPS_SRC // EXTRACT_FPS  # = 10
FRAME_W = 1920

clips_dir = Path(rf"D:\Projects\soccer-video\out\atomic_clips\{GAME}")
clip_file = next(clips_dir.glob(f"{CLIP_NUM}__*.mp4"))
stem = clip_file.stem

# Get duration
m = re.search(r"t([\d.]+)-t([\d.]+)", stem)
duration = float(m.group(2)) - float(m.group(1))
total_frames = int(duration * FPS_SRC)

try:
    # Extract frames at 3fps
    frames_dir = rf"D:\Projects\soccer-video\_tmp\filmstrip_{CLIP_NUM}_frames"
    os.makedirs(frames_dir, exist_ok=True)
    for old in Path(frames_dir).glob("*.jpg"):
        old.unlink()

    subprocess.run([
        "ffmpeg", "-y", "-i", str(clip_file),
        "-vf", f"fps={EXTRACT_FPS}", "-q:v", "2",
        os.path.join(frames_dir, "frame_%04d.jpg")
    ], capture_output=True, text=True, timeout=60)

    # Build filmstrip
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    n = len(frames)

    cols = 4
    rows_count = (n + cols - 1) // cols
    tw = 480
    sample = Image.open(frames[0])
    fw, fh = sample.size
    scale = tw / fw
    th = int(fh * scale)
    pad = 6

    sheet_w = cols * tw + (cols - 1) * pad
    sheet_h = rows_count * th + (rows_count - 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))

    try:
        font_label = ImageFont.truetype("arial.ttf", 16)
        font_grid = ImageFont.truetype("arial.ttf", 11)
    except:
        font_label = ImageFont.load_default()
        font_grid = ImageFont.load_default()

    for i, fp in enumerate(frames):
        r_idx = i // cols
        c_idx = i % cols
        x0 = c_idx * (tw + pad)
        y0 = r_idx * (th + pad)

        img = Image.open(fp).resize((tw, th), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        src_frame = i * FRAMES_PER_EXTRACT
        time_s = src_frame / FPS_SRC

        # Grid lines at 10% intervals
        for pct in range(0, 101, 10):
            gx = int(pct / 100.0 * tw)
            line_color = (255, 255, 0) if pct % 50 == 0 else (180, 180, 0)
            width = 2 if pct % 50 == 0 else 1
            draw.line([(gx, 0), (gx, th)], fill=line_color, width=width)
            if pct % 20 == 0:
                draw.text((gx + 2, th - 14), f"{pct}%", fill=(255, 255, 0), font=font_grid)

        label = f"f{src_frame}  {time_s:.1f}s"
        bbox = draw.textbbox((0, 0), label, font=font_label)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([(4, 2), (8 + lw, 6 + lh)], fill=(0, 0, 0, 180))
        draw.text((6, 3), label, fill=(255, 255, 255), font=font_label)

        sheet.paste(img, (x0, y0))

    output_png = rf"D:\Projects\soccer-video\_tmp\filmstrip_{CLIP_NUM}.png"
    sheet.save(output_png, quality=92)

    # Build review CSV - just frame/time/camera_x_pct for user to fill
    csv_rows = []
    for i in range(n):
        src_frame = i * FRAMES_PER_EXTRACT
        time_s = round(src_frame / FPS_SRC, 1)
        csv_rows.append({
            "frame": src_frame,
            "time_s": time_s,
            "camera_x_pct": "",
            "notes": ""
        })

    output_csv = rf"D:\Projects\soccer-video\_tmp\review_{CLIP_NUM}.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time_s", "camera_x_pct", "notes"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Copy both to Desktop
    shutil.copy2(output_png, rf"C:\Users\scott\Desktop\filmstrip_{CLIP_NUM}.png")
    shutil.copy2(output_csv, rf"C:\Users\scott\Desktop\review_{CLIP_NUM}.csv")

    size_mb = os.path.getsize(output_png) / (1024 * 1024)
    print(f"SUCCESS")
    print(f"Clip: {clip_file.name}")
    print(f"Duration: {duration}s, {total_frames} frames")
    print(f"Filmstrip: {n} frames at {EXTRACT_FPS}fps, {size_mb:.1f} MB")
    print(f"CSV: {len(csv_rows)} rows (camera_x_pct column for you to fill)")
    print(f"Both copied to Desktop")

except Exception as e:
    print(f"EXCEPTION: {e}")
    traceback.print_exc()
