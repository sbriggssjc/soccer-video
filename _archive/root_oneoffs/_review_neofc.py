"""Generate filmstrip for NEOFC clip 004 for review.
Extract frames at 2fps, tile into labeled contact sheet with grid.
"""
import os, sys, traceback, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CLIP_NUM = "010"
GAME = "2026-02-23__TSC_vs_NEOFC"
RESULT = rf"D:\Projects\soccer-video\_tmp\review_{CLIP_NUM}_result.txt"
os.makedirs(r"D:\Projects\soccer-video\_tmp", exist_ok=True)

# Find the clip
clips_dir = Path(rf"D:\Projects\soccer-video\out\atomic_clips\{GAME}")
clip_file = None
for f in clips_dir.glob(f"{CLIP_NUM}__*.mp4"):
    clip_file = f
    break

# Also find the FINAL portrait
reels_dir = Path(rf"D:\Projects\soccer-video\out\portrait_reels\{GAME}")
final_file = None
for f in reels_dir.glob(f"{CLIP_NUM}__*__FINAL.mp4"):
    final_file = f
    break

try:
    if not clip_file:
        raise FileNotFoundError(f"No clip found for {CLIP_NUM}")

    # Extract frames at 2fps
    frames_dir = rf"D:\Projects\soccer-video\_tmp\filmstrip_{CLIP_NUM}_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Clear old frames
    for old in Path(frames_dir).glob("*.jpg"):
        old.unlink()

    cmd = [
        "ffmpeg", "-y", "-i", str(clip_file),
        "-vf", "fps=2", "-q:v", "2",
        os.path.join(frames_dir, "frame_%04d.jpg")
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    # Get clip info
    probe = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(clip_file)], capture_output=True, text=True, timeout=30)

    # Tile frames into labeled contact sheet
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    n = len(frames)
    if n == 0:
        raise FileNotFoundError("No frames extracted")

    FPS_SRC = 30
    EXTRACT_FPS = 2
    FRAMES_PER_EXTRACT = FPS_SRC // EXTRACT_FPS

    cols = 4
    rows = (n + cols - 1) // cols
    tw = 480

    sample = Image.open(frames[0])
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

        # Frame label
        label = f"f{src_frame}  {time_s:.1f}s"
        bbox = draw.textbbox((0, 0), label, font=font_label)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        draw.rectangle([(4, 2), (8 + lw, 6 + lh)], fill=(0, 0, 0, 180))
        draw.text((6, 3), label, fill=(255, 255, 255), font=font_label)

        sheet.paste(img, (x0, y0))

    output_png = rf"D:\Projects\soccer-video\_tmp\filmstrip_{CLIP_NUM}.png"
    sheet.save(output_png, quality=92)
    size_mb = os.path.getsize(output_png) / (1024 * 1024)

    # Copy to Desktop
    import shutil
    desktop_path = rf"C:\Users\scott\Desktop\filmstrip_{CLIP_NUM}.png"
    shutil.copy2(output_png, desktop_path)

    msg = f"SUCCESS\n"
    msg += f"Clip: {clip_file.name}\n"
    msg += f"FINAL: {final_file.name if final_file else 'NOT FOUND'}\n"
    msg += f"Frames: {n}, Grid: {cols}x{rows}\n"
    msg += f"Output: {output_png} ({size_mb:.1f} MB)\n"
    msg += f"Copied to: {desktop_path}\n"
    with open(RESULT, "w") as f:
        f.write(msg)

except Exception as e:
    with open(RESULT, "w") as f:
        f.write(f"EXCEPTION: {e}\n{traceback.format_exc()}")
