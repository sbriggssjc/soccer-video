"""Extract video frames at each YOLO detection and draw the detection point.

Creates a contact sheet + individual annotated frames so you can visually
verify which YOLO detections are the real ball vs false positives.

Output:
  _yolo_verify/frame_NNN_confX.XX.jpg  - individual annotated frames
  _yolo_verify/contact_sheet.jpg       - all detections on one image
"""
import json, os, subprocess, sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageDraw, ImageFont

# --- Config ---
VIDEO = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
YOLO_FILE = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl"
OUT_DIR = r"D:\Projects\soccer-video\_yolo_verify"
FPS = 30.0  # source video fps

os.makedirs(OUT_DIR, exist_ok=True)

# Load YOLO detections
detections = []
with open(YOLO_FILE, "r") as f:
    for line in f:
        d = json.loads(line.strip())
        detections.append(d)

detections.sort(key=lambda d: d.get("frame", d.get("frame_idx", 0)))
print(f"Loaded {len(detections)} YOLO detections", flush=True)

# Extract frames using ffmpeg (one per detection)
frames_needed = set()
for d in detections:
    frames_needed.add(int(d.get("frame", d.get("frame_idx", 0))))

# Extract all unique frames at once using select filter
frame_list = sorted(frames_needed)
print(f"Extracting {len(frame_list)} unique frames...", flush=True)

# Extract each frame individually (most reliable)
tmp_dir = os.path.join(OUT_DIR, "_tmp")
os.makedirs(tmp_dir, exist_ok=True)

for fidx in frame_list:
    timestamp = fidx / FPS
    out_path = os.path.join(tmp_dir, f"f{fidx:04d}.png")
    if os.path.exists(out_path):
        continue
    cmd = [
        "ffmpeg", "-y", "-ss", f"{timestamp:.4f}",
        "-i", VIDEO,
        "-frames:v", "1",
        "-q:v", "2",
        out_path
    ]
    subprocess.run(cmd, capture_output=True)

print(f"Frames extracted to {tmp_dir}", flush=True)

# Annotate each frame with the YOLO detection point
CIRCLE_RADIUS = 25
CROSSHAIR_SIZE = 35

annotated_images = []

for d in detections:
    fidx = int(d.get("frame", d.get("frame_idx", 0)))
    conf = d.get("confidence", d.get("conf", 0))
    x = float(d.get("x", d.get("cx", 0)))
    y = float(d.get("y", d.get("cy", 0)))
    w = float(d.get("w", d.get("width", 0)))
    h = float(d.get("h", d.get("height", 0)))

    frame_path = os.path.join(tmp_dir, f"f{fidx:04d}.png")
    if not os.path.exists(frame_path):
        print(f"  WARNING: missing frame {fidx}", flush=True)
        continue

    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw bounding box if available
    if w > 0 and h > 0:
        x0 = x - w / 2
        y0_box = y - h / 2
        x1 = x + w / 2
        y1_box = y + h / 2
        draw.rectangle([x0, y0_box, x1, y1_box], outline="lime", width=3)

    # Draw crosshair at detection point
    color = "red" if conf < 0.25 else "lime"
    draw.ellipse(
        [x - CIRCLE_RADIUS, y - CIRCLE_RADIUS,
         x + CIRCLE_RADIUS, y + CIRCLE_RADIUS],
        outline=color, width=3
    )
    draw.line([x - CROSSHAIR_SIZE, y, x + CROSSHAIR_SIZE, y],
              fill=color, width=2)
    draw.line([x, y - CROSSHAIR_SIZE, x, y + CROSSHAIR_SIZE],
              fill=color, width=2)

    # Draw label with frame number and confidence
    label = f"f{fidx} conf={conf:.3f} ({x:.0f},{y:.0f})"
    # Background rectangle for text
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((10, 10), label, font=font)
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2],
                   fill="black")
    draw.text((10, 10), label, fill=color, font=font)

    # Save individual annotated frame
    out_name = f"f{fidx:04d}_conf{conf:.3f}.jpg"
    img.save(os.path.join(OUT_DIR, out_name), quality=90)
    annotated_images.append((fidx, conf, img))
    print(f"  f{fidx}: conf={conf:.3f} at ({x:.0f},{y:.0f}) -> {out_name}",
          flush=True)

# Create contact sheet - sorted by confidence (lowest first)
print(f"\nCreating contact sheet...", flush=True)
annotated_images.sort(key=lambda x: x[1])  # sort by confidence

THUMB_W = 480
THUMB_H = 270
COLS = 4
ROWS = (len(annotated_images) + COLS - 1) // COLS
PADDING = 4

sheet_w = COLS * (THUMB_W + PADDING) + PADDING
sheet_h = ROWS * (THUMB_H + PADDING) + PADDING
sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))

for idx, (fidx, conf, img) in enumerate(annotated_images):
    thumb = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    col = idx % COLS
    row = idx // COLS
    px = PADDING + col * (THUMB_W + PADDING)
    py = PADDING + row * (THUMB_H + PADDING)
    sheet.paste(thumb, (px, py))

sheet_path = os.path.join(OUT_DIR, "contact_sheet.jpg")
sheet.save(sheet_path, quality=92)
print(f"\nDone! Contact sheet: {sheet_path}", flush=True)
print(f"Individual frames: {OUT_DIR}", flush=True)
print(f"Total: {len(annotated_images)} annotated frames", flush=True)
print(f"  Red markers = conf < 0.25 (potential phantoms)", flush=True)
print(f"  Green markers = conf >= 0.25 (likely valid)", flush=True)
