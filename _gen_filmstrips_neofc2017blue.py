"""Generate filmstrip review images + prepopulated review CSVs for
2026-03-21__TSC_vs_NEOFC_2017_Blue.

For each clip:
  1. Probes actual frame count and FPS
  2. Samples every SAMPLE_INTERVAL frames (~2fps)
  3. Creates a filmstrip grid image with:
     - Each sampled frame shown at reduced size
     - Frame number + time overlaid
     - Vertical percentage guidelines (0%, 25%, 50%, 75%, 100%)
     - Row index matching the CSV row number
  4. Reads YOLO ball telemetry to prepopulate ball_x_pct
  5. Reads the auto-render follow path to prepopulate camera_x_pct
  6. Writes review CSV to Desktop for user to correct camera_x_pct

Usage: python _gen_filmstrips_neofc2017blue.py
"""
import os, csv, shutil, math, json, glob
import cv2
import numpy as np
from pathlib import Path

# ===== CONFIGURE THESE =====
GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
PREFIX = "neofc2017_"
# ============================

os.chdir(r"D:\Projects\soccer-video")

SAMPLE_INTERVAL = 15  # every 15 frames = ~2fps at 30fps source

# Grid layout
COLS = 4
THUMB_W = 480
THUMB_H = 270
PAD = 4
LABEL_H = 24
CELL_H = THUMB_H + LABEL_H

clips_dir = Path(f"out/atomic_clips/{GAME}")
telemetry_dir = Path("out/telemetry")
tmp_dir = Path("_tmp")
tmp_dir.mkdir(exist_ok=True)
film_dir = Path(f"_tmp/filmstrips/{GAME}")
film_dir.mkdir(parents=True, exist_ok=True)

# Find all clips and sort numerically
clip_files = sorted(clips_dir.glob("*.mp4"),
                    key=lambda p: int(p.stem.split("__")[0]))


def load_yolo_ball(stem):
    """Load YOLO ball detections from telemetry jsonl.
    Returns dict: frame_idx -> ball_x (pixels)."""
    candidates = [
        telemetry_dir / f"{stem}.yolo_ball.jsonl",
        telemetry_dir / f"{stem}.ball.jsonl",
    ]
    for path in candidates:
        if path.exists():
            ball_map = {}
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    frame = row.get("frame", row.get("frame_idx", -1))
                    # YOLO ball: cx is center x of bounding box
                    cx = row.get("cx", row.get("ball_cx", row.get("x", None)))
                    if cx is not None and frame >= 0:
                        ball_map[int(frame)] = float(cx)
            if ball_map:
                return ball_map
    return {}


def load_follow_path(stem):
    """Load the smooth follow path (camera center) from telemetry.
    Returns dict: frame_idx -> cam_cx (pixels)."""
    candidates = [
        telemetry_dir / f"{stem}.ball.follow__smooth.jsonl",
        telemetry_dir / f"{stem}.ball.follow.jsonl",
    ]
    for path in candidates:
        if path.exists():
            cam_map = {}
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    frame = row.get("frame", row.get("frame_idx", -1))
                    cx = row.get("cx", row.get("cam_cx", None))
                    if cx is not None and frame >= 0:
                        cam_map[int(frame)] = float(cx)
            if cam_map:
                return cam_map
    return {}


for clip_file in clip_files:
    stem = clip_file.stem
    clip_num = stem.split("__")[0]

    cap = cv2.VideoCapture(str(clip_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if actual_fps <= 0:
        actual_fps = 30.0

    # Load telemetry
    ball_map = load_yolo_ball(stem)
    cam_map = load_follow_path(stem)

    # Sample frames
    sample_indices = list(range(0, total_frames, SAMPLE_INTERVAL))
    n_samples = len(sample_indices)
    rows_needed = math.ceil(n_samples / COLS)

    # Create canvas
    canvas_w = COLS * (THUMB_W + PAD) + PAD
    canvas_h = rows_needed * (CELL_H + PAD) + PAD + 40
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)

    # Title bar
    title = f"Clip {clip_num} | {clip_file.name} | {total_frames} frames | {total_frames/actual_fps:.1f}s | {actual_fps:.0f}fps"
    cv2.putText(canvas, title, (PAD, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    csv_rows = []

    for i, frame_idx in enumerate(sample_indices):
        col = i % COLS
        row = i // COLS

        x0 = PAD + col * (THUMB_W + PAD)
        y0 = 40 + PAD + row * (CELL_H + PAD)

        # Seek and read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize to thumbnail
        thumb = cv2.resize(frame, (THUMB_W, THUMB_H))

        # Draw percentage guidelines
        for pct in [0, 25, 50, 75, 100]:
            gx = int(THUMB_W * pct / 100)
            color = (0, 200, 200) if pct == 50 else (100, 100, 100)
            thickness = 2 if pct == 50 else 1
            cv2.line(thumb, (gx, 0), (gx, THUMB_H), color, thickness)
            cv2.putText(thumb, f"{pct}%", (gx + 2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Draw ball position marker (red dot) if we have YOLO data
        ball_x = ball_map.get(frame_idx)
        if ball_x is not None:
            ball_pct = ball_x / frame_w
            bx = int(THUMB_W * ball_pct)
            bx = max(0, min(THUMB_W - 1, bx))
            cv2.circle(thumb, (bx, THUMB_H // 2), 6, (0, 0, 255), -1)
            cv2.circle(thumb, (bx, THUMB_H // 2), 6, (255, 255, 255), 1)

        # Draw camera center marker (green line) if we have follow data
        cam_x = cam_map.get(frame_idx)
        if cam_x is not None:
            cam_pct = cam_x / frame_w
            cx_px = int(THUMB_W * cam_pct)
            cx_px = max(0, min(THUMB_W - 1, cx_px))
            cv2.line(thumb, (cx_px, 0), (cx_px, THUMB_H), (0, 255, 0), 2)

        # Place thumb on canvas
        canvas[y0 + LABEL_H:y0 + CELL_H, x0:x0 + THUMB_W] = thumb

        # Label above thumb
        time_s = round(frame_idx / actual_fps, 1)
        label = f"Row {i}: frame {frame_idx} | t={time_s}s"
        cv2.putText(canvas, label, (x0 + 4, y0 + LABEL_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # CSV row with prepopulated ball_x_pct and camera_x_pct
        ball_pct_val = ""
        if ball_x is not None:
            ball_pct_val = int(round(ball_x / frame_w * 100))
            ball_pct_val = max(0, min(100, ball_pct_val))

        cam_pct_val = ""
        if cam_x is not None:
            cam_pct_val = int(round(cam_x / frame_w * 100))
            cam_pct_val = max(0, min(100, cam_pct_val))

        csv_rows.append({
            "frame": frame_idx,
            "time_s": time_s,
            "ball_x_pct": ball_pct_val,
            "camera_x_pct": cam_pct_val,
            "notes": ""
        })

    cap.release()

    # Save filmstrip
    filmstrip_path = film_dir / f"filmstrip_{clip_num}.jpg"
    cv2.imwrite(str(filmstrip_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Write CSV
    csv_path = tmp_dir / f"review_{clip_num}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", "time_s", "ball_x_pct", "camera_x_pct", "notes"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Copy to Desktop
    desktop_csv = rf"C:\Users\scott\Desktop\review_{PREFIX}{clip_num}.csv"
    shutil.copy2(str(csv_path), desktop_csv)

    desktop_film = rf"C:\Users\scott\Desktop\filmstrip_{PREFIX}{clip_num}.jpg"
    shutil.copy2(str(filmstrip_path), desktop_film)

    ball_count = sum(1 for r in csv_rows if r["ball_x_pct"] != "")
    print(f"  {clip_num}: {total_frames} frames, {n_samples} samples, "
          f"ball data: {ball_count}/{n_samples} rows -> Desktop")

print("\n" + "=" * 60)
print("DONE! Filmstrips + review CSVs on Desktop.")
print("=" * 60)
print()
print("Legend on filmstrips:")
print("  RED DOT   = YOLO ball position (what the detector saw)")
print("  GREEN LINE = current camera center (where the auto-render pointed)")
print("  CYAN 50%  = frame center")
print()
print("In each review CSV:")
print("  ball_x_pct    = detected ball position (0-100%) -- reference only")
print("  camera_x_pct  = camera center (0-100%) -- EDIT THIS to correct framing")
print("  Leave camera_x_pct blank for rows where auto is fine")
print()
print("After editing, re-render with corrected positions using:")
print("  python tools\\render_follow_unified.py --preset cinematic \\")
print("    --in <clip.mp4> --portrait 1080x1920 --plan <review_XX.csv>")
