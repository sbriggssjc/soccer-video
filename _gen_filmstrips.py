"""Generate filmstrip review images + blank CSVs for a batch of clips.

For each clip:
  1. Probes actual frame count
  2. Samples every SAMPLE_INTERVAL frames (2fps at 30fps source)
  3. Creates a filmstrip grid image with:
     - Each sampled frame shown at reduced size
     - Frame number + time overlaid
     - Vertical percentage guidelines (0%, 25%, 50%, 75%, 100%)
     - Row index matching the CSV row number
  4. Writes blank CSV to Desktop for user to fill in camera_x_pct

Usage: Edit GAME and CLIP_NUMS, then run.
"""
import os, csv, shutil, math
import cv2
import numpy as np
from pathlib import Path

# ===== CONFIGURE THESE =====
GAME = "2026-03-01__TSC_vs_OK_Celtic"
PREFIX = "celtic_"  # prefix for Desktop files to avoid collision ("" for none)
CLIP_NUMS = ["001","002","003","004","005","006","007","008","009","010"]
# ============================

os.chdir(r"D:\Projects\soccer-video")
FPS = 30
SAMPLE_INTERVAL = 15  # every 15 frames = 2fps

# Grid layout
COLS = 4
THUMB_W = 480
THUMB_H = 270
PAD = 4
LABEL_H = 24  # height for text label above each thumb
CELL_H = THUMB_H + LABEL_H

clips_dir = Path(f"out/atomic_clips/{GAME}")
tmp_dir = Path("_tmp")
tmp_dir.mkdir(exist_ok=True)
film_dir = Path(f"_tmp/filmstrips/{GAME}")
film_dir.mkdir(parents=True, exist_ok=True)

for clip_num in CLIP_NUMS:
    clip_file = next(clips_dir.glob(f"{clip_num}__*.mp4"))
    
    cap = cv2.VideoCapture(str(clip_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample frames
    sample_indices = list(range(0, total_frames, SAMPLE_INTERVAL))
    n_samples = len(sample_indices)
    rows_needed = math.ceil(n_samples / COLS)
    
    # Create canvas
    canvas_w = COLS * (THUMB_W + PAD) + PAD
    canvas_h = rows_needed * (CELL_H + PAD) + PAD + 40  # +40 for title
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)  # dark gray bg
    
    # Title bar
    title = f"Clip {clip_num} | {clip_file.name} | {total_frames} frames | {total_frames/FPS:.1f}s"
    cv2.putText(canvas, title, (PAD, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
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
            # Label at top
            cv2.putText(thumb, f"{pct}%", (gx + 2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # Place thumb on canvas
        canvas[y0 + LABEL_H:y0 + CELL_H, x0:x0 + THUMB_W] = thumb
        
        # Label above thumb
        time_s = round(frame_idx / FPS, 1)
        label = f"Row {i}: frame {frame_idx} | t={time_s}s"
        cv2.putText(canvas, label, (x0 + 4, y0 + LABEL_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        csv_rows.append({
            "frame": frame_idx,
            "time_s": time_s,
            "camera_x_pct": "",
            "notes": ""
        })
    
    cap.release()
    
    # Save filmstrip
    filmstrip_path = film_dir / f"filmstrip_{clip_num}.jpg"
    cv2.imwrite(str(filmstrip_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # Write CSV
    csv_path = tmp_dir / f"review_{clip_num}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", "time_s", "camera_x_pct", "notes"])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # Copy CSV + filmstrip to Desktop
    desktop_csv = rf"C:\Users\scott\Desktop\review_{PREFIX}{clip_num}.csv"
    shutil.copy2(str(csv_path), desktop_csv)
    
    desktop_film = rf"C:\Users\scott\Desktop\filmstrip_{PREFIX}{clip_num}.jpg"
    shutil.copy2(str(filmstrip_path), desktop_film)
    
    print(f"{clip_num}: {total_frames} frames, {n_samples} samples, "
          f"grid {COLS}x{rows_needed} -> Desktop (CSV + filmstrip)")

print("\nDone! Filmstrips + CSVs on Desktop.")
print("Instructions: Open each filmstrip, note where the ball/action is")
print("as a percentage (0-100) using the guidelines, fill in camera_x_pct.")
