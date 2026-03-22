"""Generate filmstrip review images + blank CSVs for Feb 21 Greenwood clips."""
import os, csv, shutil, math
import cv2
import numpy as np
from pathlib import Path

GAME = "2026-02-21__TSC_vs_Greenwood"
PREFIX = "gw21_"
CLIP_NUMS = [f"{i:03d}" for i in range(1, 21)]  # 001-020

os.chdir(r"D:\Projects\soccer-video")
FPS = 30
SAMPLE_INTERVAL = 15

COLS = 4
THUMB_W = 480
THUMB_H = 270
PAD = 4
LABEL_H = 24
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
    
    sample_indices = list(range(0, total_frames, SAMPLE_INTERVAL))
    n_samples = len(sample_indices)
    rows_needed = math.ceil(n_samples / COLS)
    
    canvas_w = COLS * (THUMB_W + PAD) + PAD
    canvas_h = rows_needed * (CELL_H + PAD) + PAD + 40
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)
    
    title = f"Clip {clip_num} | {clip_file.name} | {total_frames} frames | {total_frames/FPS:.1f}s"
    cv2.putText(canvas, title, (PAD, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    csv_rows = []
    for i, frame_idx in enumerate(sample_indices):
        col = i % COLS
        row = i // COLS
        x0 = PAD + col * (THUMB_W + PAD)
        y0 = 40 + PAD + row * (CELL_H + PAD)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
        for pct in [0, 25, 50, 75, 100]:
            gx = int(THUMB_W * pct / 100)
            color = (0, 200, 200) if pct == 50 else (100, 100, 100)
            thickness = 2 if pct == 50 else 1
            cv2.line(thumb, (gx, 0), (gx, THUMB_H), color, thickness)
            cv2.putText(thumb, f"{pct}%", (gx + 2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        canvas[y0 + LABEL_H:y0 + CELL_H, x0:x0 + THUMB_W] = thumb
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
    
    filmstrip_path = film_dir / f"filmstrip_{clip_num}.jpg"
    cv2.imwrite(str(filmstrip_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    csv_path = tmp_dir / f"review_{PREFIX}{clip_num}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", "time_s", "camera_x_pct", "notes"])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    desktop_csv = rf"C:\Users\scott\Desktop\review_{PREFIX}{clip_num}.csv"
    shutil.copy2(str(csv_path), desktop_csv)
    desktop_film = rf"C:\Users\scott\Desktop\filmstrip_{PREFIX}{clip_num}.jpg"
    shutil.copy2(str(filmstrip_path), desktop_film)
    
    print(f"{clip_num}: {total_frames} frames, {n_samples} samples, "
          f"grid {COLS}x{rows_needed} -> Desktop")

print(f"\nDone! Feb 21 Greenwood filmstrips + CSVs on Desktop (prefix: {PREFIX}).")
