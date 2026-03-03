"""Generate blank review CSVs for a batch of clips.
Probes each source clip for frame count, samples every 15 frames (2fps).
User fills in camera_x_pct column.
"""
import os, csv, shutil
import cv2

GAME = "2026-03-01__TSC_vs_North_OKC"
CLIP_NUMS = ["001","002","003","004","005","006","007","008","009","010"]

os.chdir(r"D:\Projects\soccer-video")
FPS = 30
SAMPLE_INTERVAL = 15  # every 15 frames = 2fps

from pathlib import Path
clips_dir = Path(f"out/atomic_clips/{GAME}")
tmp_dir = Path("_tmp")
tmp_dir.mkdir(exist_ok=True)

for clip_num in CLIP_NUMS:
    clip_file = next(clips_dir.glob(f"{clip_num}__*.mp4"))
    
    # Probe frame count
    cap = cv2.VideoCapture(str(clip_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Generate rows every SAMPLE_INTERVAL frames
    rows = []
    for f in range(0, total_frames, SAMPLE_INTERVAL):
        rows.append({
            "frame": f,
            "time_s": round(f / FPS, 1),
            "camera_x_pct": "",
            "notes": ""
        })
    
    # Write CSV
    csv_path = tmp_dir / f"review_{clip_num}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", "time_s", "camera_x_pct", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    
    # Copy to Desktop
    desktop = rf"C:\Users\scott\Desktop\review_{clip_num}.csv"
    shutil.copy2(str(csv_path), desktop)
    
    print(f"{clip_num}: {total_frames} frames, {len(rows)} sample rows -> Desktop")

print("\nDone! CSVs on Desktop ready for review.")
