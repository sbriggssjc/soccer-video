"""Diagnose frame count vs filename duration for all Greenwood clips."""
import os, re, csv
import cv2
from pathlib import Path

GAME = "2026-02-23__TSC_vs_Greenwood"
clips_dir = Path(f"D:/Projects/soccer-video/out/atomic_clips/{GAME}")

print(f"{'Clip':>4} {'Fname_dur':>10} {'Actual_dur':>10} {'Frames':>7} {'Ratio':>6}  Filename")
print("-" * 90)

for clip_file in sorted(clips_dir.glob("*.mp4")):
    stem = clip_file.stem
    m = re.search(r"t([\d.]+)-t([\d.]+)", stem)
    if not m:
        continue
    fname_dur = float(m.group(2)) - float(m.group(1))
    
    cap = cv2.VideoCapture(str(clip_file))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    actual_dur = frames / fps if fps > 0 else 0
    ratio = actual_dur / fname_dur if fname_dur > 0 else 0
    
    clip_num = stem[:3]
    print(f"{clip_num:>4} {fname_dur:>10.2f} {actual_dur:>10.2f} {frames:>7} {ratio:>6.2f}  {clip_file.name}")
