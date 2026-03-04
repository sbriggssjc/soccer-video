"""Check FPS, resolution, and zoom of Feb 21 Greenwood FINAL renders."""
import cv2
from pathlib import Path

REEL_DIR = Path(r"D:\Projects\soccer-video\out\portrait_reels\2026-02-21__TSC_vs_Greenwood")

for f in sorted(REEL_DIR.glob("*__portrait__FINAL.mp4")):
    cap = cv2.VideoCapture(str(f))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = frames / fps if fps > 0 else 0
    cap.release()
    clip_num = f.name[:3]
    print(f"{clip_num}: {w}x{h} @ {fps:.1f}fps, {frames} frames, {dur:.1f}s, {f.stat().st_size/(1024*1024):.1f}MB")
