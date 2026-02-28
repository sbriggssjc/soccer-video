"""
Extract key frames and draw our optical flow ball position as a marker.
Saves annotated frames so we can verify visually.
"""
import cv2
import json
import os
import numpy as np

os.chdir(r'D:\Projects\soccer-video')

CLIP = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'
VIDEO = f'out/atomic_clips/2026-02-23__TSC_vs_NEOFC/{CLIP}.mp4'
OUT_DIR = '_optflow_track/verify_frames'
os.makedirs(OUT_DIR, exist_ok=True)

# Load optical flow v3
optflow = {}
with open('_optflow_track/optflow_ball_path_v3.jsonl') as f:
    for line in f:
        d = json.loads(line)
        optflow[d['frame']] = d

# Key frames to check
check_frames = [
    0, 30, 60, 90, 120, 135, 140, 150, 170, 180,
    200, 220, 250, 270, 280, 290, 300, 310, 314,
    320, 330, 340, 350, 360, 370, 380, 390, 400,
    410, 416, 417, 420, 425, 430, 440, 450, 460,
    470, 480, 490, 495
]

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for target in check_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if not ret:
        continue
    
    if target in optflow:
        o = optflow[target]
        cx, cy = int(o['cx']), int(o['cy'])
        # Draw crosshair at our predicted ball position
        cv2.circle(frame, (cx, cy), 25, (0, 0, 255), 3)  # Red circle
        cv2.line(frame, (cx-35, cy), (cx+35, cy), (0, 0, 255), 2)
        cv2.line(frame, (cx, cy-35), (cx, cy+35), (0, 0, 255), 2)
        # Draw portrait crop boundaries (cam at ball position, 608px wide)
        cam_cx = min(max(cx, 304), 1616)
        crop_left = cam_cx - 304
        crop_right = cam_cx + 304
        cv2.line(frame, (crop_left, 0), (crop_left, 1080), (0, 255, 0), 2)
        cv2.line(frame, (crop_right, 0), (crop_right, 1080), (0, 255, 0), 2)
        
        label = f"f{target} ball=({cx},{cy}) src={o.get('source','?')}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    outpath = os.path.join(OUT_DIR, f'f{target:04d}.png')
    cv2.imwrite(outpath, frame)

cap.release()
print(f"Saved {len(check_frames)} annotated frames to {OUT_DIR}/")
print("Red circle = optical flow ball position")
print("Green lines = ideal portrait crop boundaries")
