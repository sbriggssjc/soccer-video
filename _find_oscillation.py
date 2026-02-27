"""Find where the v10 render oscillates between two positions.

Look at the camera cx in the diag CSV and find rapid back-and-forth jumps.
Then correlate with YOLO detection positions to identify which detections
are pulling the crop to the wrong position.
"""
import csv
import json
import numpy as np
from pathlib import Path

REPO = Path(r"D:\Projects\soccer-video")
DIAG = REPO / "out" / "portrait_reels" / "2026-02-23__TSC_vs_NEOFC" / "002__consensus_v10.diag.csv"

with open(DIAG) as f:
    rows = list(csv.DictReader(f))

# Extract camera cx and ball position timeline
cam_cx = np.array([float(r["cam_cx"]) for r in rows])
ball_x = np.array([float(r["ball_x"]) for r in rows])
sources = [r["source"] for r in rows]

# Find oscillations: frames where cam_cx reverses direction rapidly
velocity = np.diff(cam_cx)
reversals = []
for i in range(1, len(velocity)):
    if velocity[i] * velocity[i-1] < 0:  # sign change
        magnitude = abs(velocity[i]) + abs(velocity[i-1])
        if magnitude > 20:  # significant reversal
            reversals.append((i, magnitude, velocity[i-1], velocity[i]))

print(f"=== Camera position reversals (>20px/f swing) ===")
print(f"Total significant reversals: {len(reversals)}")
for frame, mag, v_before, v_after in reversals[:20]:
    src = sources[frame]
    bx = ball_x[frame]
    cx = cam_cx[frame]
    print(f"  f{frame}: cam_cx={cx:.0f}, ball_x={bx:.0f}, "
          f"vel {v_before:+.0f} -> {v_after:+.0f} (swing={mag:.0f}), src={src}")

# Find the biggest jumps in ball_x (the underlying data causing oscillation)
print(f"\n=== Ball position jumps (>100px) ===")
ball_jumps = []
for i in range(1, len(ball_x)):
    jump = ball_x[i] - ball_x[i-1]
    if abs(jump) > 100:
        ball_jumps.append((i, jump, sources[i], sources[i-1]))
        
for frame, jump, src, prev_src in ball_jumps[:20]:
    print(f"  f{frame}: ball_x jumps {jump:+.0f}px ({prev_src} -> {src}), "
          f"ball_x={ball_x[frame]:.0f}")

# Show source transitions where ball jumps
print(f"\n=== Source transitions causing jumps ===")
for i in range(1, len(ball_x)):
    if sources[i] != sources[i-1]:
        jump = ball_x[i] - ball_x[i-1]
        if abs(jump) > 50:
            print(f"  f{i}: {sources[i-1]}->{sources[i]}, "
                  f"ball_x={ball_x[i-1]:.0f}->{ball_x[i]:.0f} (jump={jump:+.0f})")

# Load YOLO detections for comparison
print(f"\n=== YOLO detection positions in consensus ===")
with open(REPO / "out" / "telemetry" / "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.consensus.jsonl") as f:
    yolo_dets = {}
    for line in f:
        d = json.loads(line)
        yolo_dets[d["frame"]] = (d["cx"], d["cy"], d["conf"])

# Show YOLO vs interpolated position at each YOLO frame
print(f"\nYOLO detection vs surrounding trajectory:")
yolo_frames = sorted(yolo_dets.keys())
for i, f in enumerate(yolo_frames):
    cx, cy, conf = yolo_dets[f]
    diag_ball_x = ball_x[f]
    # Check if this YOLO position is far from the interpolated trajectory
    # Look at detections before and after
    prev_f = yolo_frames[i-1] if i > 0 else None
    next_f = yolo_frames[i+1] if i < len(yolo_frames)-1 else None
    
    jump_from_prev = ""
    if prev_f and f - prev_f <= 20:
        prev_cx = yolo_dets[prev_f][0]
        d = cx - prev_cx
        if abs(d) > 200:
            jump_from_prev = f" *** JUMP from f{prev_f}: {d:+.0f}px"
    
    if jump_from_prev:
        print(f"  f{f}: YOLO cx={cx:.0f}, cy={cy:.0f}, conf={conf:.2f}{jump_from_prev}")
