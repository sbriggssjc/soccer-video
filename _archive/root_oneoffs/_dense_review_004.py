"""Build dense review CSV for clip 004 with real YOLO positions.
- Standard 2fps sampling for stable sections
- Frame-by-frame in transition zones (f270-f520)
- ball_x_pct from actual YOLO detections (interpolated)
- camera_x_pct from user's current anchors
"""
import os, csv, json
import numpy as np

RESULT = r"D:\Projects\soccer-video\_tmp\dense_review_004_result.txt"
os.makedirs(os.path.dirname(RESULT), exist_ok=True)

FRAME_W = 1920
FPS = 30

# Read original YOLO detections
yolo_path = r"D:\Projects\soccer-video\out\telemetry\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00.yolo_ball.yolov8x.jsonl.bak"
if not os.path.exists(yolo_path):
    yolo_path = yolo_path.replace(".bak", "")

with open(yolo_path, "r") as f:
    yolo_rows = [json.loads(l) for l in f if l.strip()]

# Build YOLO lookup and interpolated trajectory
yolo_frames = [r["frame"] for r in yolo_rows]
yolo_cx = [r["cx"] for r in yolo_rows]

# Read current patched ball.jsonl (user's anchors interpolated)
ball_path = r"D:\Projects\soccer-video\out\telemetry\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00.ball.jsonl"
with open(ball_path, "r") as f:
    ball_rows = [json.loads(l) for l in f if l.strip()]
total_frames = len(ball_rows)

# Interpolate YOLO across all frames for reference
yolo_interp = np.interp(np.arange(total_frames), yolo_frames, yolo_cx)

# Current camera path (from patched ball.jsonl = user's anchors)
cam_cx = [r["cx"] for r in ball_rows]

# Build frame list: sparse outside transition, dense inside
frames_to_include = set()

# Standard 2fps sampling
for f in range(0, total_frames, 15):
    frames_to_include.add(f)

# Dense sampling in transition zones (every 5 frames f270-f520)
for f in range(270, min(520, total_frames), 5):
    frames_to_include.add(f)

# Always include last frame
frames_to_include.add(total_frames - 1)

# Sort
frame_list = sorted(frames_to_include)

# Write CSV
output_csv = r"C:\Users\scott\Desktop\review_004.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time_s", "ball_x_pct", "camera_x_pct", "notes"])
    for fr in frame_list:
        time_s = round(fr / FPS, 1)
        ball_pct = round(yolo_interp[fr] / FRAME_W * 100, 0)
        cam_pct = round(cam_cx[fr] / FRAME_W * 100, 0)
        # Flag dense section
        note = "DENSE" if 270 <= fr <= 520 and fr % 15 != 0 else ""
        writer.writerow([fr, time_s, int(ball_pct), int(cam_pct), note])

msg = f"SUCCESS\n"
msg += f"Total frames in clip: {total_frames}\n"
msg += f"Rows in CSV: {len(frame_list)}\n"
msg += f"  Standard (every 15): {sum(1 for f in frame_list if f % 15 == 0)}\n"
msg += f"  Dense (f270-f520): {sum(1 for f in frame_list if 270 <= f <= 520 and f % 15 != 0)}\n"
msg += f"Output: {output_csv}\n"
msg += f"\nball_x_pct = actual YOLO detection (interpolated)\n"
msg += f"camera_x_pct = your current anchors (edit this)\n"

with open(RESULT, "w") as f:
    f.write(msg)
