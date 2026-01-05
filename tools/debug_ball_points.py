import json
from pathlib import Path
import cv2

# EDIT THESE TWO PATHS
VIDEO = Path(r"C:\Users\scott\soccer-video\out\atomic_clips\_quarantine\001__SHOT__t155.50-t166.40.mp4")
TEL   = Path(r"C:\Users\scott\soccer-video\out\telemetry\001__SHOT__t155.50-t166.40.ball.jsonl")

cap = cv2.VideoCapture(str(VIDEO))
ok, frame = cap.read()
if not ok:
    raise SystemExit("Could not read first frame")

h, w = frame.shape[:2]
print("Video size:", w, h)

rows = []
with TEL.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        rows.append(json.loads(line))

print("Telemetry keys example:", sorted(rows[0].keys()))
print("First 10 raw points (as-is):")
pts = []
for r in rows:
    # try common key combos
    for kx, ky in [("x","y"), ("bx","by"), ("cx","cy"), ("ball_x","ball_y"), ("ball_cx","ball_cy")]:
        if kx in r and ky in r:
            x, y = r[kx], r[ky]
            pts.append((float(x), float(y), kx, ky))
            break

for (x, y, kx, ky) in pts:
    print(f"  {kx},{ky} = {x:.2f},{y:.2f}")

# draw first 10 points directly onto the SOURCE frame
for (x, y, *_rest) in pts:
    xi, yi = int(round(x)), int(round(y))
    cv2.circle(frame, (xi, yi), 12, (255, 0, 0), 3)  # blue ring (BGR)
    cv2.circle(frame, (xi, yi), 2, (255, 0, 0), -1)

out = Path("out") / "ball_points_on_source_frame.png"
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), frame)
print("Wrote:", out)
