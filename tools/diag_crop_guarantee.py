import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("--telemetry", required=True)
args = ap.parse_args()

bad = 0
n = 0
for line in open(args.telemetry, "r", encoding="utf-8"):
    d = json.loads(line)
    n += 1
    crop = d.get("crop")
    ball = d.get("ball")
    if not crop or not ball:
        continue
    x0, y0, w, h = crop
    bx, by = ball
    inside = (x0 <= bx <= x0 + w) and (y0 <= by <= y0 + h)
    if not inside:
        bad += 1
print(f"frames: {n}  outside_crop: {bad}")
sys.exit(0 if bad == 0 else 2)
