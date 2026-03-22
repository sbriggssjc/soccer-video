"""Analyze clip 004 ball trajectory to find exact re-entry frame.
Read original YOLO detections and current diagnostic to understand the gap.
"""
import os, csv, json

RESULT = r"D:\Projects\soccer-video\_tmp\analyze_004_result.txt"
os.makedirs(os.path.dirname(RESULT), exist_ok=True)

# Read original YOLO detections (backed up)
yolo_bak = r"D:\Projects\soccer-video\out\telemetry\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00.yolo_ball.yolov8x.jsonl.bak"
# If no .bak, try the original yolov8x file
if not os.path.exists(yolo_bak):
    yolo_bak = r"D:\Projects\soccer-video\out\telemetry\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00.yolo_ball.yolov8x.jsonl"

# Read current diagnostic
diag = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00__portrait.diag.csv"

# Read original ball.jsonl backup
ball_bak = r"D:\Projects\soccer-video\out\telemetry\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00.ball.jsonl.bak"

lines = []

# Original YOLO detections
if os.path.exists(yolo_bak):
    with open(yolo_bak, "r") as f:
        yolo_rows = [json.loads(l) for l in f if l.strip()]
    lines.append(f"Original YOLO detections: {len(yolo_rows)} entries")
    lines.append("Frame | cx | cy | conf")
    for r in yolo_rows:
        fr = r.get("frame", "?")
        cx = r.get("cx", "?")
        cy = r.get("cy", "?")
        conf = r.get("conf", "?")
        pct = round(float(cx) / 1920 * 100, 1) if cx != "?" else "?"
        lines.append(f"  f{fr}: cx={cx} ({pct}%), cy={cy}, conf={conf}")
else:
    lines.append(f"No YOLO backup found at {yolo_bak}")

# Original ball.jsonl backup - focus on f300-f540 (the problematic zone)
lines.append("")
if os.path.exists(ball_bak):
    with open(ball_bak, "r") as f:
        ball_rows = [json.loads(l) for l in f if l.strip()]
    lines.append(f"Original ball.jsonl: {len(ball_rows)} entries")
    lines.append("Frames 250-546 (ball exit/re-entry zone):")
    for r in ball_rows:
        fr = r["frame"]
        if 250 <= fr <= 546 and fr % 5 == 0:
            cx = r["cx"]
            pct = round(cx / 1920 * 100, 1)
            lines.append(f"  f{fr}: cx={cx:.0f} ({pct}%)")
else:
    lines.append(f"No ball.jsonl backup found")

# Diagnostic - focus on the transition zone
lines.append("")
lines.append("Current diagnostic (frames 300-540, every 5 frames):")
with open(diag, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fr = int(row["frame"])
        if 300 <= fr <= 540 and fr % 5 == 0:
            bx = float(row["ball_x"])
            cx = float(row["cam_cx"])
            bpct = round(bx / 1920 * 100, 1)
            cpct = round(cx / 1920 * 100, 1)
            bic = row["ball_in_crop"]
            lines.append(f"  f{fr}: ball={bpct}% cam={cpct}% in_crop={bic}")

with open(RESULT, "w") as f:
    f.write("\n".join(lines))
