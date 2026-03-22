"""Deep diagnostic analysis of clip 004 framing issues.
Look at: crop dimensions, ball-in-crop, camera vs ball lag, speed limiting.
"""
import os, csv, json
import numpy as np

RESULT = r"D:\Projects\soccer-video\_tmp\diagnose_004_result.txt"
os.makedirs(os.path.dirname(RESULT), exist_ok=True)

diag = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00__portrait.diag.csv"

FRAME_W = 1920
FPS = 30

with open(diag, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

lines = []
lines.append(f"=== CLIP 004 FRAMING ANALYSIS ===")
lines.append(f"Total frames: {len(rows)}")
lines.append("")

# 1. Crop dimensions
crop_ws = [float(r["crop_w"]) for r in rows]
crop_hs = [float(r["crop_h"]) for r in rows]
lines.append("CROP DIMENSIONS:")
lines.append(f"  Width:  min={min(crop_ws):.0f} max={max(crop_ws):.0f} avg={np.mean(crop_ws):.0f}")
lines.append(f"  Height: min={min(crop_hs):.0f} max={max(crop_hs):.0f} avg={np.mean(crop_hs):.0f}")
lines.append(f"  Portrait output: 1080x1920")
lines.append(f"  Crop covers {min(crop_ws)/FRAME_W*100:.0f}-{max(crop_ws)/FRAME_W*100:.0f}% of source width")
lines.append("")

# 2. Ball in crop analysis
out_of_crop = [(int(r["frame"]), r) for r in rows if r["ball_in_crop"] == "0"]
lines.append(f"BALL OUT OF CROP: {len(out_of_crop)} frames")
if out_of_crop:
    # Group consecutive ranges
    ranges = []
    start = out_of_crop[0][0]
    prev = start
    for fr, _ in out_of_crop[1:]:
        if fr > prev + 1:
            ranges.append((start, prev))
            start = fr
        prev = fr
    ranges.append((start, prev))
    for s, e in ranges:
        dur = (e - s + 1) / FPS
        lines.append(f"  f{s}-f{e} ({dur:.1f}s)")
lines.append("")

# 3. Speed limiting analysis
speed_limited = [(int(r["frame"]), r) for r in rows if r["speed_limited"] == "1"]
lines.append(f"SPEED LIMITED FRAMES: {len(speed_limited)} / {len(rows)}")
lines.append("")

# 4. Camera vs ball lag (where camera is far from ball)
lines.append("CAMERA-BALL LAG (frames where |cam - ball| > 10% of frame width):")
big_lag = []
for r in rows:
    fr = int(r["frame"])
    bx = float(r["ball_x"])
    cx = float(r["cam_cx"])
    lag_pct = abs(cx - bx) / FRAME_W * 100
    if lag_pct > 10:
        big_lag.append((fr, bx, cx, lag_pct))

if big_lag:
    # Group ranges
    i = 0
    while i < len(big_lag):
        start_fr = big_lag[i][0]
        max_lag = big_lag[i][3]
        j = i
        while j < len(big_lag) - 1 and big_lag[j+1][0] - big_lag[j][0] <= 3:
            max_lag = max(max_lag, big_lag[j+1][3])
            j += 1
        end_fr = big_lag[j][0]
        dur = (end_fr - start_fr + 1) / FPS
        lines.append(f"  f{start_fr}-f{end_fr} ({dur:.1f}s) max_lag={max_lag:.0f}%")
        i = j + 1
else:
    lines.append("  None (camera tracks ball well)")
lines.append("")

# 5. Frame-by-frame detail in the problematic zones
lines.append("DETAILED: Fast movement zones (every 3 frames)")
for r in rows:
    fr = int(r["frame"])
    if fr % 3 != 0:
        continue
    bx = float(r["ball_x"])
    cx = float(r["cam_cx"])
    cw = float(r["crop_w"])
    x0 = float(r["crop_x0"])
    bic = r["ball_in_crop"]
    sl = r["speed_limited"]
    dte = float(r["dist_to_edge"])
    flags = r.get("clamp_flags", "")

    bpct = bx / FRAME_W * 100
    cpct = cx / FRAME_W * 100
    lag = abs(cpct - bpct)

    # Only show if interesting (lag > 5%, speed limited, or ball out)
    if lag > 5 or sl == "1" or bic == "0":
        marker = ""
        if bic == "0": marker += " [BALL_OUT]"
        if sl == "1": marker += " [SPEED_LIM]"
        if lag > 10: marker += f" [LAG={lag:.0f}%]"
        # Extract zoom flags
        zoom_flags = [f for f in flags.split("|") if "zoom" in f.lower()]
        if zoom_flags:
            marker += f" [{','.join(zoom_flags)}]"
        lines.append(f"  f{fr}: ball={bpct:.0f}% cam={cpct:.0f}% crop_w={cw:.0f} dte={dte:.0f}{marker}")

lines.append("")

# 6. Vidstab zoom impact
lines.append("VIDSTAB SETTINGS IMPACT:")
lines.append(f"  Current: zoom=3 (adds 3% zoom on top of stabilization)")
lines.append(f"  Current crop width avg: {np.mean(crop_ws):.0f}px = {np.mean(crop_ws)/FRAME_W*100:.0f}% of 1920")
lines.append(f"  After vidstab zoom=3: effective crop ~{np.mean(crop_ws)*0.97/FRAME_W*100:.0f}% of 1920")
lines.append(f"  Consider: zoom=1 or zoom=0 for wider view during fast action")

with open(RESULT, "w") as f:
    f.write("\n".join(lines))
