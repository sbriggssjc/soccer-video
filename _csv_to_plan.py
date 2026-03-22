"""Convert review CSVs (camera_x_pct) to JSON plan files for render_follow_unified.py.

Reads review_neofc2017_XX.csv from Desktop, converts camera_x_pct (0-100%)
to absolute pixel cx values, and writes JSON plan files that --plan accepts.

Usage: python _csv_to_plan.py
"""
import csv, json, os, sys
from pathlib import Path

# ===== CONFIGURE =====
GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
PREFIX = "neofc2017_"
SOURCE_W = 1920  # source frame width in pixels
SOURCE_H = 1080  # source frame height in pixels
FPS = 30.0
# ======================

os.chdir(r"D:\Projects\soccer-video")

clips_dir = Path(f"out/atomic_clips/{GAME}")
desktop = Path(r"C:\Users\scott\Desktop")
plans_dir = Path(f"_tmp/plans/{GAME}")
plans_dir.mkdir(parents=True, exist_ok=True)

clip_files = sorted(clips_dir.glob("*.mp4"),
                    key=lambda p: int(p.stem.split("__")[0]))

converted = 0
skipped = 0

for clip_file in clip_files:
    stem = clip_file.stem
    clip_num = stem.split("__")[0]

    csv_path = desktop / f"review_{PREFIX}{clip_num}.csv"
    if not csv_path.exists():
        print(f"  {clip_num}: no review CSV on Desktop, skipping")
        skipped += 1
        continue

    # Read the review CSV
    keyframes = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            time_s = float(row["time_s"])
            cam_pct = row.get("camera_x_pct", "").strip()

            if not cam_pct:
                continue  # skip rows with no camera correction

            # Convert percentage (0-100) to absolute pixels
            cx = float(cam_pct) / 100.0 * SOURCE_W
            cy = SOURCE_H / 2.0  # vertical center

            keyframes.append({
                "t": time_s,
                "frame": frame,
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "zoom": 1.0,
                "width": float(SOURCE_W),
                "height": float(SOURCE_H)
            })

    if not keyframes:
        print(f"  {clip_num}: review CSV has no camera_x_pct edits, skipping")
        skipped += 1
        continue

    # Write JSON plan
    plan = {
        "version": 1,
        "meta": {
            "game": GAME,
            "clip": clip_num,
            "source": str(csv_path)
        },
        "keyframes": keyframes
    }

    plan_path = plans_dir / f"plan_{clip_num}.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    # Also copy to Desktop for reference
    desktop_plan = desktop / f"plan_{PREFIX}{clip_num}.json"
    with open(desktop_plan, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"  {clip_num}: {len(keyframes)} keyframes -> {plan_path.name}")
    converted += 1

print(f"\n{'='*60}")
print(f"DONE! {converted} plans created, {skipped} skipped")
print(f"Plans saved to: {plans_dir}")
print(f"{'='*60}")
print()
print("Now re-render with:")
print(f"  python _rerender_with_plans.py")
