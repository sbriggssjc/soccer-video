"""Convert review CSVs (camera_x_pct) to manual-anchors CSVs for render_follow_unified.py.

Reads review_neofc2017_XX.csv from Desktop, converts camera_x_pct (0-100%)
to absolute pixel ball_x values, writes anchors CSVs that --manual-anchors accepts.

Format: frame,ball_x,ball_y  (pixel coordinates in source video space)

Usage: python _csv_to_anchors.py
"""
import csv, os
from pathlib import Path

# ===== CONFIGURE =====
GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
PREFIX = "neofc2017_"
SOURCE_W = 1920  # source frame width in pixels
SOURCE_H = 1080  # source frame height in pixels
# ======================

os.chdir(r"D:\Projects\soccer-video")

clips_dir = Path(f"out/atomic_clips/{GAME}")
desktop = Path(r"C:\Users\scott\Desktop")
anchors_dir = Path(f"_tmp/anchors/{GAME}")
anchors_dir.mkdir(parents=True, exist_ok=True)

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

    # Read the review CSV and build anchor rows
    anchor_rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            cam_pct = row.get("camera_x_pct", "").strip()

            if not cam_pct:
                continue  # skip rows with no camera value

            # Convert percentage (0-100) to absolute pixels
            ball_x = float(cam_pct) / 100.0 * SOURCE_W
            ball_y = SOURCE_H / 2.0  # vertical center

            anchor_rows.append([frame, round(ball_x, 1), round(ball_y, 1)])

    if not anchor_rows:
        print(f"  {clip_num}: review CSV has no camera_x_pct values, skipping")
        skipped += 1
        continue

    # Write anchors CSV (frame,ball_x,ball_y)
    anchors_path = anchors_dir / f"anchors_{clip_num}.csv"
    with open(anchors_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "ball_x", "ball_y"])
        writer.writerows(anchor_rows)

    print(f"  {clip_num}: {len(anchor_rows)} anchors -> {anchors_path.name}")
    converted += 1

print(f"\n{'='*60}")
print(f"DONE! {converted} anchor files created, {skipped} skipped")
print(f"Anchors saved to: {anchors_dir}")
print(f"{'='*60}")
print()
print("Now re-render with:")
print(f"  python _rerender_with_anchors.py")
