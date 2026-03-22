"""V16: Inject manual ball markers into the gap and apply extra smoothing.
Two steps:
1. Add user-provided ball x-positions for gap frames as synthetic YOLO detections
2. The pipeline's own smoothing will handle the rest, but we also increase smooth sigma

Usage: Edit MANUAL_MARKERS dict below with frame: ball_x pairs, then run.
"""
import json, os, shutil

STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"
TEL = rf"D:\Projects\soccer-video\out\telemetry\{STEM}"

CURATED = TEL + ".yolo_ball.curated_v15.jsonl"
OUTPUT = TEL + ".yolo_ball.manual_v16.jsonl"
MAIN = TEL + ".yolo_ball.jsonl"

# ============================================================
# USER: Fill in ball x-coordinates for gap frames
# Format: frame_number: ball_x_coordinate
# Use the grid lines in the review strip to estimate x position
# Ball y is typically around 800-900 for ground level
# Set to None if ball not visible in that frame
# ============================================================
MANUAL_MARKERS = {
    # Will be filled in by user
}

BALL_Y_DEFAULT = 850  # Approximate y for ground-level ball
MANUAL_CONF = 0.50    # Confidence for manual markers

def main():
    # Load curated v15
    dets = {}
    with open(CURATED) as f:
        for line in f:
            d = json.loads(line)
            dets[d["frame"]] = d

    print(f"Loaded {len(dets)} curated v15 detections")

    # Add manual markers
    added = 0
    for frame, bx in MANUAL_MARKERS.items():
        if bx is None:
            continue
        if frame in dets:
            print(f"  f{frame}: already has detection at x={dets[frame]['cx']:.0f}, skipping")
            continue
        dets[frame] = {
            "frame": frame,
            "cx": float(bx),
            "cy": float(BALL_Y_DEFAULT),
            "w": 20.0,
            "h": 20.0,
            "conf": MANUAL_CONF,
            "source": "manual"
        }
        added += 1
        print(f"  f{frame}: added manual marker at x={bx}")

    print(f"\nAdded {added} manual markers, total: {len(dets)} detections")

    # Sort by frame and write
    sorted_dets = sorted(dets.values(), key=lambda d: d["frame"])
    with open(OUTPUT, "w") as f:
        for d in sorted_dets:
            f.write(json.dumps(d) + "\n")
    print(f"Saved to {OUTPUT}")

    # Backup current main and replace
    backup = MAIN + ".pre_v16_backup"
    if os.path.exists(MAIN):
        shutil.copy2(MAIN, backup)
        print(f"Backed up main to {backup}")
    shutil.copy2(OUTPUT, MAIN)
    print(f"Copied to main YOLO file")

    # Delete stale caches that would interfere
    stale = [
        TEL + ".tracker_ball.jsonl",
        TEL + ".ball.cam_shifts.npy",
    ]
    for s in stale:
        if os.path.exists(s):
            os.remove(s)
            print(f"Deleted stale cache: {os.path.basename(s)}")

    # Show gap coverage
    gap_frames = [d["frame"] for d in sorted_dets if 170 <= d["frame"] <= 396]
    print(f"\nGap f170-f396 coverage: {len(gap_frames)} detections")
    if gap_frames:
        print(f"  Frames: {gap_frames}")
    else:
        print(f"  Still empty - add manual markers!")

if __name__ == "__main__":
    main()
