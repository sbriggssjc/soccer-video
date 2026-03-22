"""V16: Inject manual ball markers and prepare for re-render."""
import json, os, shutil

STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"
TEL = rf"D:\Projects\soccer-video\out\telemetry\{STEM}"

CURATED = TEL + ".yolo_ball.curated_v15.jsonl"
OUTPUT = TEL + ".yolo_ball.manual_v16.jsonl"
MAIN = TEL + ".yolo_ball.jsonl"

# User's markers as percentage of frame width (0-100)
# Converting to pixels: pct/100 * 1920
MARKERS_PCT = {
    170: 75,
    185: 62.5,   # halfway between 50 and 75
    200: 50,
    215: 37.5,   # halfway between 25 and 50
    230: 37.5,
    245: 37.5,
    260: 37.5,
    275: 37.5,   # user didn't mention 275, interpolate
    290: 37.5,   # user didn't mention 290, interpolate
    305: 48,     # just left of 50
    320: 37.5,
    335: 37.5,
    350: 37.5,
    365: 35,
    380: 30,
    395: 25,
    396: 25,
}

BALL_Y_DEFAULT = 850
MANUAL_CONF = 0.50

def main():
    # Load curated v15
    dets = {}
    with open(CURATED) as f:
        for line in f:
            d = json.loads(line)
            dets[d["frame"]] = d
    print(f"Loaded {len(dets)} curated v15 detections")

    # Convert percentages to pixels and add
    added = 0
    for frame, pct in MARKERS_PCT.items():
        bx = pct / 100.0 * 1920.0
        if frame in dets:
            print(f"  f{frame}: already has detection at x={dets[frame]['cx']:.0f}, REPLACING with manual x={bx:.0f}")
        dets[frame] = {
            "frame": frame,
            "cx": bx,
            "cy": float(BALL_Y_DEFAULT),
            "w": 20.0,
            "h": 20.0,
            "conf": MANUAL_CONF,
            "source": "manual"
        }
        added += 1
        print(f"  f{frame}: {pct}% -> x={bx:.0f}")

    print(f"\nAdded {added} manual markers, total: {len(dets)} detections")

    # Sort and write
    sorted_dets = sorted(dets.values(), key=lambda d: d["frame"])
    with open(OUTPUT, "w") as f:
        for d in sorted_dets:
            f.write(json.dumps(d) + "\n")
    print(f"Saved to {os.path.basename(OUTPUT)}")

    # Backup and replace main
    backup = MAIN + ".pre_v16_backup"
    if os.path.exists(MAIN):
        shutil.copy2(MAIN, backup)
    shutil.copy2(OUTPUT, MAIN)
    print(f"Copied to main YOLO file")

    # Delete stale caches
    for suffix in [".tracker_ball.jsonl", ".ball.cam_shifts.npy"]:
        s = TEL + suffix
        if os.path.exists(s):
            os.remove(s)
            print(f"Deleted stale: {os.path.basename(s)}")

    # Show coverage summary
    frames = sorted(d["frame"] for d in sorted_dets)
    gaps = [(frames[i], frames[i+1], frames[i+1]-frames[i])
            for i in range(len(frames)-1) if frames[i+1]-frames[i] > 10]
    print(f"\nTotal detections: {len(frames)}")
    print(f"Frame range: f{frames[0]}-f{frames[-1]}")
    print(f"Gaps > 10 frames: {len(gaps)}")
    for a, b, g in gaps:
        print(f"  f{a} -> f{b} ({g} frames)")

if __name__ == "__main__":
    main()
