"""V17: Add opening markers to v16 detections and prepare for re-render."""
import json, os, shutil

STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"
TEL = rf"D:\Projects\soccer-video\out\telemetry\{STEM}"

V16 = TEL + ".yolo_ball.manual_v16.jsonl"
OUTPUT = TEL + ".yolo_ball.manual_v17.jsonl"
MAIN = TEL + ".yolo_ball.jsonl"

# Opening markers (user provided, percentage of 1920px width)
OPENING_PCT = {
    0: 50,
    15: 55,
    30: 55,
    45: 62.5,
    60: 75,
    75: 90,
    90: 90,
    105: 90,
    120: 90,
    135: 90,    # user said 130 but strip showed f135
    150: 90,
    155: 85,
}

BALL_Y_DEFAULT = 850
MANUAL_CONF = 0.50

def main():
    # Load v16 detections (which include curated + gap markers)
    dets = {}
    with open(V16) as f:
        for line in f:
            d = json.loads(line)
            dets[d["frame"]] = d
    print(f"Loaded {len(dets)} v16 detections")

    # Add opening markers
    added = 0
    for frame, pct in OPENING_PCT.items():
        bx = pct / 100.0 * 1920.0
        if frame in dets:
            print(f"  f{frame}: already exists at x={dets[frame]['cx']:.0f}, replacing with x={bx:.0f}")
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

    print(f"\nAdded {added} opening markers, total: {len(dets)} detections")

    # Sort and write
    sorted_dets = sorted(dets.values(), key=lambda d: d["frame"])
    with open(OUTPUT, "w") as f:
        for d in sorted_dets:
            f.write(json.dumps(d) + "\n")
    print(f"Saved to {os.path.basename(OUTPUT)}")

    # Backup and replace main
    backup = MAIN + ".pre_v17_backup"
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

    # Coverage summary
    frames = sorted(d["frame"] for d in sorted_dets)
    print(f"\nTotal: {len(frames)} detections, f{frames[0]}-f{frames[-1]}")
    gaps = [(frames[i], frames[i+1], frames[i+1]-frames[i])
            for i in range(len(frames)-1) if frames[i+1]-frames[i] > 10]
    print(f"Gaps > 10 frames: {len(gaps)}")
    for a, b, g in gaps:
        print(f"  f{a} -> f{b} ({g} frames)")

if __name__ == "__main__":
    main()
