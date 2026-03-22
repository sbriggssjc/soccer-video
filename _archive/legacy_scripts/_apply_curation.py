"""Apply manual curation: keep only user-verified ball detections."""

import json, os, shutil

TELEMETRY = r"D:\Projects\soccer-video\out\telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"
MERGED_V13 = os.path.join(TELEMETRY, STEM + ".yolo_ball.merged_v13.jsonl")
MAIN_OUTPUT = os.path.join(TELEMETRY, STEM + ".yolo_ball.jsonl")
CURATED = os.path.join(TELEMETRY, STEM + ".yolo_ball.curated_v15.jsonl")

# User-verified KEEP frames
KEEP_FRAMES = {
    156, 157, 170, 396, 400, 410, 411, 416, 421,
    440, 441, 443, 444, 446, 447, 458, 465, 466,
    467, 468, 476, 477, 478, 479, 480, 481, 482,
    483, 484, 486, 487, 488, 495
}

# Load v13
dets = []
with open(MERGED_V13) as f:
    for line in f:
        dets.append(json.loads(line))
dets.sort(key=lambda d: d["frame"])
print(f"V13 total: {len(dets)} detections")

# Filter
kept = [d for d in dets if d["frame"] in KEEP_FRAMES]
removed = [d for d in dets if d["frame"] not in KEEP_FRAMES]
print(f"Kept: {len(kept)} (user-verified real ball)")
print(f"Removed: {len(removed)} (phantoms)")

# Check for any KEEP frames not found in v13
v13_frames = {d["frame"] for d in dets}
missing = KEEP_FRAMES - v13_frames
if missing:
    print(f"WARNING: KEEP frames not in v13: {sorted(missing)}")

# Show kept detections
print(f"\nKept detections:")
for d in kept:
    print(f"  f{d['frame']}: cx={d['cx']:.0f}, cy={d['cy']:.0f}, conf={d.get('conf',0):.3f}")

# Save
with open(CURATED, 'w') as f:
    for d in kept:
        f.write(json.dumps(d) + "\n")
print(f"\nCurated -> {CURATED}")

backup = MAIN_OUTPUT + ".pre_v15_backup"
if os.path.exists(MAIN_OUTPUT):
    shutil.copy2(MAIN_OUTPUT, backup)
shutil.copy2(CURATED, MAIN_OUTPUT)
print(f"Main file updated -> {MAIN_OUTPUT}")

# Delete stale caches
for suffix in [".tracker_ball.jsonl", ".ball.jsonl", ".ball.follow.jsonl", ".ball.follow__smooth.jsonl"]:
    p = os.path.join(TELEMETRY, STEM + suffix)
    if os.path.exists(p):
        os.remove(p)
        print(f"Deleted stale: {os.path.basename(p)}")

# Gap analysis
frames = sorted(d["frame"] for d in kept)
print(f"\nFrame range: {frames[0]}-{frames[-1]}")
print(f"Coverage: {len(frames)}/496 = {100*len(frames)/496:.1f}%")
gaps = [(frames[i-1], frames[i], frames[i]-frames[i-1]) for i in range(1, len(frames)) if frames[i]-frames[i-1] > 10]
if gaps:
    print(f"Gaps >10f:")
    for a, b, g in gaps:
        print(f"  f{a} -> f{b}: {g} frames")

print(f"\nDone! Run pipeline to regenerate v15.")
