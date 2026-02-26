import json, math

BALL_CACHE = "D:/Projects/soccer-video/out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.yolov8x.jsonl"
PERSON_CACHE = "D:/Projects/soccer-video/out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_person.jsonl"

CONF_THRESHOLD = 0.30
PROBLEM_ZONES = [(50, 140), (470, 500)]

# Load ball detections
balls = []
with open(BALL_CACHE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d["conf"] >= CONF_THRESHOLD:
            balls.append(d)

# Load person detections, grouped by frame
persons_by_frame = {}
with open(PERSON_CACHE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        fr = d["frame"]
        persons_by_frame[fr] = d["persons"]

print(f"Loaded {len(balls)} ball detections with conf >= {CONF_THRESHOLD}")
print(f"Loaded person detections across {len(persons_by_frame)} frames")
print()

# Sort balls by frame
balls.sort(key=lambda b: (b["frame"], -b["conf"]))

def is_problem_zone(frame):
    for lo, hi in PROBLEM_ZONES:
        if lo <= frame <= hi:
            return True
    return False

# Header
hdr = f"{'frame':>6} {'ball_x':>7} {'ball_y':>7} {'conf':>6} {'near_p':>7} {'clust_x':>8} {'dev_cx':>7} {'n_pers':>6} {'zone':>6}"
print(hdr)
print("-" * len(hdr))

# Stats collectors
all_near_dists = []
problem_zone_rows = []

for b in balls:
    fr = b["frame"]
    bx, by = b["cx"], b["cy"]
    conf = b["conf"]

    persons = persons_by_frame.get(fr, [])
    n_persons = len(persons)

    # Nearest person distance
    if persons:
        min_dist = min(math.hypot(bx - p["cx"], by - p["cy"]) for p in persons)
        centroid_x = sum(p["cx"] for p in persons) / len(persons)
        dev = bx - centroid_x
    else:
        min_dist = -1
        centroid_x = -1
        dev = -1

    zone_flag = "***" if is_problem_zone(fr) else ""

    print(f"{fr:>6} {bx:>7.1f} {by:>7.1f} {conf:>6.3f} {min_dist:>7.1f} {centroid_x:>8.1f} {dev:>7.1f} {n_persons:>6} {zone_flag:>6}")

    all_near_dists.append(min_dist)
    if is_problem_zone(fr):
        problem_zone_rows.append({
            "frame": fr, "bx": bx, "by": by, "conf": conf,
            "near_dist": min_dist, "centroid_x": centroid_x, "dev": dev,
            "n_persons": n_persons
        })

# Summary stats
print()
print("=" * 70)
print("SUMMARY")
print(f"  Total ball detections (conf >= {CONF_THRESHOLD}): {len(balls)}")
valid_dists = [d for d in all_near_dists if d >= 0]
if valid_dists:
    print(f"  Nearest-person distance: min={min(valid_dists):.1f}, max={max(valid_dists):.1f}, "
          f"mean={sum(valid_dists)/len(valid_dists):.1f}, median={sorted(valid_dists)[len(valid_dists)//2]:.1f}")

# Problem zone summary
print()
print("=" * 70)
print("PROBLEM ZONE DETAILS")
for lo, hi in PROBLEM_ZONES:
    zone_rows = [r for r in problem_zone_rows if lo <= r["frame"] <= hi]
    print(f"\n  --- Frames {lo}-{hi} ({len(zone_rows)} ball detections) ---")
    if zone_rows:
        for r in zone_rows:
            print(f"    frame={r['frame']:>5}, ball=({r['bx']:.0f},{r['by']:.0f}), conf={r['conf']:.3f}, "
                  f"near_person={r['near_dist']:.0f}px, centroid_x={r['centroid_x']:.0f}, "
                  f"dev_from_centroid={r['dev']:.0f}px, persons={r['n_persons']}")
    else:
        print("    (no ball detections in this range)")

# Frame gap analysis
print()
print("=" * 70)
print("FRAME GAPS (gaps > 5 frames between consecutive ball detections)")
frames_with_ball = sorted(set(b["frame"] for b in balls))
if len(frames_with_ball) > 1:
    gaps = []
    for i in range(1, len(frames_with_ball)):
        gap = frames_with_ball[i] - frames_with_ball[i-1]
        if gap > 5:
            gaps.append((frames_with_ball[i-1], frames_with_ball[i], gap))
    if gaps:
        for start, end, g in gaps:
            zone = " *** PROBLEM ZONE" if any(lo <= start <= hi or lo <= end <= hi for lo, hi in PROBLEM_ZONES) else ""
            print(f"  Gap: frame {start} -> {end} (gap={g} frames){zone}")
    else:
        print("  No gaps > 5 frames")
    print(f"  Frame range: {frames_with_ball[0]} to {frames_with_ball[-1]}")
    print(f"  Frames with ball: {len(frames_with_ball)} out of {frames_with_ball[-1] - frames_with_ball[0] + 1} possible")

