"""
Template-matching ball filter v14b.

Key fix from v14: use small template patches (20px radius) and
normalized cross-correlation instead of color histograms.
Also combine with trajectory consistency check.

Strategy:
1. Extract small ball patches from top-confidence v13 detections
2. For each candidate, extract patch and do template correlation
3. Also require trajectory consistency (within 250px of interpolated)
4. Velocity scrub remaining outliers
"""

import json, csv, os, sys, shutil, math
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: cv2 required"); sys.exit(1)

# === Paths ===
CLIP = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
TELEMETRY = r"D:\Projects\soccer-video\out\telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"
OUTPUT_DIR = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"

NANO   = os.path.join(TELEMETRY, STEM + ".yolo_ball.nano_original.jsonl")
MEDIUM = os.path.join(TELEMETRY, STEM + ".yolo_ball.yolov8m.jsonl")
XLARGE = os.path.join(TELEMETRY, STEM + ".yolo_ball.yolov8x.jsonl")
MERGED_V13 = os.path.join(TELEMETRY, STEM + ".yolo_ball.merged_v13.jsonl")
MAIN_OUTPUT = os.path.join(TELEMETRY, STEM + ".yolo_ball.jsonl")
TEMPLATE_OUTPUT = os.path.join(TELEMETRY, STEM + ".yolo_ball.template_v14b.jsonl")
REVIEW_STRIP = os.path.join(OUTPUT_DIR, "002__ball_review_strip_v14b.png")

# === Config ===
PATCH_RADIUS = 18         # Small patch around detection center
REF_TOP_N = 8             # Reference detections to use
TEMPLATE_THRESH = 0.25    # Min template correlation to keep (NCC, -1 to 1)
TRAJ_MAX_DIST = 250       # Max distance from interpolated trajectory
MAX_VELOCITY = 80         # For velocity scrub
MAX_FIELD_Y = 650
TOTAL_FRAMES = 496

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    dets = []
    with open(path) as f:
        for line in f:
            dets.append(json.loads(line))
    return sorted(dets, key=lambda d: d["frame"])

def save_jsonl(dets, path):
    with open(path, 'w') as f:
        for d in dets:
            f.write(json.dumps(d) + "\n")

def extract_patch(cap, frame_idx, cx, cy, radius=PATCH_RADIUS):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    h, w = frame.shape[:2]
    x0 = max(0, int(cx - radius))
    y0 = max(0, int(cy - radius))
    x1 = min(w, int(cx + radius))
    y1 = min(h, int(cy + radius))
    patch = frame[y0:y1, x0:x1]
    if patch.shape[0] < 10 or patch.shape[1] < 10:
        return None
    # Resize to standard size for comparison
    return cv2.resize(patch, (2*radius, 2*radius))

def template_similarity(patch, ref_patches):
    """Compare patch against multiple reference patches using NCC."""
    if patch is None:
        return -1
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    scores = []
    for ref in ref_patches:
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Normalized cross-correlation
        result = cv2.matchTemplate(
            gray.reshape(1, -1),  # Flatten for 1D correlation
            ref_gray.reshape(1, -1),
            cv2.TM_CCOEFF_NORMED
        )
        scores.append(float(result[0, 0]))
    return max(scores) if scores else -1

# === Load data ===
print("=== Loading data ===")
v13_dets = load_jsonl(MERGED_V13)
nano = load_jsonl(NANO)
medium = load_jsonl(MEDIUM)
xlarge = load_jsonl(XLARGE)
print(f"  V13: {len(v13_dets)}, Nano: {len(nano)}, Medium: {len(medium)}, Xlarge: {len(xlarge)}")

# First, restore v13 to main output
backup_v14 = MAIN_OUTPUT + ".pre_v14b_backup"
v13_backup = MAIN_OUTPUT + ".pre_v14_backup"
if os.path.exists(v13_backup):
    shutil.copy2(v13_backup, MAIN_OUTPUT)
    print(f"  Reverted main file to pre-v14 state (v13)")
else:
    shutil.copy2(MERGED_V13, MAIN_OUTPUT)
    print(f"  Restored main file from v13 merged")

# Build combined candidates
all_cands = {}
for d in xlarge:
    f = d["frame"]
    if f not in all_cands or d.get("conf", 0) > all_cands[f].get("conf", 0):
        d["_model"] = "xlarge"
        all_cands[f] = d
for d in medium:
    f = d["frame"]
    if f not in all_cands or d.get("conf", 0) > all_cands[f].get("conf", 0):
        d["_model"] = "medium"
        all_cands[f] = d
for d in nano:
    f = d["frame"]
    if f not in all_cands or d.get("conf", 0) > all_cands[f].get("conf", 0):
        d["_model"] = "nano"
        all_cands[f] = d
# Add v13 (which includes portrait pass detections)
for d in v13_dets:
    f = d["frame"]
    if f not in all_cands or d.get("conf", 0) > all_cands[f].get("conf", 0):
        d["_model"] = "v13"
        all_cands[f] = d

candidates = sorted(all_cands.values(), key=lambda d: d["frame"])
print(f"  Total candidates: {len(candidates)}")

# === Build interpolated trajectory from v13 ===
print("\n=== Building interpolated trajectory from v13 ===")
v13_sorted = sorted(v13_dets, key=lambda d: d["frame"])
interp_cx = np.full(TOTAL_FRAMES, np.nan)
for d in v13_dets:
    interp_cx[d["frame"]] = d["cx"]
for i in range(len(v13_sorted) - 1):
    f0, f1 = v13_sorted[i]["frame"], v13_sorted[i+1]["frame"]
    cx0, cx1 = v13_sorted[i]["cx"], v13_sorted[i+1]["cx"]
    for f in range(f0 + 1, f1):
        if f < TOTAL_FRAMES:
            t = (f - f0) / (f1 - f0)
            interp_cx[f] = cx0 + t * (cx1 - cx0)
if v13_sorted:
    for f in range(0, v13_sorted[0]["frame"]):
        interp_cx[f] = v13_sorted[0]["cx"]
    for f in range(v13_sorted[-1]["frame"] + 1, TOTAL_FRAMES):
        interp_cx[f] = v13_sorted[-1]["cx"]

# === Build reference patches ===
print("\n=== Building reference ball patches ===")
v13_by_conf = sorted(v13_dets, key=lambda d: -d.get("conf", 0))
ref_dets = v13_by_conf[:REF_TOP_N]

cap = cv2.VideoCapture(CLIP)
ref_patches = []
for d in ref_dets:
    patch = extract_patch(cap, d["frame"], d["cx"], d["cy"])
    if patch is not None:
        ref_patches.append(patch)
        print(f"  f{d['frame']}: cx={d['cx']:.0f}, conf={d.get('conf',0):.3f}, patch {patch.shape}")

print(f"  Reference patches: {len(ref_patches)}")

# === Score candidates ===
print(f"\n=== Scoring {len(candidates)} candidates ===")
scored = []
for d in candidates:
    f = d["frame"]
    if d.get("cy", 0) > MAX_FIELD_Y:
        continue

    # Trajectory check
    if not np.isnan(interp_cx[f]):
        traj_dist = abs(d["cx"] - interp_cx[f])
    else:
        traj_dist = 0

    # Template matching
    patch = extract_patch(cap, f, d["cx"], d["cy"])
    sim = template_similarity(patch, ref_patches)

    d["_traj_dist"] = round(traj_dist, 1)
    d["_template_sim"] = round(sim, 4)
    d["_patch"] = patch
    scored.append(d)

# Analyze distributions
sims = [d["_template_sim"] for d in scored]
dists = [d["_traj_dist"] for d in scored]
print(f"  Template sim range: {min(sims):.3f} - {max(sims):.3f}")
print(f"  Template sim mean: {np.mean(sims):.3f}, median: {np.median(sims):.3f}")
print(f"  Traj dist range: {min(dists):.0f} - {max(dists):.0f}")

# Score distribution
bins = [-0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist_counts, _ = np.histogram(sims, bins=bins)
print(f"  Template similarity distribution:")
for i in range(len(bins)-1):
    bar = "#" * hist_counts[i]
    print(f"    {bins[i]:+.1f} to {bins[i+1]:+.1f}: {hist_counts[i]:3d} {bar}")

# === Filter: template + trajectory ===
print(f"\n=== Filtering (template >= {TEMPLATE_THRESH} AND traj_dist <= {TRAJ_MAX_DIST}) ===")
accepted = []
rejected = []
for d in scored:
    passes_template = d["_template_sim"] >= TEMPLATE_THRESH
    passes_traj = d["_traj_dist"] <= TRAJ_MAX_DIST
    # Must pass BOTH checks
    if passes_template and passes_traj:
        accepted.append(d)
    else:
        rejected.append(d)

print(f"  Accepted: {len(accepted)}")
print(f"  Rejected: {len(rejected)}")

# Show rejections
if rejected:
    print(f"\n  Rejected detections:")
    for d in sorted(rejected, key=lambda x: x["frame"]):
        reason = []
        if d["_template_sim"] < TEMPLATE_THRESH:
            reason.append(f"template={d['_template_sim']:.3f}")
        if d["_traj_dist"] > TRAJ_MAX_DIST:
            reason.append(f"traj_dist={d['_traj_dist']:.0f}")
        print(f"    f{d['frame']}: cx={d['cx']:.0f}, conf={d.get('conf',0):.2f}, {', '.join(reason)}")

# === Velocity scrub ===
print(f"\n=== Velocity scrub ===")
accepted.sort(key=lambda d: d["frame"])
scrubbed = set()
for iteration in range(5):
    active = [d for d in accepted if d["frame"] not in scrubbed]
    new_scrubs = set()
    for i in range(1, len(active)):
        d0, d1 = active[i-1], active[i]
        dt = d1["frame"] - d0["frame"]
        if dt == 0 or dt > 8:
            continue
        vel = abs(d1["cx"] - d0["cx"]) / dt
        if vel > MAX_VELOCITY:
            # Which is the outlier?
            v_in = abs(d0["cx"] - active[i-2]["cx"]) / max(1, d0["frame"] - active[i-2]["frame"]) if i >= 2 else 0
            v_out = abs(d1["cx"] - active[i+1]["cx"]) / max(1, active[i+1]["frame"] - d1["frame"]) if i < len(active)-1 else 0
            # Remove the one with higher velocity on its other side
            if v_in > MAX_VELOCITY * 0.6:
                new_scrubs.add(d0["frame"])
            elif v_out > MAX_VELOCITY * 0.6:
                new_scrubs.add(d1["frame"])
            elif d0.get("conf", 0) < d1.get("conf", 0):
                new_scrubs.add(d0["frame"])
            else:
                new_scrubs.add(d1["frame"])
    if not new_scrubs:
        print(f"  Iteration {iteration+1}: clean")
        break
    print(f"  Iteration {iteration+1}: scrubbed {len(new_scrubs)}: {sorted(new_scrubs)}")
    scrubbed.update(new_scrubs)

final = [d for d in accepted if d["frame"] not in scrubbed]
print(f"\n  Final: {len(final)} detections ({100*len(final)/TOTAL_FRAMES:.1f}% coverage)")

# === Generate review strip ===
print(f"\n=== Generating review strip ===")
# Include accepted and rejected, sorted by frame
all_items = [(("KEEP" if d["frame"] not in scrubbed else "SCRUB"), d) for d in accepted]
all_items += [("REJECT", d) for d in rejected]
all_items.sort(key=lambda x: x[1]["frame"])

THUMB = 64
COLS = 20
n = min(len(all_items), COLS * 16)
ROWS = math.ceil(n / COLS)
strip_w = COLS * (THUMB + 4) + 4
strip_h = ROWS * (THUMB + 22) + 4
strip = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 30

for idx, (label, d) in enumerate(all_items[:n]):
    row, col = idx // COLS, idx % COLS
    x0 = col * (THUMB + 4) + 4
    y0 = row * (THUMB + 22) + 4
    patch = d.get("_patch")
    if patch is not None:
        thumb = cv2.resize(patch, (THUMB, THUMB))
    else:
        thumb = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
    colors = {"KEEP": (0, 200, 0), "REJECT": (0, 0, 200), "SCRUB": (0, 140, 255)}
    cv2.rectangle(thumb, (0, 0), (THUMB-1, THUMB-1), colors.get(label, (128,128,128)), 2)
    strip[y0:y0+THUMB, x0:x0+THUMB] = thumb
    text = f"f{d['frame']} {d['_template_sim']:.2f}"
    cv2.putText(strip, text, (x0, y0+THUMB+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1)

cv2.imwrite(REVIEW_STRIP, strip)
print(f"  Review strip -> {REVIEW_STRIP}")

cap.release()

# === Save final ===
print(f"\n=== Saving ===")
clean_final = []
for d in final:
    clean = {k: v for k, v in d.items() if not k.startswith("_")}
    clean_final.append(clean)
clean_final.sort(key=lambda d: d["frame"])

# Dedup by frame
deduped = {}
for d in clean_final:
    f = d["frame"]
    if f not in deduped or d.get("conf", 0) > deduped[f].get("conf", 0):
        deduped[f] = d
clean_final = sorted(deduped.values(), key=lambda d: d["frame"])

save_jsonl(clean_final, TEMPLATE_OUTPUT)
print(f"  Template-filtered -> {TEMPLATE_OUTPUT}")

backup = MAIN_OUTPUT + ".pre_v14b_backup"
if os.path.exists(MAIN_OUTPUT):
    shutil.copy2(MAIN_OUTPUT, backup)
shutil.copy2(TEMPLATE_OUTPUT, MAIN_OUTPUT)
print(f"  Main file updated")

# Stale caches
for suffix in [".tracker_ball.jsonl", ".ball.jsonl", ".ball.follow.jsonl", ".ball.follow__smooth.jsonl"]:
    p = os.path.join(TELEMETRY, STEM + suffix)
    if os.path.exists(p):
        os.remove(p)
        print(f"  Deleted stale: {os.path.basename(p)}")

# Gap analysis
frames = [d["frame"] for d in clean_final]
gaps = [(frames[i-1], frames[i], frames[i]-frames[i-1]) for i in range(1, len(frames)) if frames[i]-frames[i-1] > 10]
print(f"\n  Gaps >10f: {len(gaps)}")
for a, b, g in gaps:
    print(f"    f{a} -> f{b}: {g} frames")

print(f"\nDone! {len(clean_final)} detections. Run pipeline to regenerate.")
