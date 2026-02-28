"""Build consensus YOLO detections from multiple models.

Strategy:
1. Load detections from yolov8n, yolov8m, yolov8x (all at 1920px via BallTracker)
2. For each frame, check if 2+ models agree (within AGREE_RADIUS pixels)
3. Also keep any single-model detection with conf > HIGH_CONF_SOLO
4. Filter out obvious false positives (y > 700 = bottom of frame, not field)
5. Save as pipeline-compatible .yolo_ball.jsonl
"""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(r"D:\Projects\soccer-video")
TEL_DIR = REPO / "out" / "telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"

AGREE_RADIUS = 120  # px - two models must agree within this
HIGH_CONF_SOLO = 0.40  # single-model detections above this are kept
MAX_FIELD_Y = 650  # detections below this y are likely false (bottom of frame)
N_FRAMES = 496

def load_jsonl(path):
    dets = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            dets[d["frame"]] = (d["cx"], d["cy"], d["conf"])
    return dets

# Load all models
nano = load_jsonl(TEL_DIR / f"{STEM}.yolo_ball.jsonl")
medium = load_jsonl(TEL_DIR / f"{STEM}.yolo_ball.yolov8m.jsonl")
xlarge = load_jsonl(TEL_DIR / f"{STEM}.yolo_ball.yolov8x.jsonl")

print(f"Raw detections: nano={len(nano)}, medium={len(medium)}, xlarge={len(xlarge)}")

# Gather all frames with any detection
all_frames = sorted(set(nano.keys()) | set(medium.keys()) | set(xlarge.keys()))
print(f"Frames with any detection: {len(all_frames)}")

consensus = {}
stats = defaultdict(int)

for f in range(N_FRAMES):
    candidates = []
    if f in nano:
        candidates.append(("nano", *nano[f]))
    if f in medium:
        candidates.append(("medium", *medium[f]))
    if f in xlarge:
        candidates.append(("xlarge", *xlarge[f]))
    
    if not candidates:
        continue
    
    # Filter out obvious false positives (bottom of frame)
    field_candidates = [(name, cx, cy, conf) for name, cx, cy, conf in candidates if cy <= MAX_FIELD_Y]
    
    if not field_candidates:
        stats["filtered_bottom"] += 1
        continue
    
    # Check for multi-model agreement
    best = None
    best_score = 0
    
    for i, (n1, cx1, cy1, c1) in enumerate(field_candidates):
        agreeing = [(n1, cx1, cy1, c1)]
        for j, (n2, cx2, cy2, c2) in enumerate(field_candidates):
            if i == j:
                continue
            dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
            if dist < AGREE_RADIUS:
                agreeing.append((n2, cx2, cy2, c2))
        
        if len(agreeing) >= 2:
            # Multi-model consensus - use confidence-weighted average
            total_conf = sum(c for _, _, _, c in agreeing)
            avg_cx = sum(cx * c for _, cx, _, c in agreeing) / total_conf
            avg_cy = sum(cy * c for _, _, cy, c in agreeing) / total_conf
            score = total_conf * len(agreeing)  # reward more agreement
            if score > best_score:
                best = (avg_cx, avg_cy, max(c for _, _, _, c in agreeing), len(agreeing))
                best_score = score
    
    if best:
        consensus[f] = best[:3]  # (cx, cy, conf)
        stats[f"agree_{best[3]}"] += 1
    else:
        # No multi-model agreement - check for high-confidence solo
        top = max(field_candidates, key=lambda x: x[3])
        if top[3] >= HIGH_CONF_SOLO:
            consensus[f] = (top[1], top[2], top[3])
            stats["solo_highconf"] += 1
        else:
            stats["rejected_lowconf_solo"] += 1

print(f"\nConsensus results: {len(consensus)} frames")
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")

# Show coverage
if consensus:
    frames_sorted = sorted(consensus.keys())
    gaps = []
    for i in range(1, len(frames_sorted)):
        gap = frames_sorted[i] - frames_sorted[i-1]
        if gap > 5:
            gaps.append((frames_sorted[i-1], frames_sorted[i], gap))
    gaps.sort(key=lambda x: -x[2])
    print(f"\nCoverage: frames {frames_sorted[0]}-{frames_sorted[-1]}")
    print(f"Largest gaps:")
    for start, end, length in gaps[:8]:
        print(f"  frames {start}-{end} ({length} frames, {length/30:.1f}s)")

# Save as pipeline-compatible .yolo_ball.jsonl
out_path = TEL_DIR / f"{STEM}.yolo_ball.consensus.jsonl"
with open(out_path, "w") as f:
    for frame in sorted(consensus.keys()):
        cx, cy, conf = consensus[frame]
        t = frame / 30.0
        f.write(json.dumps({"frame": frame, "t": round(t, 6), "cx": round(cx, 2), "cy": round(cy, 2), "conf": round(conf, 4)}) + "\n")

print(f"\nSaved {len(consensus)} consensus detections to {out_path.name}")

# Also save as the main yolo_ball.jsonl for pipeline use (backup first)
main_path = TEL_DIR / f"{STEM}.yolo_ball.jsonl"
backup_path = TEL_DIR / f"{STEM}.yolo_ball.nano_original.jsonl"
import shutil
if not backup_path.exists():
    shutil.copy2(main_path, backup_path)
    print(f"Backed up original nano detections to {backup_path.name}")

shutil.copy2(out_path, main_path)
print(f"Replaced {main_path.name} with consensus detections")
