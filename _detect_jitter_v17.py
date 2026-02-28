"""Detect rapid side-to-side jitter in v17 camera path.
Jitter = rapid direction changes within a short window."""
import csv, os
import numpy as np

diag = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__manual_v17.diag.csv"
rows = list(csv.DictReader(open(diag)))
cam_cx = np.array([float(r["cam_cx"]) for r in rows])
sources = [r["source"] for r in rows]

# Frame-to-frame deltas
dx = np.diff(cam_cx)
deltas = np.abs(dx)

# Detect direction reversals
directions = np.zeros(len(dx), dtype=int)
directions[dx > 1.0] = 1
directions[dx < -1.0] = -1

# Find rapid reversals: direction changes within short windows
print("=== V17 JITTER DETECTION ===\n")

# Method 1: Count reversals in sliding 10-frame windows
WINDOW = 10
jitter_zones = []
for i in range(len(directions) - WINDOW):
    segment = directions[i:i+WINDOW]
    nonzero = segment[segment != 0]
    if len(nonzero) < 3:
        continue
    changes = np.sum(np.diff(nonzero) != 0)
    max_delta = deltas[i:i+WINDOW].max()
    if changes >= 3 and max_delta > 5:
        jitter_zones.append((i, changes, max_delta))

# Merge overlapping zones
merged = []
for zone in jitter_zones:
    if merged and zone[0] <= merged[-1][0] + WINDOW:
        # Extend existing zone
        merged[-1] = (merged[-1][0], max(merged[-1][1], zone[1]), max(merged[-1][2], zone[2]))
    else:
        merged.append(zone)

print(f"Found {len(merged)} jitter zones (>= 3 reversals in {WINDOW}-frame window):\n")
for start, changes, max_d in merged:
    end = min(start + WINDOW + 5, len(cam_cx) - 1)
    # Expand to find full extent
    while end < len(cam_cx) - 1 and (start, end) and any(
        abs(cam_cx[j+1] - cam_cx[j]) > 3 for j in range(max(start-2, end-5), min(end+1, len(cam_cx)-1))
    ):
        end += 1
        if end - start > 40:
            break
    
    segment_src = sources[start:end+1]
    src_counts = {}
    for s in segment_src:
        src_counts[s] = src_counts.get(s, 0) + 1
    src_str = ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items(), key=lambda x: -x[1]))
    
    print(f"  f{start}-f{end} (t={start/30:.1f}s-{end/30:.1f}s)")
    print(f"    reversals={changes}, max_delta={max_d:.1f}px, sources: {src_str}")
    
    # Show the actual cam_cx values
    vals = cam_cx[start:end+1]
    frame_strs = []
    for j, v in enumerate(vals):
        f = start + j
        d = f"({dx[f-1]:+.1f})" if f > 0 and f-1 < len(dx) else ""
        frame_strs.append(f"f{f}={v:.0f}{d}")
    # Show first and last few
    if len(frame_strs) > 12:
        show = frame_strs[:6] + ["..."] + frame_strs[-6:]
    else:
        show = frame_strs
    print(f"    path: {', '.join(show)}")
    print()

# Method 2: High-frequency energy (second derivative)
print("\n=== HIGH-FREQUENCY ANALYSIS ===\n")
accel = np.diff(dx)  # second derivative = acceleration
abs_accel = np.abs(accel)

# Rolling sum of absolute acceleration (jerk proxy)
JERK_WINDOW = 8
jerk = np.convolve(abs_accel, np.ones(JERK_WINDOW), mode='valid')

print(f"Overall jerk stats: mean={jerk.mean():.1f}, p95={np.percentile(jerk, 95):.1f}, max={jerk.max():.1f}")
print(f"\nTop 10 jerkiest {JERK_WINDOW}-frame windows:")
top_idx = np.argsort(jerk)[-10:][::-1]
seen = set()
for idx in top_idx:
    # Skip if overlapping with already reported
    if any(abs(idx - s) < JERK_WINDOW for s in seen):
        continue
    seen.add(idx)
    src_seg = sources[idx:idx+JERK_WINDOW]
    src_str = ", ".join(set(src_seg))
    cx_seg = cam_cx[idx:idx+JERK_WINDOW+1]
    dx_seg = np.diff(cx_seg)
    print(f"  f{idx}-f{idx+JERK_WINDOW}: jerk={jerk[idx]:.1f}, deltas=[{', '.join(f'{d:+.1f}' for d in dx_seg)}], sources={src_str}")
