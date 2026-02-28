"""Analyze v15 camera jitter and generate gap frame strip for manual marking."""
import csv, os, sys
import numpy as np

diag = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__curated_v15.diag.csv"
rows = list(csv.DictReader(open(diag)))

cam_cx = np.array([float(r["cam_cx"]) for r in rows])
sources = [r["source"] for r in rows]

# Frame-to-frame deltas
deltas = np.abs(np.diff(cam_cx))

# Find the wiggliest sections (rolling window of 30 frames)
window = 30
rolling_mean = np.convolve(deltas, np.ones(window)/window, mode='valid')

print("=== V15 CAMERA JITTER ANALYSIS ===\n")
print(f"Overall: mean_delta={deltas.mean():.1f}, max={deltas.max():.1f}, p95={np.percentile(deltas, 95):.1f}")
print(f"Frames with delta > 15px: {(deltas > 15).sum()}")
print(f"Frames with delta > 10px: {(deltas > 10).sum()}")
print(f"Frames with delta > 5px:  {(deltas > 5).sum()}")

# Find top 5 wiggliest 30-frame windows
top_windows = np.argsort(rolling_mean)[-5:][::-1]
print(f"\nWiggliest 30-frame windows (by mean delta):")
for idx in top_windows:
    src_counts = {}
    for s in sources[idx:idx+window]:
        src_counts[s] = src_counts.get(s, 0) + 1
    src_str = ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items(), key=lambda x: -x[1]))
    print(f"  f{idx}-f{idx+window}: mean_delta={rolling_mean[idx]:.1f} px/f  sources: {src_str}")

# Show the reversal points
print(f"\nCamera reversals (direction changes):")
prev_dir = 0
for i in range(1, len(cam_cx)):
    dx = cam_cx[i] - cam_cx[i-1]
    direction = 1 if dx > 2 else (-1 if dx < -2 else 0)
    if direction != 0 and prev_dir != 0 and direction != prev_dir:
        print(f"  f{i}: cam_cx={cam_cx[i]:.0f}, delta={dx:+.1f}, source={sources[i]}, prev_source={sources[i-1]}")
    if direction != 0:
        prev_dir = direction

# Show camera path in the gap region
print(f"\nGap region f170-f396 camera path (every 10 frames):")
for f in range(170, 400, 10):
    print(f"  f{f}: cam_cx={cam_cx[f]:.0f}, source={sources[f]}")
