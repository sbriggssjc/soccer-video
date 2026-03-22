"""Show velocity profile and find all acceleration/deceleration spikes."""
import csv
import numpy as np

diag = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__manual_v17.diag.csv"
rows = list(csv.DictReader(open(diag)))
cam_cx = np.array([float(r["cam_cx"]) for r in rows])
sources = [r["source"] for r in rows]

dx = np.diff(cam_cx)
velocity = dx  # signed velocity
abs_vel = np.abs(dx)
accel = np.diff(velocity)  # acceleration

print("=== VELOCITY PROFILE (every 10 frames) ===\n")
print(f"{'Frame':>6} {'cam_cx':>8} {'vel':>8} {'src':>8}")
for f in range(0, len(cam_cx), 10):
    v = f"{dx[f-1]:+.1f}" if f > 0 else "  ---"
    print(f"f{f:>5} {cam_cx[f]:>8.0f} {v:>8} {sources[f]:>8}")

print(f"\n\n=== ALL FRAMES WITH |velocity| > 15 px/f ===\n")
for i in range(len(dx)):
    if abs(dx[i]) > 15:
        print(f"  f{i}-f{i+1}: vel={dx[i]:+.1f} px/f, src_from={sources[i]}, src_to={sources[i+1]}")

print(f"\n\n=== ACCELERATION SPIKES (|accel| > 5 px/f^2) ===\n")
spike_count = 0
for i in range(len(accel)):
    if abs(accel[i]) > 5:
        spike_count += 1
        if spike_count <= 20:
            print(f"  f{i+1}: accel={accel[i]:+.1f} px/f^2, vel_before={dx[i]:+.1f}, vel_after={dx[i+1]:+.1f}, src={sources[i+1]}")
print(f"\nTotal acceleration spikes: {spike_count}")

# What would a max velocity cap achieve?
for cap in [20, 15, 12, 10, 8]:
    violations = (abs_vel > cap).sum()
    pct = violations / len(abs_vel) * 100
    print(f"  Cap at {cap} px/f: {violations} frames ({pct:.1f}%) would be affected")
