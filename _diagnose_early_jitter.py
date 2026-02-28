"""Diagnose jitter in v18 from 0.5s-3s (f15-f90)."""
import csv
import numpy as np

diag = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__smooth_v18.diag.csv"
rows = list(csv.DictReader(open(diag)))
cam_cx = np.array([float(r["cam_cx"]) for r in rows])
crop_x0 = np.array([float(r["crop_x0"]) for r in rows])
crop_w = np.array([float(r["crop_w"]) for r in rows])
crop_y0 = np.array([float(r["crop_y0"]) for r in rows])
crop_h = np.array([float(r["crop_h"]) for r in rows])

print("=== FRAME-BY-FRAME f10-f95 ===\n")
print(f"{'f':>4} {'cam_cx':>8} {'dx':>7} {'crop_x0':>8} {'crop_y0':>8} {'crop_w':>7} {'crop_h':>7} {'zoom':>6}")
prev_cx = cam_cx[9]
for i in range(10, 96):
    dx = cam_cx[i] - prev_cx
    # Approximate zoom from crop_w (smaller crop_w = more zoomed in)
    zoom = 1920.0 / crop_w[i] if crop_w[i] > 0 else 0
    print(f"f{i:>3} {cam_cx[i]:>8.1f} {dx:>+7.1f} {crop_x0[i]:>8.1f} {crop_y0[i]:>8.1f} {crop_w[i]:>7.1f} {crop_h[i]:>7.1f} {zoom:>6.2f}")
    prev_cx = cam_cx[i]

# Check for crop geometry changes (zoom jitter)
print(f"\n\n=== CROP GEOMETRY CHANGES f10-f95 ===\n")
for i in range(11, 96):
    dw = crop_w[i] - crop_w[i-1]
    dh = crop_h[i] - crop_h[i-1]
    dy0 = crop_y0[i] - crop_y0[i-1]
    if abs(dw) > 0.5 or abs(dh) > 0.5 or abs(dy0) > 0.5:
        print(f"  f{i}: dw={dw:+.1f}, dh={dh:+.1f}, dy0={dy0:+.1f}")

# Check acceleration in this region
dx = np.diff(cam_cx[10:96])
accel = np.diff(dx)
print(f"\n\n=== ACCELERATION f10-f95 ===")
print(f"Max |accel|: {np.abs(accel).max():.2f} px/f^2")
print(f"Mean |accel|: {np.abs(accel).mean():.2f} px/f^2")
for i, a in enumerate(accel):
    if abs(a) > 1.5:
        f = i + 11
        print(f"  f{f}: accel={a:+.2f}, vel_before={dx[i]:+.1f}, vel_after={dx[i+1]:+.1f}")
