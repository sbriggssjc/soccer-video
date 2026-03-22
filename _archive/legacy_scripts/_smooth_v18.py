"""V18: Post-process v17 camera path to eliminate jitter.
Apply velocity cap + Gaussian smooth to the camera cx path,
then overwrite the diag CSV so we can re-render frames from it.
"""
import csv, os, shutil
import numpy as np
from scipy.ndimage import gaussian_filter1d

DIAG_IN = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__manual_v17.diag.csv"
DIAG_OUT = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__smooth_v18.diag.csv"

MAX_VELOCITY = 12.0   # px/frame cap
SMOOTH_SIGMA = 5.0    # additional Gaussian smooth after velocity cap
ITERATIONS = 3        # apply velocity cap iteratively for convergence

def velocity_cap(cx, max_vel, iterations=3):
    """Iteratively cap velocity while preserving trajectory shape."""
    result = cx.copy()
    for iteration in range(iterations):
        changed = 0
        # Forward pass: limit how fast we can move from left to right
        for i in range(1, len(result)):
            dx = result[i] - result[i-1]
            if abs(dx) > max_vel:
                result[i] = result[i-1] + np.sign(dx) * max_vel
                changed += 1
        # Backward pass: limit from right to left
        for i in range(len(result)-2, -1, -1):
            dx = result[i] - result[i+1]
            if abs(dx) > max_vel:
                old = result[i]
                result[i] = result[i+1] + np.sign(dx) * max_vel
                # Average with forward pass result to avoid bias
                result[i] = (result[i] + old) / 2.0
                changed += 1
        print(f"  Iteration {iteration+1}: {changed} frames adjusted")
        if changed == 0:
            break
    return result

def main():
    # Read diag CSV
    with open(DIAG_IN) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    cam_cx = np.array([float(r["cam_cx"]) for r in rows])
    original_cx = cam_cx.copy()

    print(f"Loaded {len(rows)} frames from v17 diag")
    dx_before = np.abs(np.diff(cam_cx))
    print(f"Before: mean_vel={dx_before.mean():.1f}, max_vel={dx_before.max():.1f}, p95={np.percentile(dx_before, 95):.1f}")

    # Step 1: Velocity cap
    print(f"\nApplying velocity cap at {MAX_VELOCITY} px/f:")
    cam_cx = velocity_cap(cam_cx, MAX_VELOCITY, ITERATIONS)

    # Step 2: Additional Gaussian smooth
    print(f"\nApplying Gaussian smooth (sigma={SMOOTH_SIGMA})...")
    cam_cx = gaussian_filter1d(cam_cx, sigma=SMOOTH_SIGMA, mode='nearest')

    # Step 3: Clamp to valid range
    src_width = 1920.0
    for i, r in enumerate(rows):
        crop_w = float(r["crop_w"])
        half_w = crop_w / 2.0
        cam_cx[i] = np.clip(cam_cx[i], half_w, src_width - half_w)

    dx_after = np.abs(np.diff(cam_cx))
    print(f"\nAfter: mean_vel={dx_after.mean():.1f}, max_vel={dx_after.max():.1f}, p95={np.percentile(dx_after, 95):.1f}")

    # Count reversals
    prev_dir = 0
    rev_before = 0
    rev_after = 0
    dx_orig = np.diff(original_cx)
    dx_new = np.diff(cam_cx)
    for arr, label in [(dx_orig, "before"), (dx_new, "after")]:
        prev_dir = 0
        rev = 0
        for d in arr:
            direction = 1 if d > 2 else (-1 if d < -2 else 0)
            if direction != 0 and prev_dir != 0 and direction != prev_dir:
                rev += 1
            if direction != 0:
                prev_dir = direction
        print(f"Reversals {label}: {rev}")

    # Update rows with smoothed camera path
    for i, r in enumerate(rows):
        crop_w = float(r["crop_w"])
        half_w = crop_w / 2.0
        r["cam_cx"] = f"{cam_cx[i]:.1f}"
        new_x0 = cam_cx[i] - half_w
        new_x0 = max(0.0, min(new_x0, src_width - crop_w))
        r["crop_x0"] = f"{new_x0:.1f}"

    # Write output diag
    with open(DIAG_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved smoothed diag to {os.path.basename(DIAG_OUT)}")

if __name__ == "__main__":
    main()
