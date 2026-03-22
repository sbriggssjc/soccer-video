"""
Analyze YOLO ball detections by extracting 40x40 crops from source video
and computing color statistics to distinguish real ball from red headbands.
"""

import json
import cv2
import numpy as np
from collections import defaultdict

CACHE_PATH = "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.yolov8x.jsonl"
VIDEO_PATH = "out/atomic_clips/2026-02-23__TSC_vs_NEOFC/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
CONF_THRESHOLD = 0.30
CROP_SIZE = 40  # 40x40 pixels

def load_cache(path):
    """Load YOLO ball cache, return list of (frame, x, y, conf)."""
    detections = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frame = rec.get("frame") or rec.get("frame_idx") or rec.get("f")
            conf = rec.get("conf") or rec.get("confidence") or 0.0
            x = rec.get("x") or rec.get("cx") or 0.0
            y = rec.get("y") or rec.get("cy") or 0.0
            if conf >= CONF_THRESHOLD:
                detections.append((int(frame), float(x), float(y), float(conf)))
    detections.sort(key=lambda d: (d[0], -d[3]))
    return detections

def extract_crop(frame_img, cx, cy, size=CROP_SIZE):
    """Extract a size x size crop centered on (cx, cy), clamped to image bounds."""
    h, w = frame_img.shape[:2]
    half = size // 2
    x1 = max(0, int(cx) - half)
    y1 = max(0, int(cy) - half)
    x2 = min(w, int(cx) + half)
    y2 = min(h, int(cy) + half)
    return frame_img[y1:y2, x1:x2]

def analyze_crop(crop):
    """Compute color stats from a BGR crop. Returns dict of stats."""
    if crop.size == 0:
        return None
    # OpenCV loads as BGR
    mean_b = float(np.mean(crop[:, :, 0]))
    mean_g = float(np.mean(crop[:, :, 1]))
    mean_r = float(np.mean(crop[:, :, 2]))
    rg_ratio = mean_r / mean_g if mean_g > 0 else 999.0
    rb_ratio = mean_r / mean_b if mean_b > 0 else 999.0
    is_red = (mean_r > mean_g * 1.3) and (mean_r > mean_b * 1.3)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return {
        "mean_R": mean_r,
        "mean_G": mean_g,
        "mean_B": mean_b,
        "R/G": rg_ratio,
        "R/B": rb_ratio,
        "is_red": is_red,
        "brightness": brightness,
    }

def main():
    print(f"Loading cache: {CACHE_PATH}")
    detections = load_cache(CACHE_PATH)
    print(f"Found {len(detections)} detections with conf >= {CONF_THRESHOLD}")

    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h}, {total_frames} frames, {fps:.2f} fps")
    print()

    # Group detections by frame for efficient seeking
    by_frame = defaultdict(list)
    for frame_idx, x, y, conf in detections:
        by_frame[frame_idx].append((x, y, conf))

    sorted_frames = sorted(by_frame.keys())
    results = []

    print(f"Processing {len(sorted_frames)} unique frames...")
    for i, frame_idx in enumerate(sorted_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_img = cap.read()
        if not ret:
            print(f"  WARNING: Could not read frame {frame_idx}")
            continue

        for (x, y, conf) in by_frame[frame_idx]:
            crop = extract_crop(frame_img, x, y)
            stats = analyze_crop(crop)
            if stats is None:
                continue
            results.append({
                "frame": frame_idx,
                "x": x,
                "y": y,
                "conf": conf,
                **stats,
            })

        if (i + 1) % 100 == 0:
            print(f"  ... processed {i+1}/{len(sorted_frames)} frames")

    cap.release()

    # Print table
    print()
    header = f"{'frame':>6} {'x':>7} {'y':>7} {'conf':>5} {'mean_R':>7} {'mean_G':>7} {'mean_B':>7} {'R/G':>6} {'R/B':>6} {'red?':>5} {'bright':>7}"
    print(header)
    print("-" * len(header))

    for r in results:
        red_str = "YES" if r["is_red"] else "no"
        print(
            f"{r['frame']:>6} "
            f"{r['x']:>7.1f} "
            f"{r['y']:>7.1f} "
            f"{r['conf']:>5.2f} "
            f"{r['mean_R']:>7.1f} "
            f"{r['mean_G']:>7.1f} "
            f"{r['mean_B']:>7.1f} "
            f"{r['R/G']:>6.2f} "
            f"{r['R/B']:>6.2f} "
            f"{red_str:>5} "
            f"{r['brightness']:>7.1f}"
        )

    # Summary
    if not results:
        print("No results to summarize.")
        return

    red_count = sum(1 for r in results if r["is_red"])
    not_red_count = len(results) - red_count
    print()
    print("=" * 60)
    print(f"SUMMARY")
    print(f"  Total detections (conf >= {CONF_THRESHOLD}): {len(results)}")
    print(f"  Red-dominant:     {red_count} ({100*red_count/len(results):.1f}%)")
    print(f"  Not red-dominant: {not_red_count} ({100*not_red_count/len(results):.1f}%)")
    print()

    # Additional breakdown by confidence tier
    print("Breakdown by confidence tier:")
    tiers = [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    for lo, hi in tiers:
        tier_results = [r for r in results if lo <= r["conf"] < hi]
        if not tier_results:
            continue
        tier_red = sum(1 for r in tier_results if r["is_red"])
        tier_not = len(tier_results) - tier_red
        print(f"  conf [{lo:.2f}-{hi:.2f}): {len(tier_results):>4} total, {tier_red:>4} red ({100*tier_red/len(tier_results):.1f}%), {tier_not:>4} not-red")

    # Show average color stats for red vs not-red
    print()
    for label, filt in [("RED-DOMINANT", True), ("NOT-RED", False)]:
        subset = [r for r in results if r["is_red"] == filt]
        if not subset:
            continue
        avg_r = np.mean([r["mean_R"] for r in subset])
        avg_g = np.mean([r["mean_G"] for r in subset])
        avg_b = np.mean([r["mean_B"] for r in subset])
        avg_br = np.mean([r["brightness"] for r in subset])
        avg_conf = np.mean([r["conf"] for r in subset])
        print(f"  {label} avg: R={avg_r:.1f} G={avg_g:.1f} B={avg_b:.1f} brightness={avg_br:.1f} conf={avg_conf:.3f}")

if __name__ == "__main__":
    main()
