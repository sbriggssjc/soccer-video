"""Extract camera_x_pct from NEOFC FINAL clips using ORB feature matching.

ORB features are scale-invariant, so zoom=3 vidstab distortion should not
prevent matching. We detect keypoints in both source and FINAL frames,
match them, and use the spatial offset to determine where in the source
the FINAL frame's content came from.
"""
import cv2, csv, os, sys
import numpy as np
from pathlib import Path

GAME = "2026-02-23__TSC_vs_NEOFC"
SRC_DIR = Path(rf"D:\Projects\soccer-video\out\atomic_clips\{GAME}")
REEL_DIR = Path(rf"D:\Projects\soccer-video\out\portrait_reels\{GAME}")
TMP_DIR = r"D:\Projects\soccer-video\_tmp"
PREFIX = "neofc_"
FPS_SRC = 30
SAMPLE_EVERY = 15  # sample every 15th source frame
CLIP_NUMS = [f"{i:03d}" for i in range(6, 35)]

# ORB config
ORB_FEATURES = 2000
MATCH_RATIO = 0.75  # Lowe's ratio test threshold


def extract_clip(clip_num):
    src_file = next(SRC_DIR.glob(f"{clip_num}__*.mp4"))
    final_file = next(REEL_DIR.glob(f"{clip_num}__*FINAL.mp4"))

    src_cap = cv2.VideoCapture(str(src_file))
    fin_cap = cv2.VideoCapture(str(final_file))

    src_w = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 1920
    src_h = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 1080
    src_frames = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fin_w = int(fin_cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 1080
    fin_h = int(fin_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 1920
    fin_fps = fin_cap.get(cv2.CAP_PROP_FPS)

    # The original crop width in the source frame
    CROP_W = int(src_h * 1080 / 1920)  # 608 (portrait aspect from 1080p)
    CROP_W = CROP_W + (CROP_W % 2)

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    results = []
    sample_frames = list(range(0, src_frames, SAMPLE_EVERY))

    for src_frame_idx in sample_frames:
        time_s = src_frame_idx / FPS_SRC
        fin_frame_idx = int(round(time_s * fin_fps))
        fin_frames = int(fin_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fin_frame_idx >= fin_frames:
            fin_frame_idx = fin_frames - 1

        # Read source frame
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
        ret_s, src_frame = src_cap.read()
        if not ret_s:
            continue

        # Read FINAL frame
        fin_cap.set(cv2.CAP_PROP_POS_FRAMES, fin_frame_idx)
        ret_f, fin_frame = fin_cap.read()
        if not ret_f:
            continue

        # Convert to grayscale
        src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
        fin_gray = cv2.cvtColor(fin_frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB features
        kp_src, des_src = orb.detectAndCompute(src_gray, None)
        kp_fin, des_fin = orb.detectAndCompute(fin_gray, None)

        if des_src is None or des_fin is None or len(kp_src) < 10 or len(kp_fin) < 10:
            continue

        # Match with knnMatch + Lowe's ratio test
        matches = bf.knnMatch(des_fin, des_src, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < MATCH_RATIO * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 5:
            continue

        # For each good match, compute the x-position in source that corresponds
        # to the center of the FINAL frame.
        #
        # The FINAL frame is 1080 wide. The original crop from source was CROP_W=608 wide.
        # vidstab zoom=3 means the visible area is about 1/3 of the stabilized frame.
        # But the stabilized frame was 1080 wide (portrait), so visible ≈ 360 px of
        # the 1080-wide output. The 1080 output came from the 608-wide crop.
        #
        # Actually, let's think about this differently:
        # A keypoint at position (fx, fy) in the FINAL frame corresponds to
        # some position (sx, sy) in the source frame (matched via ORB).
        # The camera_x_pct is defined as the center of the crop window in the source.
        # We need to figure out the x-center of the crop from the matched source positions.
        #
        # Simple approach: collect all matched source x-positions and take the median
        # as the crop center estimate.

        src_xs = []
        for m in good_matches:
            src_pt = kp_src[m.trainIdx].pt  # (x, y) in source
            src_xs.append(src_pt[0])

        if not src_xs:
            continue

        # The median of matched source x positions is our best estimate
        # of where the crop center was
        center_x = float(np.median(src_xs))
        camera_x_pct = round(center_x / src_w * 100, 1)
        camera_x_pct = max(0, min(100, camera_x_pct))

        # Confidence: use fraction of good matches relative to total
        confidence = round(len(good_matches) / max(len(matches), 1), 3)

        results.append({
            "frame": src_frame_idx,
            "time_s": round(time_s, 1),
            "camera_x_pct": camera_x_pct,
            "confidence": confidence,
            "n_matches": len(good_matches),
            "notes": ""
        })

    src_cap.release()
    fin_cap.release()

    # Write CSVs
    csv_tmp = os.path.join(TMP_DIR, f"review_{PREFIX}{clip_num}.csv")
    csv_desk = rf"C:\Users\scott\Desktop\review_{PREFIX}{clip_num}.csv"
    fieldnames = ["frame", "time_s", "camera_x_pct", "confidence", "n_matches", "notes"]
    for p in [csv_tmp, csv_desk]:
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

    confs = [r["confidence"] for r in results]
    n_matches_list = [r["n_matches"] for r in results]
    avg_c = sum(confs) / len(confs) if confs else 0
    avg_m = sum(n_matches_list) / len(n_matches_list) if n_matches_list else 0
    return len(results), avg_c, avg_m


print("Extracting anchors v4 (ORB feature matching)...", flush=True)
for cn in CLIP_NUMS:
    try:
        n, avg_c, avg_m = extract_clip(cn)
        print(f"{cn}: {n} anchors, avg conf={avg_c:.3f}, avg matches={avg_m:.0f}", flush=True)
    except Exception as e:
        print(f"{cn}: ERROR - {e}", flush=True)
print("DONE", flush=True)
