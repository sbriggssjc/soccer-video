"""Extract camera_x_pct from NEOFC FINAL clips using ORB + RANSAC homography.

Instead of just taking median x of matched features, we use RANSAC to find
the actual geometric transform between FINAL and source, which properly
handles the scale change from zoom=3 vidstab.

The homography maps FINAL pixel coords -> source pixel coords.
We map the center of the FINAL frame to source coords to get anchor_x.
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
SAMPLE_EVERY = 15
CLIP_NUMS = [f"{i:03d}" for i in range(6, 35)]

ORB_FEATURES = 5000
RANSAC_THRESH = 5.0


def extract_clip(clip_num):
    src_file = next(SRC_DIR.glob(f"{clip_num}__*.mp4"))
    final_file = next(REEL_DIR.glob(f"{clip_num}__*FINAL.mp4"))

    src_cap = cv2.VideoCapture(str(src_file))
    fin_cap = cv2.VideoCapture(str(final_file))

    src_w = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_frames = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fin_w = int(fin_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fin_h = int(fin_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fin_fps = fin_cap.get(cv2.CAP_PROP_FPS)
    fin_frames = int(fin_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    CROP_W = int(src_h * 1080 / 1920)
    CROP_W = CROP_W + (CROP_W % 2)

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    results = []
    sample_frames = list(range(0, src_frames, SAMPLE_EVERY))

    for src_frame_idx in sample_frames:
        time_s = src_frame_idx / FPS_SRC
        fin_frame_idx = int(round(time_s * fin_fps))
        if fin_frame_idx >= fin_frames:
            fin_frame_idx = fin_frames - 1

        src_cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
        ret_s, src_frame = src_cap.read()
        if not ret_s:
            continue

        fin_cap.set(cv2.CAP_PROP_POS_FRAMES, fin_frame_idx)
        ret_f, fin_frame = fin_cap.read()
        if not ret_f:
            continue

        src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
        fin_gray = cv2.cvtColor(fin_frame, cv2.COLOR_BGR2GRAY)

        kp_src, des_src = orb.detectAndCompute(src_gray, None)
        kp_fin, des_fin = orb.detectAndCompute(fin_gray, None)

        if des_src is None or des_fin is None or len(kp_src) < 10 or len(kp_fin) < 10:
            continue

        # knnMatch + ratio test
        matches = bf.knnMatch(des_fin, des_src, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 8:
            results.append({
                "frame": src_frame_idx, "time_s": round(time_s, 1),
                "camera_x_pct": -1, "inliers": 0,
                "total_good": len(good), "notes": "too few matches"
            })
            continue

        pts_fin = np.float32([kp_fin[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_src = np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # RANSAC homography: maps FINAL coords -> source coords
        H, mask = cv2.findHomography(pts_fin, pts_src, cv2.RANSAC, RANSAC_THRESH)

        if H is None:
            results.append({
                "frame": src_frame_idx, "time_s": round(time_s, 1),
                "camera_x_pct": -1, "inliers": 0,
                "total_good": len(good), "notes": "homography failed"
            })
            continue

        inliers = int(mask.sum()) if mask is not None else 0

        # Map the center of the FINAL frame to source coordinates
        fin_center = np.float32([[fin_w / 2, fin_h / 2]]).reshape(-1, 1, 2)
        src_center = cv2.perspectiveTransform(fin_center, H)
        src_cx = float(src_center[0, 0, 0])
        src_cy = float(src_center[0, 0, 1])

        # Sanity check: src_cx should be within source frame
        if src_cx < 0 or src_cx > src_w or src_cy < 0 or src_cy > src_h:
            results.append({
                "frame": src_frame_idx, "time_s": round(time_s, 1),
                "camera_x_pct": -1, "inliers": inliers,
                "total_good": len(good), "notes": f"out of bounds ({src_cx:.0f},{src_cy:.0f})"
            })
            continue

        camera_x_pct = round(src_cx / src_w * 100, 1)
        camera_x_pct = max(0, min(100, camera_x_pct))

        results.append({
            "frame": src_frame_idx, "time_s": round(time_s, 1),
            "camera_x_pct": camera_x_pct, "inliers": inliers,
            "total_good": len(good), "notes": ""
        })

    src_cap.release()
    fin_cap.release()

    # Write CSVs
    csv_tmp = os.path.join(TMP_DIR, f"review_{PREFIX}{clip_num}.csv")
    csv_desk = rf"C:\Users\scott\Desktop\review_{PREFIX}{clip_num}.csv"
    fieldnames = ["frame", "time_s", "camera_x_pct", "inliers", "total_good", "notes"]
    for p in [csv_tmp, csv_desk]:
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

    valid = [r for r in results if r["camera_x_pct"] >= 0]
    avg_inliers = sum(r["inliers"] for r in valid) / len(valid) if valid else 0
    failed = len(results) - len(valid)
    return len(results), len(valid), avg_inliers, failed


print("Extracting anchors v5 (ORB + RANSAC homography)...", flush=True)
for cn in CLIP_NUMS:
    try:
        total, valid, avg_inl, failed = extract_clip(cn)
        flag = f" *** {failed} FAILED" if failed > 0 else ""
        print(f"{cn}: {valid}/{total} valid, avg inliers={avg_inl:.0f}{flag}", flush=True)
    except Exception as e:
        print(f"{cn}: ERROR - {e}", flush=True)
print("DONE", flush=True)
