"""Extract camera_x_pct from NEOFC FINAL clips using 1D scanline matching.

Strategy: vidstab zoom=3 shows center 1/3 of the portrait crop.
Take narrow horizontal bands from the VERY center of the FINAL frame,
compress them 3x to source resolution, and do 1D cross-correlation
against the same scanlines in the source frame. This avoids 2D
template matching issues and vidstab border artifacts.
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
SAMPLE_EVERY = 15  # match filmstrip CSV sampling

CLIP_NUMS = [f"{i:03d}" for i in range(6, 35)]

def extract_clip(clip_num):
    src_file = next(SRC_DIR.glob(f"{clip_num}__*.mp4"))
    final_file = next(REEL_DIR.glob(f"{clip_num}__*FINAL.mp4"))

    src_cap = cv2.VideoCapture(str(src_file))
    fin_cap = cv2.VideoCapture(str(final_file))

    src_w = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_frames = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fin_w = int(fin_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1080
    fin_h = int(fin_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 1920
    fin_frames = int(fin_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fin_fps = fin_cap.get(cv2.CAP_PROP_FPS)

    # Portrait crop was 608x1080 from 1920x1080
    CROP_W = int(src_h * 1080 / 1920)
    CROP_W = CROP_W + (CROP_W % 2)  # 608

    # zoom=3: visible width in source coords = CROP_W / 3 ≈ 203px
    vis_w_src = CROP_W / 3.0

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

        # Extract center horizontal band from FINAL (avoid vidstab edges)
        # Take rows from 45%-55% of height (center 10%)
        fy0 = int(fin_h * 0.45)
        fy1 = int(fin_h * 0.55)
        fin_band = fin_frame[fy0:fy1, :, :]  # shape: (~192, 1080, 3)

        # Compress horizontally by 3x to match source resolution
        # fin_w=1080 -> 1080/3=360... but we need source-scale
        # Actually: fin_w corresponds to CROP_W in source
        # So compress fin_band from 1080 wide to CROP_W (608) wide
        # Then the visible center 1/3 maps to ~203px
        # But zoom=3 means fin_w shows only 1/3 of CROP_W
        # So fin_w pixels = vis_w_src source pixels
        # Compress from fin_w to int(vis_w_src)
        target_w = int(vis_w_src)
        band_h = fy1 - fy0

        # Corresponding source rows (same relative position)
        # fin 45%-55% of 1920 -> in source coords, center of 1080 height
        # zoom=3 vertically too: visible height = 1080/3 = 360
        # center 10% of 1920 = rows 864-1056 in fin
        # corresponds to center 10% of visible 360 in source = 36px
        # at center of 1080 = rows 522-558
        sy0 = int(src_h * 0.45)
        sy1 = int(src_h * 0.55)
        src_band = src_frame[sy0:sy1, :, :]  # full width source band

        # Resize source band height to match
        src_band_r = cv2.resize(src_band, (src_w, band_h))

        # Resize FINAL band to source-scale width
        fin_band_r = cv2.resize(fin_band, (target_w, band_h))

        # Convert to grayscale for correlation
        src_gray = cv2.cvtColor(src_band_r, cv2.COLOR_BGR2GRAY).astype(np.float32)
        fin_gray = cv2.cvtColor(fin_band_r, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1D-ish template match: slide fin_gray across src_gray horizontally
        result = cv2.matchTemplate(src_gray, fin_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # max_loc[0] = left edge of the visible region in source
        # Center of visible region:
        center_x = max_loc[0] + target_w / 2.0
        camera_x_pct = round(center_x / src_w * 100, 1)
        camera_x_pct = max(0, min(100, camera_x_pct))

        results.append({
            "frame": src_frame_idx,
            "time_s": round(time_s, 1),
            "camera_x_pct": camera_x_pct,
            "confidence": round(max_val, 3),
            "notes": ""
        })

    src_cap.release()
    fin_cap.release()

    # Write CSV
    csv_tmp = os.path.join(TMP_DIR, f"review_{PREFIX}{clip_num}.csv")
    csv_desk = rf"C:\Users\scott\Desktop\review_{PREFIX}{clip_num}.csv"
    for p in [csv_tmp, csv_desk]:
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame","time_s","camera_x_pct","confidence","notes"])
            w.writeheader()
            w.writerows(results)

    confs = [r["confidence"] for r in results]
    avg_c = sum(confs) / len(confs) if confs else 0
    low = sum(1 for c in confs if c < 0.5)
    return len(results), avg_c, low

print("Extracting anchors v3 (1D scanline match)...", flush=True)
for cn in CLIP_NUMS:
    try:
        n, avg, low = extract_clip(cn)
        flag = f" *** {low} LOW" if low > 0 else ""
        print(f"{cn}: {n} anchors, avg conf={avg:.3f}{flag}", flush=True)
    except Exception as e:
        print(f"{cn}: ERROR - {e}", flush=True)
print("DONE", flush=True)
