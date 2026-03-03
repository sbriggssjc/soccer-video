"""Extract camera_x_pct from NEOFC FINAL clips (zoom=3 vidstab).

Strategy: The FINAL frames have vidstab zoom=3 applied, meaning only the
center ~1/3 of the portrait crop is visible. We extract the center strip
of each FINAL frame, scale it to match source resolution, then do a 
horizontal-only cross-correlation against the source frame.

Uses grayscale + normalized cross-correlation for robustness.
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

# Which clips to extract (all 24fps clips that need redo)
CLIP_NUMS = [f"{i:03d}" for i in range(6, 35)]

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
    fin_frames = int(fin_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fin_fps = fin_cap.get(cv2.CAP_PROP_FPS)
    
    # The portrait crop was 608x1080 from 1920x1080 source
    CROP_W = int(src_h * 1080 / 1920)
    CROP_W = CROP_W + (CROP_W % 2)  # 608
    
    # With zoom=3, vidstab shows center 1/3 of the crop
    # So visible width in source coords ≈ CROP_W / 3 ≈ 203px
    # We'll extract the center vertical strip of the FINAL frame
    # and match it against a horizontal band of the source
    
    # Center strip: take middle 60% vertically, middle 50% horizontally
    # to avoid black borders from vidstab
    vy0 = int(fin_h * 0.2)
    vy1 = int(fin_h * 0.8)
    vx0 = int(fin_w * 0.25)
    vx1 = int(fin_w * 0.75)
    
    # What this center strip corresponds to in source pixels:
    # fin_w=1080 corresponds to CROP_W=608 in source
    # zoom=3 means source visible width = CROP_W/3 ≈ 203
    # So center 50% of fin_w → center 50% of 203 ≈ 101px in source
    vis_w_src = CROP_W / 3.0  # visible width in source coords
    strip_w_src = vis_w_src * 0.5  # center 50%
    strip_h_src = src_h * 0.6  # center 60% height
    
    # Template size in source pixels
    tmpl_w = max(int(strip_w_src), 50)
    tmpl_h = max(int(strip_h_src), 200)
    
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
        
        # Extract center strip from FINAL and resize to source-equivalent size
        fin_strip = fin_frame[vy0:vy1, vx0:vx1]
        fin_resized = cv2.resize(fin_strip, (tmpl_w, tmpl_h), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for matching
        tmpl_gray = cv2.cvtColor(fin_resized, cv2.COLOR_BGR2GRAY)
        
        # Source search band: same vertical center strip
        sy0 = int(src_h * 0.2)
        sy1 = int(src_h * 0.8)
        src_band = src_frame[sy0:sy1, :]
        src_band_resized = cv2.resize(src_band, (src_w, tmpl_h), interpolation=cv2.INTER_AREA)
        src_gray = cv2.cvtColor(src_band_resized, cv2.COLOR_BGR2GRAY)
        
        # Template match
        result = cv2.matchTemplate(src_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # max_loc[0] = left edge of template in source
        # Center of visible area in source:
        center_x = max_loc[0] + tmpl_w / 2
        
        # But the template represents the CENTER of the zoom=3 crop
        # So center_x IS the center of the original crop
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
    csv_path_tmp = os.path.join(TMP_DIR, f"review_{PREFIX}{clip_num}.csv")
    csv_path_desk = rf"C:\Users\scott\Desktop\review_{PREFIX}{clip_num}.csv"
    
    for csv_path in [csv_path_tmp, csv_path_desk]:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame","time_s","camera_x_pct","confidence","notes"])
            writer.writeheader()
            writer.writerows(results)
    
    confidences = [r["confidence"] for r in results]
    avg_conf = sum(confidences) / len(confidences)
    low_conf = sum(1 for c in confidences if c < 0.5)
    
    return len(results), avg_conf, low_conf

# --- Main ---
print("Extracting anchor positions (v2, zoom-aware)...", flush=True)
for cn in CLIP_NUMS:
    try:
        n_anchors, avg_conf, low_conf = extract_clip(cn)
        flag = f" *** {low_conf} LOW" if low_conf > 0 else ""
        print(f"{cn}: {n_anchors} anchors, avg conf={avg_conf:.3f}{flag}", flush=True)
    except Exception as e:
        print(f"{cn}: ERROR - {e}", flush=True)
print("DONE", flush=True)
