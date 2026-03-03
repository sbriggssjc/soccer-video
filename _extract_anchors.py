"""Extract camera_x_pct from rendered FINAL clips by template matching
against source frames. Recovers anchor data lost from cleaned-up CSVs."""
import cv2, csv, os, sys
import numpy as np
from pathlib import Path

GAME = "2026-02-23__TSC_vs_Greenwood"
SRC_DIR = Path(rf"D:\Projects\soccer-video\out\atomic_clips\{GAME}")
REEL_DIR = Path(rf"D:\Projects\soccer-video\out\portrait_reels\{GAME}")
TMP_DIR = r"D:\Projects\soccer-video\_tmp"
FPS_SRC = 30
SAMPLE_EVERY = 10  # every 10 source frames = 3fps sampling

CLIP_NUMS = ["001","002","003","004","005",
             "006","007","008","009","010",
             "011","012","013","014","015"]

def extract_clip(clip_num):
    src_file = next(SRC_DIR.glob(f"{clip_num}__*.mp4"))
    final_file = next(REEL_DIR.glob(f"{clip_num}__*FINAL.mp4"))
    
    # Open both videos
    src_cap = cv2.VideoCapture(str(src_file))
    fin_cap = cv2.VideoCapture(str(final_file))
    
    src_w = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_frames = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fin_w = int(fin_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fin_h = int(fin_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fin_frames = int(fin_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fin_fps = fin_cap.get(cv2.CAP_PROP_FPS)
    
    # Crop dimensions (what was originally cropped from source)
    CROP_W = int(src_h * 1080 / 1920)
    CROP_W = CROP_W + (CROP_W % 2)  # 608
    CROP_H = src_h
    
    # Build frame mapping: for each sampled source frame, find corresponding rendered frame
    # Rendered clips are at 24fps (current), source at 30fps
    src_duration = src_frames / FPS_SRC
    
    results = []
    sample_frames = list(range(0, src_frames, SAMPLE_EVERY))

    for src_frame_idx in sample_frames:
        # What time does this source frame correspond to?
        time_s = src_frame_idx / FPS_SRC
        # What rendered frame is closest to this time?
        fin_frame_idx = int(round(time_s * fin_fps))
        if fin_frame_idx >= fin_frames:
            fin_frame_idx = fin_frames - 1
        
        # Read source frame
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
        ret_s, src_frame = src_cap.read()
        if not ret_s:
            continue
        
        # Read rendered frame
        fin_cap.set(cv2.CAP_PROP_POS_FRAMES, fin_frame_idx)
        ret_f, fin_frame = fin_cap.read()
        if not ret_f:
            continue
        
        # Downscale rendered frame from 1080x1920 back to CROP_Wx CROP_H
        fin_resized = cv2.resize(fin_frame, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
        
        # Shrink both for faster matching (half size)
        scale = 0.5
        src_small = cv2.resize(src_frame, None, fx=scale, fy=scale)
        fin_small = cv2.resize(fin_resized, None, fx=scale, fy=scale)

        # Template match: slide the rendered crop across the source frame horizontally
        # Only search horizontally (y is always 0 since full height crop)
        # Use a center strip to avoid vidstab edge artifacts
        margin_y = int(fin_small.shape[0] * 0.1)
        margin_x = int(fin_small.shape[1] * 0.1)
        template = fin_small[margin_y:-margin_y, margin_x:-margin_x]
        search_area = src_small[margin_y:-margin_y, :]
        
        result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # max_loc[0] is the x offset of the template in the search area
        # Adjust for the margin we trimmed from template
        crop_x_small = max_loc[0]  # in half-scale coords
        # The template was trimmed by margin_x on left, so actual crop_x:
        crop_x_small_adjusted = crop_x_small - margin_x  # could be negative, that's ok
        # But actually we want center of crop
        center_x_small = crop_x_small + template.shape[1] / 2
        # Convert back to full scale
        center_x_full = (center_x_small + margin_x) / scale  # account for search area margin
        # Wait, let me reconsider. search_area starts at margin_y but full width.
        # template was cropped margin_x from each side.
        # So max_loc[0] + margin_x = offset of fin_small left edge in src_small
        # center of crop in src_small = max_loc[0] + margin_x + fin_small.shape[1]/2
        crop_center_small = max_loc[0] + margin_x + fin_small.shape[1] / 2
        crop_center_full = crop_center_small / scale

        camera_x_pct = round(crop_center_full / src_w * 100, 1)
        # Clamp to valid range
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
    
    # Write CSV to _tmp (as backup) and Desktop
    csv_path_tmp = os.path.join(TMP_DIR, f"review_{clip_num}.csv")
    csv_path_desk = rf"C:\Users\scott\Desktop\review_{clip_num}.csv"
    
    for csv_path in [csv_path_tmp, csv_path_desk]:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame","time_s","camera_x_pct","confidence","notes"])
            writer.writeheader()
            writer.writerows(results)
    
    # Check quality
    confidences = [r["confidence"] for r in results]
    avg_conf = sum(confidences) / len(confidences)
    low_conf = sum(1 for c in confidences if c < 0.7)
    
    return len(results), avg_conf, low_conf

# --- Main ---
print("Extracting anchor positions from rendered clips...", flush=True)
for cn in CLIP_NUMS:
    try:
        n_anchors, avg_conf, low_conf = extract_clip(cn)
        flag = f" *** {low_conf} LOW CONFIDENCE" if low_conf > 0 else ""
        print(f"{cn}: {n_anchors} anchors, avg confidence={avg_conf:.3f}{flag}", flush=True)
    except Exception as e:
        print(f"{cn}: ERROR - {e}", flush=True)
print("DONE", flush=True)
