"""Extract camera_x_pct using phase correlation - more robust to vidstab distortion.
For each sampled frame, slides all possible crops and uses MSE to find best match."""
import cv2, csv, os, sys
import numpy as np
from pathlib import Path

GAME = "2026-02-23__TSC_vs_Greenwood"
SRC_DIR = Path(rf"D:\Projects\soccer-video\out\atomic_clips\{GAME}")
REEL_DIR = Path(rf"D:\Projects\soccer-video\out\portrait_reels\{GAME}")
TMP_DIR = r"D:\Projects\soccer-video\_tmp"
FPS_SRC = 30
SAMPLE_EVERY = 10  # 3fps sampling

CLIP_NUMS = ["001","002","003","004","005",
             "006","007","008","009","010",
             "011","012","013","014","015"]

def extract_clip(clip_num):
    src_file = next(SRC_DIR.glob(f"{clip_num}__*.mp4"))
    final_file = next(REEL_DIR.glob(f"{clip_num}__*FINAL.mp4"))
    
    src_cap = cv2.VideoCapture(str(src_file))
    fin_cap = cv2.VideoCapture(str(final_file))
    
    src_w = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 1920
    src_h = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 1080
    src_frames = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fin_fps = fin_cap.get(cv2.CAP_PROP_FPS)
    fin_frames = int(fin_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    CROP_W = int(src_h * 1080 / 1920)
    CROP_W = CROP_W + (CROP_W % 2)  # 608
    
    results = []
    sample_frames = list(range(0, src_frames, SAMPLE_EVERY))
    
    for src_frame_idx in sample_frames:
        time_s = src_frame_idx / FPS_SRC
        fin_frame_idx = int(round(time_s * fin_fps))
        if fin_frame_idx >= fin_frames:
            fin_frame_idx = fin_frames - 1

        src_cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
        ret_s, src_frame = src_cap.read()
        if not ret_s: continue
        
        fin_cap.set(cv2.CAP_PROP_POS_FRAMES, fin_frame_idx)
        ret_f, fin_frame = fin_cap.read()
        if not ret_f: continue
        
        # Convert rendered back to crop size
        fin_crop = cv2.resize(fin_frame, (CROP_W, src_h), interpolation=cv2.INTER_AREA)
        
        # Use center 60% vertically and 60% horizontally to avoid vidstab edges
        cy1 = int(src_h * 0.2)
        cy2 = int(src_h * 0.8)
        cx1 = int(CROP_W * 0.2)
        cx2 = int(CROP_W * 0.8)
        template = fin_crop[cy1:cy2, cx1:cx2]
        
        # Convert to grayscale for matching
        tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.cvtColor(src_frame[cy1:cy2, :], cv2.COLOR_BGR2GRAY)
        
        # matchTemplate with CCORR_NORMED (more robust than CCOEFF)
        result = cv2.matchTemplate(src_gray, tmpl_gray, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # max_loc[0] = left edge of template match in src_gray
        # Template was extracted starting at cx1 into the crop
        # So: crop_left + cx1 = max_loc[0]
        # crop_left = max_loc[0] - cx1
        crop_left = max_loc[0] - cx1
        crop_center = crop_left + CROP_W / 2
        camera_x_pct = round(crop_center / src_w * 100, 1)
        camera_x_pct = max(0, min(100, camera_x_pct))
        
        results.append({
            "frame": src_frame_idx,
            "time_s": round(time_s, 1),
            "camera_x_pct": camera_x_pct,
            "confidence": round(max_val, 4),
            "notes": ""
        })
    
    src_cap.release()
    fin_cap.release()
    
    # Save to _tmp (backup) and Desktop
    for csv_path in [os.path.join(TMP_DIR, f"review_{clip_num}.csv"),
                     rf"C:\Users\scott\Desktop\review_{clip_num}.csv"]:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame","time_s","camera_x_pct","confidence","notes"])
            w.writeheader()
            w.writerows(results)
    
    confs = [r["confidence"] for r in results]
    avg_c = sum(confs)/len(confs) if confs else 0
    
    # Check smoothness: average absolute delta between consecutive anchors
    pcts = [r["camera_x_pct"] for r in results]
    deltas = [abs(pcts[i]-pcts[i-1]) for i in range(1,len(pcts))]
    avg_delta = sum(deltas)/len(deltas) if deltas else 0
    max_delta = max(deltas) if deltas else 0
    
    return len(results), avg_c, avg_delta, max_delta

print("Extracting anchors v2 (center crop matching)...", flush=True)
for cn in CLIP_NUMS:
    try:
        n, conf, avg_d, max_d = extract_clip(cn)
        print(f"{cn}: {n} anchors, conf={conf:.4f}, avg_delta={avg_d:.1f}%, max_delta={max_d:.1f}%", flush=True)
    except Exception as e:
        print(f"{cn}: ERROR - {e}", flush=True)
        import traceback; traceback.print_exc()
print("DONE", flush=True)
