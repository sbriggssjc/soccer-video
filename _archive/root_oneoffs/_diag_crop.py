"""Diagnostic: extract frames from source + FINAL clips to compare crop positions.
Saves annotated images to outputs folder for visual inspection.
"""
import cv2, csv, os
import numpy as np
from pathlib import Path

GAME = "2026-02-23__TSC_vs_Greenwood"
SRC_DIR = Path(f"D:/Projects/soccer-video/out/atomic_clips/{GAME}")
REEL_DIR = Path(f"D:/Projects/soccer-video/out/portrait_reels/{GAME}")
OUT_DIR = Path("D:/Projects/soccer-video/_tmp/diag_crops")
OUT_DIR.mkdir(exist_ok=True)

SRC_W = 1920; SRC_H = 1080
CROP_W = 608; CROP_H = 1080
FPS = 30

# Check clips that had high duration ratio (frame sync likely worst)
TEST_CLIPS = ["003", "009", "012", "017", "018"]

for clip_num in TEST_CLIPS:
    src_file = next(SRC_DIR.glob(f"{clip_num}__*.mp4"))
    final_file = next(REEL_DIR.glob(f"{clip_num}__*FINAL.mp4"))
    
    src_cap = cv2.VideoCapture(str(src_file))
    fin_cap = cv2.VideoCapture(str(final_file))
    
    src_frames = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fin_frames = int(fin_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fin_fps = fin_cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n=== Clip {clip_num} ===")
    print(f"Source: {src_frames} frames, FINAL: {fin_frames} frames @ {fin_fps}fps")
    
    # Sample 3 time points: 25%, 50%, 75% through the clip
    for pct in [0.25, 0.50, 0.75]:
        src_idx = int(src_frames * pct)
        fin_idx = int(fin_frames * pct)
        time_s = src_idx / FPS
        
        # Read source frame
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
        ret_s, src_frame = src_cap.read()
        
        # Read FINAL frame and scale back to crop dimensions for comparison
        fin_cap.set(cv2.CAP_PROP_POS_FRAMES, fin_idx)
        ret_f, fin_frame = fin_cap.read()
        
        if not ret_s or not ret_f:
            print(f"  {pct:.0%}: Failed to read frames")
            continue
        
        # Downscale FINAL from 1080x1920 to 608x1080 (original crop size)
        fin_resized = cv2.resize(fin_frame, (CROP_W, CROP_H))
        
        # Template match to find where the crop is in the source
        # Use center strip to avoid vidstab artifacts
        margin = int(CROP_H * 0.15)
        template = fin_resized[margin:-margin, 20:-20]  # trim edges
        search = src_frame[margin:-margin, :]
        
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = result[0], *cv2.minMaxLoc(result)[1:]
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Crop x position in source
        crop_x = max_loc[0] - 20  # adjust for template trim
        crop_center = crop_x + CROP_W / 2
        crop_pct = crop_center / SRC_W * 100
        
        # Draw the crop rectangle on source frame
        annotated = src_frame.copy()
        cv2.rectangle(annotated, (max(0,crop_x), 0), (min(SRC_W, crop_x+CROP_W), SRC_H), 
                      (0, 255, 0), 3)
        cv2.putText(annotated, f"Clip {clip_num} | frame {src_idx} | t={time_s:.1f}s | crop@{crop_pct:.0f}% | conf={max_val:.3f}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save annotated source frame (scaled down for size)
        out_path = OUT_DIR / f"{clip_num}_f{src_idx}_src_annotated.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Also save the FINAL frame for comparison
        out_path2 = OUT_DIR / f"{clip_num}_f{fin_idx}_final.jpg"
        cv2.imwrite(str(out_path2), fin_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        print(f"  {pct:.0%}: src_f={src_idx}, fin_f={fin_idx}, crop_pct={crop_pct:.1f}%, conf={max_val:.3f}")
    
    src_cap.release()
    fin_cap.release()

print("\nDiagnostic images saved to:", OUT_DIR)
