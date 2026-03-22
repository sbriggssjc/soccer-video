"""V18: Re-render portrait frames using the smoothed diag CSV camera path.
Reads crop geometry from the smoothed diag, extracts portrait frames, stitches with ffmpeg.
"""
import csv, os, sys
import cv2
import numpy as np

SRC_VIDEO = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
DIAG = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__smooth_v18.diag.csv"
FRAME_DIR = r"D:\Projects\soccer-video\out\_scratch\v18_frames"
OUT_VIDEO = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__smooth_v18.mp4"

PORTRAIT_W = 1080
PORTRAIT_H = 1920
FPS = 30

def main():
    # Read diag
    with open(DIAG) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} frames from diag")

    # Open source video
    cap = cv2.VideoCapture(SRC_VIDEO)
    if not cap.isOpened():
        print("ERROR: Cannot open source video")
        return
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source: {src_w}x{src_h}, {total} frames")

    # Create frame output dir
    os.makedirs(FRAME_DIR, exist_ok=True)

    # Process each frame
    for i, row in enumerate(rows):
        ret, frame = cap.read()
        if not ret:
            print(f"ERROR: Could not read frame {i}")
            break

        crop_x0 = float(row["crop_x0"])
        crop_y0 = float(row["crop_y0"])
        crop_w = float(row["crop_w"])
        crop_h = float(row["crop_h"])

        # Clamp crop to valid bounds
        crop_x0 = max(0, min(crop_x0, src_w - crop_w))
        crop_y0 = max(0, min(crop_y0, src_h - crop_h))

        # Extract crop region
        x0 = int(round(crop_x0))
        y0 = int(round(crop_y0))
        x1 = int(round(crop_x0 + crop_w))
        y1 = int(round(crop_y0 + crop_h))

        # Clamp to frame bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(src_w, x1)
        y1 = min(src_h, y1)

        crop = frame[y0:y1, x0:x1]

        # Resize to portrait dimensions
        portrait = cv2.resize(crop, (PORTRAIT_W, PORTRAIT_H), interpolation=cv2.INTER_LANCZOS4)

        # Save frame
        out_path = os.path.join(FRAME_DIR, f"frame_{i:06d}.png")
        cv2.imwrite(out_path, portrait)

        if i % 50 == 0:
            print(f"  Frame {i}/{len(rows)}: crop=({x0},{y0})-({x1},{y1})")

    cap.release()
    print(f"\nRendered {len(rows)} frames to {FRAME_DIR}")

    # Stitch with ffmpeg
    print(f"\nStitching with ffmpeg...")
    ffmpeg_cmd = (
        f'ffmpeg -y -framerate {FPS} '
        f'-i "{FRAME_DIR}\\frame_%06d.png" '
        f'-c:v libx264 -preset medium -crf 17 '
        f'-pix_fmt yuv420p '
        f'-g 96 '
        f'"{OUT_VIDEO}"'
    )
    print(f"  {ffmpeg_cmd}")
    ret = os.system(ffmpeg_cmd)
    if ret == 0:
        print(f"\n[DONE] Video saved to {OUT_VIDEO}")
    else:
        print(f"\n[ERROR] ffmpeg returned {ret}")

if __name__ == "__main__":
    main()
