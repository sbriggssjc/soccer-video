"""
Visual audit: extract frames from source video, draw optical flow 
ball position as a red circle, and save annotated frames.
Covers every 30 frames (~1 second intervals) plus YOLO anchor frames.
"""
import json, subprocess, sys
import numpy as np
from pathlib import Path

base = Path(r"D:\Projects\soccer-video")
src = base / r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
of_path = base / "_optflow_track/optflow_ball_path_v3.jsonl"
out_dir = base / "out" / "visual_audit"
out_dir.mkdir(parents=True, exist_ok=True)

# Load optical flow positions
ofv3 = {}
with open(of_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        if rec.get("_meta"): continue
        ofv3[int(rec["frame"])] = (float(rec["cx"]), float(rec["cy"]))

# Load YOLO detections
yolo_path = base / "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl.orig_backup"
yolo = {}
with open(yolo_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        if rec.get("_meta"): continue
        yolo[int(rec["frame"])] = (float(rec["cx"]), float(rec["cy"]), float(rec.get("conf",0)))

# Frames to extract: every 30 frames + all YOLO frames
target_frames = set(range(0, 496, 30))
target_frames.update(yolo.keys())
target_frames = sorted(target_frames)

print(f"Extracting {len(target_frames)} frames for visual audit...")

# Decode all frames
SRC_W, SRC_H = 1920, 1080
frame_bytes = SRC_W * SRC_H * 3

decode_cmd = [
    "ffmpeg", "-y", "-i", str(src),
    "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-v", "error", "pipe:1"
]
decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

target_set = set(target_frames)
fr = 0
saved = 0

try:
    while True:
        raw = decoder.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        
        if fr in target_set:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(SRC_H, SRC_W, 3).copy()
            
            # Draw optical flow position as red filled circle
            if fr in ofv3:
                bx, by = ofv3[fr]
                bx_i, by_i = int(round(bx)), int(round(by))
                # Draw circle radius 15
                for dy in range(-15, 16):
                    for dx in range(-15, 16):
                        if dx*dx + dy*dy <= 15*15:
                            py, px = by_i + dy, bx_i + dx
                            if 0 <= py < SRC_H and 0 <= px < SRC_W:
                                frame[py, px] = [255, 0, 0]  # Red
                # Draw crosshair lines (20px each direction)
                for d in range(-25, 26):
                    py = by_i + d
                    if 0 <= py < SRC_H and 0 <= bx_i < SRC_W:
                        frame[py, bx_i] = [255, 0, 0]
                    px = bx_i + d
                    if 0 <= by_i < SRC_H and 0 <= px < SRC_W:
                        frame[by_i, px] = [255, 0, 0]
            
            # Draw YOLO position as green circle if available
            if fr in yolo:
                yx, yy, yc = yolo[fr]
                yx_i, yy_i = int(round(yx)), int(round(yy))
                for dy in range(-12, 13):
                    for dx in range(-12, 13):
                        if 10*10 <= dx*dx + dy*dy <= 12*12:  # ring
                            py, px = yy_i + dy, yx_i + dx
                            if 0 <= py < SRC_H and 0 <= px < SRC_W:
                                frame[py, px] = [0, 255, 0]  # Green ring
            
            # Save as JPEG (small for viewing)
            # Use PIL or just save raw and convert
            from PIL import Image
            img = Image.fromarray(frame)
            # Resize to 640x360 for manageability
            img = img.resize((640, 360), Image.LANCZOS)
            
            # Add frame label
            label = f"f{fr:03d}"
            if fr in yolo:
                label += f" YOLO"
            
            fname = out_dir / f"audit_f{fr:04d}.jpg"
            img.save(fname, quality=75)
            saved += 1
            
            if saved % 10 == 0:
                print(f"  saved {saved}/{len(target_frames)} frames...")
        
        fr += 1

except Exception as e:
    print(f"Error at frame {fr}: {e}")
    import traceback; traceback.print_exc()
finally:
    decoder.stdout.close()
    decoder.wait()

print(f"Done: saved {saved} annotated frames to {out_dir}")
print("Red circle+crosshair = optical flow position")
print("Green ring = YOLO detection")
