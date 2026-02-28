"""
World-Space Ball Path Portrait Renderer v5
============================================
Strategy:
  1. Use camera motion analysis to get world coordinates
  2. Interpolate ball position in WORLD space between confirmed YOLO anchors
     (+ tracker anchors for extra coverage)
  3. Convert world ball positions back to frame coordinates
  4. Set portrait crop center = frame ball position (clamped)
  5. Light smoothing, then render via ffmpeg pipes
"""
import json, pickle, math, time, subprocess, sys
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator

base = Path(r"D:\Projects\soccer-video")
src_video = base / r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
OUT_VIDEO = base / r"out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__worldpath_v5.mp4"
DIAG_CSV  = base / r"out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__worldpath_v5.diag.csv"

SRC_W, SRC_H = 1920, 1080
CROP_W, CROP_H = 608, 1080
OUT_W, OUT_H = 1080, 1920
FPS_OUT = 24
SMOOTH_SIGMA = 3.0  # slightly more smoothing for smooth panning

def main():
    t0 = time.time()
    
    # ── 1. Load camera analysis data ────────────────────────────────
    with open(base / "_camera_analysis.pkl", "rb") as f:
        analysis = pickle.load(f)
    
    cam_cum_x = analysis["cam_cum_x"]
    cam_cum_y = analysis["cam_cum_y"]
    yolo = analysis["yolo"]           # {frame: (x, y, conf)}
    yolo_world = analysis["yolo_world"]  # {frame: (world_x, world_y, conf)}
    tracker = analysis["tracker"]     # {frame: (x, y)}
    tracker_world = analysis["tracker_world"]  # {frame: (world_x, world_y)}
    n_frames = analysis["n_frames"]
    
    print(f"[LOAD] {n_frames} frames, {len(yolo)} YOLO, {len(tracker)} tracker")
    
    # ── 2. Build anchor points in world space ───────────────────────
    # Combine YOLO (high confidence) and tracker (medium confidence)
    # YOLO takes priority when both exist for same frame
    anchors = {}  # frame -> (world_x, world_y, weight)
    
    for fr, (wx, wy) in tracker_world.items():
        anchors[fr] = (wx, wy, 0.5)  # tracker = weight 0.5
    
    for fr, (wx, wy, conf) in yolo_world.items():
        anchors[fr] = (wx, wy, 1.0)  # YOLO = weight 1.0
    
    anchor_frames = sorted(anchors.keys())
    print(f"[ANCHORS] {len(anchor_frames)} total ({len(yolo)} YOLO + "
          f"{len(set(tracker.keys()) - set(yolo.keys()))} tracker-only)")
    print(f"  First anchor: f{anchor_frames[0]}, Last: f{anchor_frames[-1]}")
    print(f"  Gaps > 20 frames:")
    for i in range(1, len(anchor_frames)):
        gap = anchor_frames[i] - anchor_frames[i-1]
        if gap > 20:
            print(f"    f{anchor_frames[i-1]} -> f{anchor_frames[i]}: {gap} frames")

    # ── 3. Interpolate ball position in world space ─────────────────
    # Use PCHIP (monotonic cubic) for smooth interpolation
    ax = np.array([anchors[f][0] for f in anchor_frames])
    ay = np.array([anchors[f][1] for f in anchor_frames])
    af = np.array(anchor_frames, dtype=float)
    
    # PCHIP interpolation (handles non-monotonic data well)
    interp_x = PchipInterpolator(af, ax)
    interp_y = PchipInterpolator(af, ay)
    
    all_frames = np.arange(n_frames, dtype=float)
    world_ball_x = np.zeros(n_frames)
    world_ball_y = np.zeros(n_frames)
    
    # For frames within anchor range, interpolate
    first_anchor = anchor_frames[0]
    last_anchor = anchor_frames[-1]
    
    for fr in range(n_frames):
        if fr < first_anchor:
            # Before first anchor: hold first anchor position
            world_ball_x[fr] = ax[0]
            world_ball_y[fr] = ay[0]
        elif fr > last_anchor:
            # After last anchor: hold last anchor position
            world_ball_x[fr] = ax[-1]
            world_ball_y[fr] = ay[-1]
        else:
            world_ball_x[fr] = interp_x(fr)
            world_ball_y[fr] = interp_y(fr)
    
    # ── 4. Convert world coordinates back to frame coordinates ──────
    frame_ball_x = world_ball_x - cam_cum_x[:n_frames]
    frame_ball_y = world_ball_y - cam_cum_y[:n_frames]
    
    print(f"\n[WORLD PATH] Ball world X range: [{world_ball_x.min():.0f}, {world_ball_x.max():.0f}]")
    print(f"[FRAME PATH] Ball frame X range: [{frame_ball_x.min():.0f}, {frame_ball_x.max():.0f}]")
    
    # Show frame ball positions every 30 frames
    print(f"\n  Frame ball X every 30 frames:")
    for fr in range(0, n_frames, 30):
        yl = " YOLO" if fr in yolo else (" TRKR" if fr in tracker else "")
        print(f"    f{fr:3d}: frame_x={frame_ball_x[fr]:7.1f}  world_x={world_ball_x[fr]:7.1f}"
              f"  cam_x={cam_cum_x[fr]:7.1f}{yl}")

    # ── 5. Compute crop centers ─────────────────────────────────────
    half_w = CROP_W / 2.0
    half_h = CROP_H / 2.0
    
    # Crop center = ball frame position (clamped)
    crop_cx = np.clip(frame_ball_x, half_w, SRC_W - half_w)
    crop_cy = np.full(n_frames, SRC_H / 2.0)  # fixed vertical center
    
    # Light Gaussian smoothing
    crop_cx = gaussian_filter1d(crop_cx, sigma=SMOOTH_SIGMA, mode="nearest")
    crop_cy = gaussian_filter1d(crop_cy, sigma=SMOOTH_SIGMA, mode="nearest")
    crop_cx = np.clip(crop_cx, half_w, SRC_W - half_w)
    crop_cy = np.clip(crop_cy, half_h, SRC_H - half_h)
    
    x0 = np.round(crop_cx - half_w).astype(int)
    y0 = np.round(crop_cy - half_h).astype(int)
    x0 = np.clip(x0, 0, SRC_W - CROP_W)
    y0 = np.clip(y0, 0, SRC_H - CROP_H)
    
    # ── 6. Diagnostics ──────────────────────────────────────────────
    n_bic = 0
    n_yolo_bic = 0
    max_escape = 0.0
    escape_frames = []
    
    with open(DIAG_CSV, "w") as df:
        df.write("frame,ball_frame_x,ball_frame_y,ball_world_x,crop_cx,x0,y0,ball_in_crop,escape_px,is_yolo\n")
        for fr in range(n_frames):
            bfx = float(frame_ball_x[fr])
            bfy = float(frame_ball_y[fr])
            bwx = float(world_ball_x[fr])
            cx_v = float(crop_cx[fr])
            x0f, y0f = int(x0[fr]), int(y0[fr])
            
            # Check if the YOLO-confirmed ball is in crop (only for YOLO frames)
            is_yolo = fr in yolo
            if is_yolo:
                ybx = yolo[fr][0]  # use actual YOLO frame x, not interpolated
                in_x = (x0f <= ybx <= x0f + CROP_W)
                bic = in_x
                if bic:
                    n_yolo_bic += 1
                esc_x = max(0, x0f - ybx, ybx - (x0f + CROP_W))
                esc = esc_x
                if esc > 0:
                    escape_frames.append((fr, esc))
                max_escape = max(max_escape, esc)
            else:
                bic = True  # non-YOLO frames counted as OK
                esc = 0.0
            
            n_bic += int(bic)
            df.write(f"{fr},{bfx:.1f},{bfy:.1f},{bwx:.1f},{cx_v:.1f},{x0f},{y0f},{int(bic)},{esc:.1f},{int(is_yolo)}\n")
    
    print(f"\n[DIAG] YOLO ball-in-crop: {n_yolo_bic}/{len(yolo)} ({100*n_yolo_bic/len(yolo):.1f}%)")
    print(f"[DIAG] YOLO escape frames: {len(escape_frames)}")
    if escape_frames:
        worst5 = sorted(escape_frames, key=lambda x: -x[1])[:5]
        for efr, ed in worst5:
            print(f"        f{efr}: {ed:.1f}px (ball_frame_x={yolo[efr][0]:.1f})")
    print(f"[DIAG] Crop cx range: [{crop_cx.min():.1f}, {crop_cx.max():.1f}]")

    # ── 7. Render via ffmpeg pipes ──────────────────────────────────
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    frame_size = SRC_W * SRC_H * 3
    
    decode_cmd = [
        "ffmpeg", "-y", "-i", str(src_video),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "error", "pipe:1"
    ]
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{CROP_W}x{CROP_H}", "-r", str(FPS_OUT),
        "-i", "pipe:0",
        "-vf", f"scale={OUT_W}:{OUT_H}:flags=lanczos",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-v", "error", str(OUT_VIDEO)
    ]
    
    print(f"\n[RENDER] Starting pipe render ({n_frames} frames)...")
    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    fr = 0
    try:
        while True:
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(SRC_H, SRC_W, 3)
            idx = min(fr, n_frames - 1)
            cx0, cy0 = int(x0[idx]), int(y0[idx])
            crop = frame[cy0:cy0+CROP_H, cx0:cx0+CROP_W]
            if crop.shape[0] != CROP_H or crop.shape[1] != CROP_W:
                padded = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
                h = min(crop.shape[0], CROP_H)
                w = min(crop.shape[1], CROP_W)
                padded[:h, :w] = crop[:h, :w]
                crop = padded
            encoder.stdin.write(crop.tobytes())
            fr += 1
            if fr % 100 == 0:
                print(f"  ... frame {fr}/{n_frames}")
    except BrokenPipeError:
        print(f"[WARN] Encoder pipe closed at frame {fr}")
    except Exception as e:
        print(f"[ERROR] Frame {fr}: {e}")
    finally:
        decoder.stdout.close()
        try: encoder.stdin.close()
        except: pass
        decoder.wait()
        enc_rc = encoder.wait()
    
    elapsed = time.time() - t0
    out_size = OUT_VIDEO.stat().st_size / (1024*1024) if OUT_VIDEO.exists() else 0
    print(f"\n[RENDER] Done: {fr} frames in {elapsed:.1f}s")
    print(f"[RENDER] Output: {OUT_VIDEO} ({out_size:.1f} MB)")
    print(f"[RENDER] Encoder exit code: {enc_rc}")

if __name__ == "__main__":
    main()
