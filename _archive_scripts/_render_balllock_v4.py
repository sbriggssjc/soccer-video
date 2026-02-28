"""
Ball-Lock Portrait Renderer v4
===============================
Bypasses render_follow_unified's 3 layers of smoothing
(post-plan Gaussian σ=18, Legacy Gaussian σ=6, speed limiter).

Instead:
  1. Read optical-flow v3 ball positions (pixel-accurate)
  2. Camera center = ball center (clamped to keep crop in frame)
  3. Apply MINIMAL Gaussian smoothing (σ=2.0 → ~0.08s at 24fps)
  4. Use ffmpeg to extract portrait crops frame-by-frame via pipes

Result: ball is ALWAYS centered in frame (within clamp limits).
"""

import json, os, subprocess, sys, math, time
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# ── CONFIG ──────────────────────────────────────────────────────────
SRC_VIDEO  = r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
BALL_PATH  = r"_optflow_track\optflow_ball_path_v3.jsonl"
OUT_VIDEO  = r"out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__balllock_v4.mp4"
DIAG_CSV   = r"out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__balllock_v4.diag.csv"

SRC_W, SRC_H = 1920, 1080
CROP_W, CROP_H = 608, 1080   # portrait aspect crop from source
OUT_W, OUT_H = 1080, 1920     # final output resolution
FPS_OUT = 24

SMOOTH_SIGMA = 2.0   # very light: ~0.08s at 24fps
VPOS = 0.55          # ball at 55% from top (slight look-ahead below ball)


def main():
    t0 = time.time()
    base = Path(__file__).resolve().parent
    src_video = base / SRC_VIDEO
    ball_path = base / BALL_PATH
    out_video = base / OUT_VIDEO
    diag_csv  = base / DIAG_CSV
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ball positions ──────────────────────────────────────
    ball_frames = {}
    with open(ball_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                continue
            fr = int(rec["frame"])
            bx = float(rec["cx"])
            by = float(rec["cy"])
            ball_frames[fr] = (bx, by)

    n_frames = max(ball_frames.keys()) + 1
    print(f"[LOAD] {len(ball_frames)} ball positions, {n_frames} total frames")

    # ── 2. Build raw camera centers ─────────────────────────────────
    half_w = CROP_W / 2.0
    half_h = CROP_H / 2.0
    raw_cx = np.zeros(n_frames, dtype=np.float64)
    raw_cy = np.zeros(n_frames, dtype=np.float64)

    last_bx, last_by = SRC_W / 2.0, SRC_H / 2.0
    for fr in range(n_frames):
        if fr in ball_frames:
            last_bx, last_by = ball_frames[fr]
        raw_cx[fr] = last_bx
        raw_cy[fr] = last_by - (VPOS * CROP_H - half_h)

    raw_cx = np.clip(raw_cx, half_w, SRC_W - half_w)
    raw_cy = np.clip(raw_cy, half_h, SRC_H - half_h)

    # ── 3. Light Gaussian smoothing ─────────────────────────────────
    smooth_cx = gaussian_filter1d(raw_cx, sigma=SMOOTH_SIGMA, mode="nearest")
    smooth_cy = gaussian_filter1d(raw_cy, sigma=SMOOTH_SIGMA, mode="nearest")
    smooth_cx = np.clip(smooth_cx, half_w, SRC_W - half_w)
    smooth_cy = np.clip(smooth_cy, half_h, SRC_H - half_h)

    # ── 4. Crop top-left corners ────────────────────────────────────
    x0 = np.round(smooth_cx - half_w).astype(int)
    y0 = np.round(smooth_cy - half_h).astype(int)
    x0 = np.clip(x0, 0, SRC_W - CROP_W)
    y0 = np.clip(y0, 0, SRC_H - CROP_H)

    # ── 5. Diagnostics ──────────────────────────────────────────────
    n_bic = 0
    max_escape = 0.0
    escape_frames = []
    with open(diag_csv, "w") as df:
        df.write("frame,ball_x,ball_y,cam_cx,cam_cy,x0,y0,ball_in_crop,escape_px\n")
        for fr in range(n_frames):
            bx, by = ball_frames.get(fr, (np.nan, np.nan))
            cx_v, cy_v = float(smooth_cx[fr]), float(smooth_cy[fr])
            x0f, y0f = int(x0[fr]), int(y0[fr])
            if not np.isnan(bx):
                in_x = (x0f <= bx <= x0f + CROP_W)
                in_y = (y0f <= by <= y0f + CROP_H)
                bic = in_x and in_y
                if bic:
                    n_bic += 1
                esc_x = max(0, x0f - bx, bx - (x0f + CROP_W))
                esc_y = max(0, y0f - by, by - (y0f + CROP_H))
                esc = math.sqrt(esc_x**2 + esc_y**2)
                if esc > 0:
                    escape_frames.append((fr, esc))
                max_escape = max(max_escape, esc)
            else:
                bic = False
                esc = 0.0
            df.write(f"{fr},{bx:.1f},{by:.1f},{cx_v:.1f},{cy_v:.1f},{x0f},{y0f},{int(bic)},{esc:.1f}\n")

    bic_pct = 100.0 * n_bic / n_frames
    print(f"[DIAG] Ball in crop: {n_bic}/{n_frames} ({bic_pct:.1f}%)")
    print(f"[DIAG] Max escape: {max_escape:.1f}px")
    print(f"[DIAG] Escape frames: {len(escape_frames)}")
    if escape_frames:
        worst5 = sorted(escape_frames, key=lambda x: -x[1])[:5]
        for efr, ed in worst5:
            print(f"        f{efr}: {ed:.1f}px")
    print(f"[DIAG] Camera cx range: [{smooth_cx.min():.1f}, {smooth_cx.max():.1f}]")

    # ── 6. Render via ffmpeg pipes ──────────────────────────────────
    # Decoder → raw frames → Python crops → Encoder (with scaling)
    frame_size = SRC_W * SRC_H * 3
    crop_size = CROP_W * CROP_H * 3

    decode_cmd = [
        "ffmpeg", "-y",
        "-i", str(src_video),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "error",
        "pipe:1"
    ]

    # Encoder accepts crop-sized frames and scales to output resolution
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{CROP_W}x{CROP_H}",
        "-r", str(FPS_OUT),
        "-i", "pipe:0",
        "-vf", f"scale={OUT_W}:{OUT_H}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-v", "error",
        str(out_video)
    ]

    print(f"[RENDER] Starting pipe render ({n_frames} frames)...")
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
            cx0 = int(x0[idx])
            cy0 = int(y0[idx])

            # Crop the portrait region
            crop = frame[cy0:cy0+CROP_H, cx0:cx0+CROP_W]

            # Ensure exact size (edge case safety)
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
        try:
            encoder.stdin.close()
        except:
            pass
        decoder.wait()
        enc_rc = encoder.wait()

    elapsed = time.time() - t0
    out_size = out_video.stat().st_size / (1024*1024) if out_video.exists() else 0
    print(f"[RENDER] Done: {fr} frames in {elapsed:.1f}s")
    print(f"[RENDER] Output: {out_video} ({out_size:.1f} MB)")
    print(f"[RENDER] Encoder exit code: {enc_rc}")


if __name__ == "__main__":
    main()
