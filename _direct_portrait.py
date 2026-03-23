"""Direct portrait cropper v3 — crop-position stabilization, no edge blur.

Based on the version Scott said was "nearly perfect":
  - camera_x_pct as primary target (smooth auto-follow positions)
  - ball_x_pct as fallback for frames without camera data
  - Stabilization via crop position shift (no warpAffine = no edge blur)
  - Max correction capped to prevent overcorrection

Usage:
  python _direct_portrait.py                  # all clips
  python _direct_portrait.py --clip 1         # single clip
  python _direct_portrait.py --smooth 4       # tighter follow
"""
import os, csv, sys, argparse, subprocess
import cv2
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

# ===== CONFIGURE =====
GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
PREFIX = "neofc2017_"
TARGET_W = 1080
TARGET_H = 1920
CRF = 17
FPS_OUT = 24
STAB_RADIUS = 60
MAX_STAB_PX = 40  # max correction in pixels — prevents overcorrection
# ======================

os.chdir(r"D:\Projects\soccer-video")

ap = argparse.ArgumentParser()
ap.add_argument("--clip", type=int, default=None)
ap.add_argument("--smooth", type=float, default=8.0,
                help="Gaussian sigma for camera path smoothing")
ap.add_argument("--fps", type=int, default=FPS_OUT)
args = ap.parse_args()

clips_dir = Path(f"out/atomic_clips/{GAME}")
out_dir = Path(f"out/portrait_reels/{GAME}")
out_dir.mkdir(parents=True, exist_ok=True)
desktop = Path(r"C:\Users\scott\Desktop")

clip_files = sorted(clips_dir.glob("*.mp4"),
                    key=lambda p: int(p.stem.split("__")[0]))


def load_review_csv(clip_num):
    """Load review CSV. Uses camera_x_pct first (smooth), ball_x_pct as fallback."""
    csv_path = desktop / f"review_{PREFIX}{clip_num}.csv"
    if not csv_path.exists():
        return []
    anchors = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            cam = row.get("camera_x_pct", "").strip()
            ball = row.get("ball_x_pct", "").strip()
            if cam:
                anchors.append((frame, float(cam)))
            elif ball:
                anchors.append((frame, float(ball)))
    return anchors


def build_camera_path(anchors, total_frames, src_w, smooth_sigma):
    """PCHIP interpolation + gentle Gaussian smooth. Returns per-frame cx in pixels."""
    if not anchors:
        return np.full(total_frames, src_w / 2.0)
    frames = np.array([a[0] for a in anchors])
    cx_vals = np.array([a[1] / 100.0 * src_w for a in anchors])
    if frames[0] > 0:
        frames = np.insert(frames, 0, 0)
        cx_vals = np.insert(cx_vals, 0, cx_vals[0])
    if frames[-1] < total_frames - 1:
        frames = np.append(frames, total_frames - 1)
        cx_vals = np.append(cx_vals, cx_vals[-1])
    interp = PchipInterpolator(frames, cx_vals)
    cx_path = interp(np.arange(total_frames))
    if smooth_sigma > 0:
        cx_path = gaussian_filter1d(cx_path, sigma=smooth_sigma)
    return cx_path


def detect_shake(frames_gray):
    """Detect frame-to-frame camera shake via phase correlation."""
    n = len(frames_gray)
    dx = np.zeros(n)
    dy = np.zeros(n)
    for i in range(1, n):
        shift, _ = cv2.phaseCorrelate(
            frames_gray[i - 1].astype(np.float64),
            frames_gray[i].astype(np.float64)
        )
        dx[i] = shift[0]
        dy[i] = shift[1]
    return np.cumsum(dx), np.cumsum(dy)


def smooth_trajectory(traj, radius):
    n = len(traj)
    out = np.copy(traj)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = np.mean(traj[lo:hi])
    return out


def render_clip(clip_file, clip_num, smooth_sigma):
    """Render clip to portrait with crop-position stabilization."""
    anchors = load_review_csv(clip_num)

    cap = cv2.VideoCapture(str(clip_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Source: {src_w}x{src_h} @ {src_fps:.0f}fps, {total_frames} frames")
    print(f"  Anchors: {len(anchors)} positions")

    cx_path = build_camera_path(anchors, total_frames, src_w, smooth_sigma)
    crop_w = int(src_h * 9.0 / 16.0)  # 607
    half_crop = crop_w / 2.0

    # === PASS 1: Read all frames, detect shake ===
    print(f"  Pass 1: reading frames...")
    all_frames = []
    all_grays = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        small = cv2.resize(frame, (480, 270))
        all_grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cap.release()
    actual = len(all_frames)

    traj_x, traj_y = detect_shake(all_grays)
    del all_grays

    smooth_x = smooth_trajectory(traj_x, STAB_RADIUS)
    smooth_y = smooth_trajectory(traj_y, STAB_RADIUS)

    # Correction in source pixels, CAPPED to prevent overcorrection
    scale = src_w / 480.0
    corr_x = np.clip((smooth_x - traj_x) * scale, -MAX_STAB_PX, MAX_STAB_PX)
    corr_y = np.clip((smooth_y - traj_y) * scale, -MAX_STAB_PX, MAX_STAB_PX)

    max_corr = max(np.max(np.abs(corr_x)), np.max(np.abs(corr_y)))
    print(f"  Stabilization: max correction {max_corr:.1f}px (capped at {MAX_STAB_PX}px)")

    # === Build final crop positions: ball follow + shake correction ===
    # Combine camera path with capped stabilization correction
    final_cx = np.array([
        cx_path[min(i, len(cx_path) - 1)] + corr_x[min(i, len(corr_x) - 1)]
        for i in range(actual)
    ])
    # Final heavy smooth pass to kill any remaining micro-jitter
    final_cx = gaussian_filter1d(final_cx, sigma=28.0)
    # Clamp to valid crop range
    final_cx = np.clip(final_cx, half_crop, src_w - half_crop)

    # === PASS 2: Crop with final smoothed positions ===
    print(f"  Pass 2: cropping {actual} frames...")
    tmp_dir = Path(f"out/_scratch/direct_portrait/{clip_num}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(actual):
        x0 = int(round(final_cx[i] - half_crop))
        x0 = max(0, min(x0, src_w - crop_w))

        cropped = all_frames[i][0:src_h, x0:x0 + crop_w]
        resized = cv2.resize(cropped, (TARGET_W, TARGET_H),
                             interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(tmp_dir / f"frame_{i:06d}.png"), resized)

    del all_frames
    print(f"  Wrote {actual} frames")

    # === ENCODE ===
    out_path = out_dir / f"{clip_file.stem}_portrait_FINAL.mp4"
    tmp_out = out_dir / f".tmp.{clip_file.stem}_portrait_FINAL.mp4"

    print(f"  Encoding...")
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-framerate", str(int(src_fps)),
        "-i", str(tmp_dir / "frame_%06d.png"),
        "-r", str(args.fps),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(tmp_out)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        if out_path.exists():
            out_path.unlink()
        tmp_out.rename(out_path)
        for f in tmp_dir.glob("*.png"):
            f.unlink()
        print(f"  -> OK: {out_path.name}")
        return True
    else:
        print(f"  -> FAILED: {result.stderr.decode('utf-8','replace')[-500:]}")
        return False


# === MAIN ===
rendered = 0
failed = 0

for clip_file in clip_files:
    stem = clip_file.stem
    clip_num = int(stem.split("__")[0])
    if args.clip is not None and clip_num != args.clip:
        continue
    print(f"\n{'='*60}")
    print(f"Clip {clip_num}: {clip_file.name}")
    print(f"{'='*60}")
    if render_clip(clip_file, str(clip_num), args.smooth):
        rendered += 1
    else:
        failed += 1

print(f"\n{'='*60}")
print(f"DONE: {rendered} rendered, {failed} failed")
print(f"Settings: smooth={args.smooth}, stab_radius={STAB_RADIUS}, max_stab={MAX_STAB_PX}px")
print(f"Output: {out_dir}")
print(f"{'='*60}")
