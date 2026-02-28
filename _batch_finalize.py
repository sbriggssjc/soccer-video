"""Batch post-process all portrait reels: vidstab + 4K upscale.

For each *__portrait.mp4 in the game folder:
  1. Skip if *__portrait__FINAL.mp4 already exists
  2. Apply ffmpeg vidstab (2-pass: detect + transform)
  3. Apply 2x lanczos upscale with light denoise + sharpen
  4. Rename to *__portrait__FINAL.mp4

Usage:
  python _batch_finalize.py
"""
import os, subprocess, sys, time, glob, re

REEL_DIR = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
TEMP_DIR = r"D:\Projects\soccer-video\_tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

# vidstab params (same as v19)
VIDSTAB_SHAKINESS = 5
VIDSTAB_ACCURACY = 15
VIDSTAB_SMOOTHING = 15
VIDSTAB_ZOOM = 3

def probe_resolution(path):
    p = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height',
         '-of', 'csv=p=0:s=x', path],
        capture_output=True, text=True
    )
    parts = (p.stdout or "").strip().split("x")
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return 0, 0

def vidstab(src, dst, clip_idx):
    """Two-pass vidstab stabilization."""
    # Use short numeric filename for trf to avoid ffmpeg filter path parsing issues
    trf = os.path.join(TEMP_DIR, f"clip{clip_idx:03d}.trf").replace("\\", "/")
    src_fwd = src.replace("\\", "/")
    dst_fwd = dst.replace("\\", "/")
    # Pass 1: detect
    subprocess.run([
        'ffmpeg', '-y', '-i', src_fwd,
        '-vf', f'vidstabdetect=shakiness={VIDSTAB_SHAKINESS}:accuracy={VIDSTAB_ACCURACY}:result={trf}',
        '-f', 'null', '-'
    ], capture_output=True)
    # Pass 2: transform
    subprocess.check_call([
        'ffmpeg', '-y', '-i', src_fwd,
        '-vf', f'vidstabtransform=input={trf}:smoothing={VIDSTAB_SMOOTHING}:interpol=bicubic:crop=black:zoom={VIDSTAB_ZOOM}',
        '-c:v', 'libx264', '-crf', '17', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', dst_fwd
    ])
    # Cleanup trf
    trf_win = trf.replace("/", "\\")
    if os.path.exists(trf_win):
        os.remove(trf_win)

def upscale_4k(src, dst):
    """2x lanczos upscale with denoise + sharpen."""
    subprocess.check_call([
        'ffmpeg', '-y', '-i', src.replace("\\", "/"),
        '-vf', 'scale=iw*2:ih*2:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
        '-pix_fmt', 'yuv420p', dst.replace("\\", "/")
    ])

def process_clip(portrait_mp4, clip_idx=0):
    """Process one portrait clip through vidstab + 4K upscale."""
    base = portrait_mp4.replace("__portrait.mp4", "")
    final_mp4 = base + "__portrait__FINAL.mp4"

    if os.path.exists(final_mp4):
        w, h = probe_resolution(final_mp4)
        if w >= 2000:
            print(f"  SKIP (already final {w}x{h}): {os.path.basename(final_mp4)}")
            return "skip"

    clip_name = os.path.basename(portrait_mp4)
    print(f"\n{'='*60}")
    print(f"  Processing: {clip_name}")

    t0 = time.time()

    # Step 1: vidstab
    stab_tmp = os.path.join(TEMP_DIR, f"stab_{clip_idx:03d}.mp4")
    print(f"  [1/2] Vidstab stabilization...")
    try:
        vidstab(portrait_mp4, stab_tmp, clip_idx)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR in vidstab: {e}")
        return "error"

    # Step 2: 4K upscale
    upscale_tmp = os.path.join(TEMP_DIR, f"upscale_{clip_idx:03d}.mp4")
    print(f"  [2/2] 4K lanczos upscale...")
    try:
        upscale_4k(stab_tmp, upscale_tmp)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR in upscale: {e}")
        return "error"

    # Move upscaled file to final location
    import shutil
    shutil.move(upscale_tmp, final_mp4)

    # Cleanup temp
    if os.path.exists(stab_tmp):
        os.remove(stab_tmp)

    w, h = probe_resolution(final_mp4)
    elapsed = time.time() - t0
    sz_mb = os.path.getsize(final_mp4) / (1024 * 1024)
    print(f"  DONE: {w}x{h}, {sz_mb:.1f} MB, {elapsed:.0f}s")
    return "ok"

def main():
    # Find all __portrait.mp4 files
    pattern = os.path.join(REEL_DIR, "*__portrait.mp4")
    portraits = sorted(glob.glob(pattern))

    print(f"Found {len(portraits)} portrait clips to process")
    print(f"Output dir: {REEL_DIR}")
    print()

    results = {"ok": 0, "skip": 0, "error": 0}
    total_t0 = time.time()

    for i, p in enumerate(portraits, 1):
        print(f"\n[{i}/{len(portraits)}]", end="")
        result = process_clip(p, clip_idx=i)
        results[result] = results.get(result, 0) + 1

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  OK: {results['ok']}, Skipped: {results['skip']}, Errors: {results['error']}")

    # Cleanup temp dir
    try:
        os.rmdir(TEMP_DIR)
    except OSError:
        pass

if __name__ == "__main__":
    main()
