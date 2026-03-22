"""Batch render + finalize all 20 clips for 2026-02-21 TSC vs Greenwood.
Full pipeline per clip: portrait render → vidstab → 4K upscale → copy to clean.
"""
import subprocess, sys, os, time, traceback, tempfile, shutil
from pathlib import Path

os.chdir(r"D:\Projects\soccer-video")
RESULT = r"D:\Projects\soccer-video\_tmp\batch_greenwood_0221_result.txt"
os.makedirs(os.path.dirname(RESULT), exist_ok=True)
TEMP_DIR = tempfile.gettempdir()

GAME = "2026-02-21__TSC_vs_Greenwood"
CLIPS_DIR = Path(f"out/atomic_clips/{GAME}")
REELS_DIR = Path(f"out/portrait_reels/{GAME}")
CLEAN_DIR = Path("out/portrait_reels/clean")
PYTHON = sys.executable
FFMPEG = "ffmpeg"

def log(msg):
    with open(RESULT, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

def process_clip(clip_path, clip_num):
    """Run full pipeline for one clip. Returns (success, elapsed)."""
    stem = clip_path.stem
    portrait = REELS_DIR / f"{stem}__portrait.mp4"
    final = REELS_DIR / f"{stem}__portrait__FINAL.mp4"
    cinematic = f"{stem}__CINEMATIC_portrait_FINAL.mp4"
    clip_log = REELS_DIR / f"{clip_num}__render.log"

    t0 = time.time()

    # Step 1: Portrait render
    log(f"  [{clip_num}] Step 1: Portrait render...")
    cmd = [
        PYTHON, "tools/render_follow_unified.py",
        "--in", str(clip_path),
        "--src", str(clip_path),
        "--out", str(portrait),
        "--portrait", "1080x1920",
        "--preset", "cinematic",
        "--fps", "24",
        "--diagnostics",
        "--use-ball-telemetry",
        "--keep-scratch",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    with open(clip_log, "w") as f:
        f.write(result.stdout or "")
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if result.returncode != 0:
        log(f"  [{clip_num}] FAIL render rc={result.returncode}")
        return False, time.time() - t0

    pmb = portrait.stat().st_size / (1024*1024) if portrait.exists() else 0
    log(f"  [{clip_num}] Step 1 done: {time.time()-t0:.0f}s, {pmb:.1f} MB")

    # Step 2: vidstab detect
    t1 = time.time()
    trf = f"gw0221_{clip_num}_transforms.trf"
    r1 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], capture_output=True, text=True, timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0:
        log(f"  [{clip_num}] FAIL vidstab detect")
        return False, time.time() - t0
    log(f"  [{clip_num}] Step 2 done: {time.time()-t1:.0f}s")

    # Step 3: vidstab transform
    t1 = time.time()
    stab_tmp = os.path.join(TEMP_DIR, f"gw0221_{clip_num}_stab.mp4")
    r2 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom=3",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k", stab_tmp],
        capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r2.returncode != 0:
        log(f"  [{clip_num}] FAIL vidstab transform")
        return False, time.time() - t0
    log(f"  [{clip_num}] Step 3 done: {time.time()-t1:.0f}s")

    # Step 4: 4K upscale
    t1 = time.time()
    final_abs = str(final.resolve())
    if os.path.exists(final_abs):
        os.remove(final_abs)
    r3 = subprocess.run([FFMPEG, "-y", "-i", stab_tmp, "-vf",
        "scale=iw*2:ih*2:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "copy", final_abs],
        capture_output=True, text=True, timeout=900, cwd=TEMP_DIR)
    if r3.returncode != 0:
        log(f"  [{clip_num}] FAIL 4K upscale")
        return False, time.time() - t0
    log(f"  [{clip_num}] Step 4 done: {time.time()-t1:.0f}s")

    # Step 5: Copy to clean
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = CLEAN_DIR / cinematic
    shutil.copy2(final_abs, clean_path)

    # Cleanup temp
    for tmp in [os.path.join(TEMP_DIR, trf), stab_tmp]:
        if os.path.exists(tmp):
            os.remove(tmp)

    elapsed = time.time() - t0
    fmb = os.path.getsize(final_abs) / (1024*1024)
    log(f"  [{clip_num}] COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f}m), {fmb:.1f} MB")
    return True, elapsed


try:
    REELS_DIR.mkdir(parents=True, exist_ok=True)

    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    total = len(clips)

    with open(RESULT, "w") as f:
        f.write(f"Batch pipeline: {GAME}\n")
        f.write(f"Total clips: {total}\n\n")

    t_batch = time.time()
    success = 0
    failed = 0
    total_time = 0

    for i, clip_path in enumerate(clips):
        clip_num = clip_path.name.split("__")[0]  # e.g. "001"
        stem = clip_path.stem
        final_check = REELS_DIR / f"{stem}__portrait__FINAL.mp4"
        if final_check.exists():
            log(f"=== Clip {i+1}/{total}: {clip_num} SKIPPED (already done) ===")
            success += 1
            continue
        log(f"=== Clip {i+1}/{total}: {clip_num} ({clip_path.name[:50]}...) ===")

        ok, elapsed = process_clip(clip_path, clip_num)
        total_time += elapsed
        if ok:
            success += 1
        else:
            failed += 1

        avg = total_time / (i + 1)
        remaining = avg * (total - i - 1)
        log(f"  Progress: {success} done, {failed} failed, ~{remaining/60:.0f}m remaining\n")

    batch_time = time.time() - t_batch
    log(f"\n{'='*60}")
    log(f"BATCH COMPLETE: {success}/{total} succeeded, {failed} failed")
    log(f"Total time: {batch_time:.0f}s ({batch_time/60:.1f}m)")

except Exception as e:
    with open(RESULT, "a") as f:
        f.write(f"\nEXCEPTION: {e}\n{traceback.format_exc()}")
