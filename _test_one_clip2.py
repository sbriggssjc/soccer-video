"""Quick test: vidstab one short clip using cwd approach."""
import subprocess, os

TEMP = r"D:\Projects\soccer-video\_tmp"
os.makedirs(TEMP, exist_ok=True)
SRC = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\009__2026-02-23__TSC_vs_NEOFC__CORNER__t850.00-t856.00__portrait.mp4"
DST = os.path.join(TEMP, "test_stab.mp4")

print("Pass 1: detect...")
r = subprocess.run([
    'ffmpeg', '-y', '-i', SRC,
    '-vf', 'vidstabdetect=shakiness=5:accuracy=15:result=test.trf',
    '-f', 'null', '-'
], capture_output=True, text=True, cwd=TEMP)
print(f"  detect exit={r.returncode}")
print(f"  trf exists: {os.path.exists(os.path.join(TEMP, 'test.trf'))}")

print("Pass 2: transform...")
r2 = subprocess.run([
    'ffmpeg', '-y', '-i', SRC,
    '-vf', 'vidstabtransform=input=test.trf:smoothing=15:interpol=bicubic:crop=black:zoom=3',
    '-c:v', 'libx264', '-crf', '17', '-preset', 'medium',
    '-pix_fmt', 'yuv420p', DST
], capture_output=True, text=True, cwd=TEMP)
print(f"  transform exit={r2.returncode}")
if r2.returncode != 0:
    print(f"  STDERR (last 500): {r2.stderr[-500:]}")
else:
    sz = os.path.getsize(DST) / (1024*1024)
    print(f"  SUCCESS: {sz:.1f} MB")
