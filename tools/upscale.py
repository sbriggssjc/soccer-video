import subprocess, tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

REALESRGAN_EXE = _SCRIPT_DIR / "realesrgan" / "realesrgan-ncnn-vulkan.exe"
UPSCALE_OUT_ROOT = _REPO_ROOT / "out" / "upscaled"
UPSCALE_OUT_ROOT.mkdir(parents=True, exist_ok=True)

def _out_path(src: Path, scale: int) -> Path:
    return UPSCALE_OUT_ROOT / f"{src.stem}__x{scale}.mp4"


def _probe_fps(src: Path) -> str:
    p = subprocess.run(
        ['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate',
         '-of','default=noprint_wrappers=1:nokey=1', str(src)],
        capture_output=True, text=True
    )
    return (p.stdout or "").strip() or "30/1"


def _probe_resolution(path: Path) -> tuple[int, int]:
    """Return (width, height) of a video file, or (0, 0) on failure."""
    p = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height',
         '-of', 'csv=p=0:s=x', str(path)],
        capture_output=True, text=True,
    )
    parts = (p.stdout or "").strip().split("x")
    if len(parts) >= 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return 0, 0


def upscale_video(inp: str, scale: int = 2, model: str = "realesrgan-x4plus",
                  *, force: bool = False) -> str:
    src = Path(inp)
    out = _out_path(src, scale)
    if not force and out.exists() and out.stat().st_mtime > src.stat().st_mtime:
        # Verify cached output has correct dimensions
        src_w, src_h = _probe_resolution(src)
        out_w, out_h = _probe_resolution(out)
        if src_w > 0 and out_w > 0:
            actual_ratio = out_w / src_w
            if abs(actual_ratio - scale) < 0.5:
                print(f"[UPSCALE] Using cached {out_w}x{out_h} ({actual_ratio:.1f}x): {out}")
                return str(out)
            print(f"[UPSCALE] Cached file has wrong ratio ({actual_ratio:.1f}x vs {scale}x), regenerating")
        else:
            return str(out)

    exe = Path(REALESRGAN_EXE)
    if exe.exists():
        # Use frame-by-frame mode (more reliable than direct video mode).
        # Use --tile 0 to avoid tile-boundary kaleidoscope artifacts.
        fps = _probe_fps(src)
        print(f"[UPSCALE] Real-ESRGAN frame-by-frame ({model}, {scale}x, tile=0)...")
        with tempfile.TemporaryDirectory() as td:
            tdin  = Path(td) / "in";  tdin.mkdir()
            tdout = Path(td) / "out"; tdout.mkdir()

            subprocess.check_call(['ffmpeg','-hide_banner','-y','-i',str(src),
                                   '-map','0:v:0','-vsync','0', str(tdin / '%06d.png')])

            subprocess.check_call([str(exe), '-i', str(tdin), '-o', str(tdout),
                                   '-n', model, '-s', str(scale), '-t', '0'])

            subprocess.check_call(['ffmpeg','-hide_banner','-y','-framerate',fps,
                                   '-i', str(tdout / '%06d.png'),
                                   '-i', str(src), '-map','0:v:0','-map','1:a?',
                                   '-c:v','libx264','-preset','slow','-crf','18',
                                   '-pix_fmt','yuv420p', str(out)])

        if out.exists():
            out_w, out_h = _probe_resolution(out)
            print(f"[UPSCALE] Real-ESRGAN done: {out_w}x{out_h}")
            return str(out)

        print("[UPSCALE] Real-ESRGAN failed, falling back to FFmpeg lanczos")

    # ffmpeg-only fallback (lanczos upscale + light sharpening)
    print(f"[UPSCALE] FFmpeg lanczos {scale}x upscale...")
    subprocess.check_call(['ffmpeg','-hide_banner','-y','-i',str(src),
                           '-vf', f'scale=iw*{scale}:ih*{scale}:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0',
                           '-map','0:v:0','-map','0:a?',
                           '-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p',
                           '-c:a','aac','-b:a','160k',
                           str(out)])
    return str(out)
