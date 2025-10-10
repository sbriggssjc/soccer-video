import subprocess, shlex, tempfile
from pathlib import Path

REALESRGAN_EXE = r"C:\Users\scott\soccer-video\tools\realesrgan\realesrgan-ncnn-vulkan.exe"
UPSCALE_OUT_ROOT = Path(r"C:\Users\scott\soccer-video\out\upscaled")
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


def upscale_video(inp: str, scale: int = 2, model: str = "realesrgan-x4plus") -> str:
    src = Path(inp)
    out = _out_path(src, scale)
    if out.exists() and out.stat().st_mtime > src.stat().st_mtime:
        return str(out)

    exe = Path(REALESRGAN_EXE)
    if exe.exists():
        # Try direct video path
        cmd = f'"{exe}" -i "{src}" -o "{out}" -n {model} -s {scale}'
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
        if proc.returncode == 0 and out.exists():
            return str(out)

        # Frame path fallback
        fps = _probe_fps(src)
        with tempfile.TemporaryDirectory() as td:
            tdin  = Path(td) / "in";  tdin.mkdir()
            tdout = Path(td) / "out"; tdout.mkdir()

            subprocess.check_call(['ffmpeg','-hide_banner','-y','-i',str(src),
                                   '-map','0:v:0','-vsync','0', str(tdin / '%06d.png')])

            subprocess.check_call([str(exe), '-i', str(tdin), '-o', str(tdout),
                                   '-n', model, '-s', str(scale)])

            subprocess.check_call(['ffmpeg','-hide_banner','-y','-framerate',fps,
                                   '-i', str(tdout / '%06d.png'),
                                   '-i', str(src), '-map','0:v:0','-map','1:a?',
                                   '-c:v','libx264','-preset','slow','-crf','18',
                                   '-pix_fmt','yuv420p', str(out)])
        return str(out)

    # ffmpeg-only fallback
    subprocess.check_call(['ffmpeg','-hide_banner','-y','-i',str(src),
                           '-vf', f'scale=iw*{scale}:ih*{scale}:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0',
                           '-map','0:v:0','-map','0:a?',
                           '-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p',
                           '-c:a','aac','-b:a','160k',
                           str(out)])
    return str(out)
