"""Helpers for Real-ESRGAN-driven upscaling with ffmpeg fallback."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

REALESRGAN_EXE = Path(r"C:\\Users\\scott\\soccer-video\\tools\\realesrgan\\realesrgan-ncnn-vulkan.exe")
UPSCALE_OUT_ROOT = Path(r"C:\\Users\\scott\\soccer-video\\out\\upscaled")
UPSCALE_OUT_ROOT.mkdir(parents=True, exist_ok=True)


def upscaled_path(inp: Path, scale: int) -> Path:
    return UPSCALE_OUT_ROOT / f"{inp.stem}__x{scale}.mp4"


def upscale_video(inp: str, scale: int = 2, model: str = "realesrgan-x4plus") -> str:
    src = Path(inp)
    out = upscaled_path(src, scale)
    if out.exists() and out.stat().st_mtime > src.stat().st_mtime:
        return str(out)

    if REALESRGAN_EXE.exists():
        # Try direct video IO first.
        cmd_direct = [
            str(REALESRGAN_EXE),
            "-i",
            str(src),
            "-o",
            str(out),
            "-n",
            model,
            "-s",
            str(scale),
        ]
        proc = subprocess.run(cmd_direct, capture_output=True, text=True)
        if proc.returncode == 0 and out.exists():
            return str(out)

        # Fall back to frame extraction pipeline.
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            tmp_in = tmp_root / "in"
            tmp_out = tmp_root / "out"
            tmp_in.mkdir(parents=True, exist_ok=True)
            tmp_out.mkdir(parents=True, exist_ok=True)

            fps_probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=r_frame_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(src),
                ],
                capture_output=True,
                text=True,
            )
            fps = fps_probe.stdout.strip() or "30/1"

            subprocess.check_call(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-y",
                    "-i",
                    str(src),
                    "-map",
                    "0:v:0",
                    "-vsync",
                    "0",
                    str(tmp_in / "%06d.png"),
                ]
            )

            subprocess.check_call(
                [
                    str(REALESRGAN_EXE),
                    "-i",
                    str(tmp_in),
                    "-o",
                    str(tmp_out),
                    "-n",
                    model,
                    "-s",
                    str(scale),
                ]
            )

            subprocess.check_call(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-y",
                    "-framerate",
                    fps,
                    "-i",
                    str(tmp_out / "%06d.png"),
                    "-i",
                    str(src),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a?",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(out),
                ]
            )
        return str(out)

    # ffmpeg-only fallback.
    subprocess.check_call(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"scale=iw*{scale}:ih*{scale}:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            str(out),
        ]
    )
    return str(out)
