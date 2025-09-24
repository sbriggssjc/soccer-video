#!/usr/bin/env python3
"""Generate a probe VF script and render a sanity clip.

This utility mirrors the PowerShell snippet from the Autoframe workflow by
writing a ``.vf`` filtergraph file and invoking ``ffmpeg`` with
``-filter_complex_script``.  The script also creates a tiny ``testsrc`` input so
that the command can be executed in isolation.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


FPS = 24
Z_EXPR = "min(max(1.8,1.1),3)"
CX_EXPR = f"(iw/2)+50*sin(t*{FPS}/24)"
CY_EXPR = f"(ih/2)+30*cos(t*{FPS}/24)"

W_EXPR = f"floor((((ih*9/16)/{Z_EXPR}))/2)*2"
H_EXPR = f"floor(((ih/{Z_EXPR}))/2)*2"
X_EXPR = f"({CX_EXPR})-({W_EXPR})/2"
Y_EXPR = f"({CY_EXPR})-({H_EXPR})/2"

VF_CONTENT = (
    "[0:v]" "crop=" f"{W_EXPR}:{H_EXPR}:{X_EXPR}:{Y_EXPR},"
    "scale=-2:1080:flags=lanczos,setsar=1,format=yuv420p"
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
WORK_DIR = OUT_DIR / "autoframe_work"
VF_PATH = WORK_DIR / "_probe.vf"
LOG_PATH = WORK_DIR / "_probe.log"
INPUT_PATH = OUT_DIR / "_sanity_in.mp4"
OUTPUT_PATH = OUT_DIR / "_probe.mp4"


def ensure_sanity_input() -> None:
    """Create a short ``testsrc`` clip for the probe run."""
    if INPUT_PATH.exists():
        return
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=1920x1080:rate={FPS}:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(INPUT_PATH),
        ]
    )


def write_vf_file() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    VF_PATH.write_text(VF_CONTENT + "\n", encoding="utf-8")
    LOG_PATH.write_text(
        "\n".join(
            [
                f"FPS={FPS}",
                f"W={W_EXPR}",
                f"H={H_EXPR}",
                f"X={X_EXPR}",
                f"Y={Y_EXPR}",
                "",
                VF_CONTENT,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_probe() -> None:
    run_ffmpeg(
        [
            "-hide_banner",
            "-y",
            "-nostdin",
            "-i",
            str(INPUT_PATH),
            "-filter_complex_script",
            str(VF_PATH),
            "-an",
            str(OUTPUT_PATH),
        ]
    )


def run_ffmpeg(args: list[str]) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg executable not found on PATH")

    subprocess.run([ffmpeg_path, *args], check=True)


if __name__ == "__main__":
    write_vf_file()
    try:
        ensure_sanity_input()
        run_probe()
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)
