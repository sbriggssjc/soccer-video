"""Helpers for Tulsa Soccer Club video branding."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def _ffprobe_dimensions(path: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(proc.stdout)
    stream = data.get("streams", [{}])[0]
    return int(stream.get("width", 0)), int(stream.get("height", 0))


def _bool_to_switch(name: str, value: bool) -> List[str]:
    if value:
        return [f"-{name}"]
    return [f"-{name}:$false"]


def brand_video(**kwargs: Dict[str, object]) -> Tuple[Path, List[str]]:
    """Brand a video using the PowerShell entry point.

    Parameters mirror ``tsc_brand.ps1``. The function returns the output path and
    the command list that was executed for logging.
    """

    if "In" not in kwargs or "Out" not in kwargs:
        raise ValueError("'In' and 'Out' parameters are required")

    src = Path(str(kwargs["In"]))
    dst = Path(str(kwargs["Out"]))

    if not src.exists():
        raise FileNotFoundError(f"Missing input video: {src}")

    if shutil.which("pwsh") is None:
        raise RuntimeError("pwsh is required to run the branding script")

    width, height = _ffprobe_dimensions(src)
    aspect = kwargs.get("Aspect") or ("9x16" if height > width else "16x9")

    script = Path(__file__).with_name("tsc_brand.ps1")
    cmd: List[str] = ["pwsh", "-File", str(script)]

    def add_arg(name: str, value: object) -> None:
        cmd.extend([f"-{name}", str(value)])

    add_arg("In", src)
    add_arg("Out", dst)
    add_arg("Aspect", aspect)

    optional_strs = ["Title", "Subtitle", "FontFile"]
    for key in optional_strs:
        if key in kwargs and kwargs[key] is not None:
            add_arg(key, kwargs[key])

    if "BitrateMbps" in kwargs and kwargs["BitrateMbps"] is not None:
        add_arg("BitrateMbps", kwargs["BitrateMbps"])

    switches = {
        "Watermark": kwargs.get("Watermark", True),
        "LowerThird": kwargs.get("LowerThird", False),
        "EndCard": kwargs.get("EndCard", False),
        "ShowGuides": kwargs.get("ShowGuides", False),
    }
    for name, enabled in switches.items():
        cmd.extend(_bool_to_switch(name, bool(enabled)))

    subprocess.run(cmd, check=True)
    return dst, cmd


__all__ = ["brand_video"]
