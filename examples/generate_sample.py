"""Generate a tiny synthetic soccer-like clip for testing using FFmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path


def main(output: Path = Path("sample_game.mp4"), duration: int = 12, fps: int = 24) -> None:
    size = "640x360"
    filters = ",".join([
        "drawbox=x=10:y=h/3:w=30:h=h/3:color=white@1:t=2",
        "drawbox=x=w-40:y=h/3:w=30:h=h/3:color=white@1:t=2",
        "drawbox=x=80+80*sin(PI*(t-3)):y=h/2+60*cos(PI*(t-3)):w=24:h=24:color=white@1:enable='between(t,3,5)'",
        "drawbox=x=w-(80+80*sin(PI*(t-8))):y=h/2+60*sin(1.5*PI*(t-8)):w=24:h=24:color=white@1:enable='between(t,8,10)'",
        "drawbox=x=w/2+120*sin(t*0.6):y=h/2+60*cos(t*0.6):w=30:h=50:color=0x3c3cee@1",
    ])
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x1e6e1e:s={size}:d={duration}:r={fps}",
        "-vf",
        filters,
        str(output),
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
