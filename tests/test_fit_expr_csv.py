import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import fit_expr


def _write_track_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# fps=29.97\n")
        handle.write("# cli=autoframe.py --ball-detector=yolo\n")
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "cx",
                "cy",
                "z",
                "w",
                "h",
                "goal_x",
                "goal_y",
                "conf",
            ]
        )
        writer.writerow([0, 320.0, 180.0, 1.2, 640, 360, 300.0, 170.0, 0.8])
        writer.writerow([1, 324.0, 182.0, 1.18, 638, 358, 302.0, 171.0, 0.75])


def test_read_track_handles_extra_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "autoframe.csv"
    _write_track_csv(csv_path)

    frames, cx, cy, zoom, metadata, extras = fit_expr.read_track(csv_path)

    assert np.allclose(frames, np.array([0, 1], dtype=np.int64))
    assert np.allclose(cx, np.array([320.0, 324.0]))
    assert np.allclose(cy, np.array([180.0, 182.0]))
    assert np.allclose(zoom, np.array([1.2, 1.18]))
    assert metadata.get("cli") == "autoframe.py --ball-detector=yolo"
    assert set(extras.keys()) == {"w", "h", "goal_x", "goal_y", "conf"}
    assert np.allclose(extras["conf"], np.array([0.8, 0.75]))
