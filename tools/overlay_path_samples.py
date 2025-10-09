import argparse
import json
import os
from typing import List, Optional, Tuple

import cv2


def _coerce_pair(x: object, y: object) -> Tuple[Optional[float], Optional[float]]:
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None, None


def _pick_xy(record: dict) -> Tuple[Optional[float], Optional[float]]:
    key_pairs = (
        ("bx_stab", "by_stab"),
        ("bx_raw", "by_raw"),
        ("bx", "by"),
    )

    for key_x, key_y in key_pairs:
        if key_x in record and key_y in record:
            x, y = _coerce_pair(record.get(key_x), record.get(key_y))
            if x is not None and y is not None:
                return x, y

    ball_val = record.get("ball")
    if isinstance(ball_val, dict):
        for key_x, key_y in (*key_pairs, ("x", "y")):
            if key_x in ball_val and key_y in ball_val:
                x, y = _coerce_pair(ball_val.get(key_x), ball_val.get(key_y))
                if x is not None and y is not None:
                    return x, y
    elif isinstance(ball_val, (list, tuple)) and len(ball_val) >= 2:
        x, y = _coerce_pair(ball_val[0], ball_val[1])
        if x is not None and y is not None:
            return x, y

    # fall back to any two keys that end with x/y
    x_key = y_key = None
    for key in record:
        if key.lower().endswith("x") and x_key is None:
            x_key = key
        elif key.lower().endswith("y") and y_key is None:
            y_key = key
        if x_key and y_key:
            x, y = _coerce_pair(record.get(x_key), record.get(y_key))
            if x is not None and y is not None:
                return x, y

    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--outdir", default=r"out\diag_path")
    ap.add_argument("--every", type=int, default=60, help="save every N frames")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pts: List[Tuple[int, float, float]] = []
    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            x, y = _pick_xy(record)
            if x is None or y is None:
                continue
            try:
                frame_idx_val = record.get("f", len(pts))
                frame_idx = int(frame_idx_val)
            except (TypeError, ValueError):
                frame_idx = len(pts)
            pts.append((frame_idx, x, y))

    if not pts:
        raise SystemExit("No usable ball positions found in path file")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.input}")

    for frame_idx, bx, by in pts[:: args.every]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            continue
        center = (int(round(bx)), int(round(by)))
        cv2.circle(img, center, 10, (0, 0, 255), 3)
        cv2.putText(
            img,
            f"f={frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (50, 220, 50),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(os.path.join(args.outdir, f"{frame_idx:06d}.png"), img)

    print("Wrote:", args.outdir)


if __name__ == "__main__":
    main()
