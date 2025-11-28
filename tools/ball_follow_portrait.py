#!/usr/bin/env python
import argparse
import json
import cv2
import os


def load_telemetry(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise RuntimeError(f"No telemetry rows found in {path}")
    print(f"[BALL-FOLLOW] Loaded {len(rows)} telemetry rows")
    return rows


def get_ball_xy(row):
    """
    Prefer ball_src (explicit source coords), fall back to ball, then ball_out.
    All are expected to be [x, y] in source space (1920x1080).
    """
    for key in ("ball_src", "ball", "ball_out"):
        v = row.get(key)
        if (
            isinstance(v, (list, tuple))
            and len(v) == 2
            and all(isinstance(c, (int, float)) for c in v)
        ):
            return float(v[0]), float(v[1])
    return None


def main():
    parser = argparse.ArgumentParser(description="Simple portrait follow using ball telemetry.")
    parser.add_argument("--clip", required=True, help="Source clip (landscape, e.g. 1920x1080)")
    parser.add_argument("--telemetry", required=True, help="Telemetry .jsonl with ball positions")
    parser.add_argument("--out", required=True, help="Output portrait mp4 path")
    parser.add_argument("--width", type=int, default=1080, help="Portrait width (e.g. 1080)")
    parser.add_argument("--height", type=int, default=1920, help="Portrait height (e.g. 1920)")
    parser.add_argument("--debug-overlay", action="store_true", help="Draw ball dot / center line")
    args = parser.parse_args()

    tele_rows = load_telemetry(args.telemetry)

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.clip}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[BALL-FOLLOW] clip size: {src_w}x{src_h}, fps={fps:.3f}")

    out_w = args.width
    out_h = args.height

    # Scale so that the source height becomes out_h (1920 for portrait)
    scale = out_h / float(src_h)
    scaled_w = int(round(src_w * scale))
    scaled_h = out_h
    print(f"[BALL-FOLLOW] scale={scale:.4f}, scaled size={scaled_w}x{scaled_h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {args.out}")
    print(f"[BALL-FOLLOW] VideoWriter opened with size=({out_w}x{out_h}), fps={fps:.3f}")

    half_w = out_w // 2
    max_x0 = max(0, scaled_w - out_w)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to scaled frame (landscape, same height as portrait)
        scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Get telemetry row (clamp at end if telemetry shorter)
        tel_idx = min(frame_idx, len(tele_rows) - 1)
        row = tele_rows[tel_idx]
        ball_xy = get_ball_xy(row)

        if ball_xy is not None:
            bx_src, by_src = ball_xy
            # Scale to match scaled frame
            bx_scaled = bx_src * scale
            by_scaled = by_src * scale

            # Clamp into scaled frame
            bx_scaled = max(0.0, min(float(scaled_w - 1), bx_scaled))
            by_scaled = max(0.0, min(float(scaled_h - 1), by_scaled))

            # Center crop on ball horizontally, but keep within bounds
            cx = bx_scaled
            cx = max(float(half_w), min(float(scaled_w - half_w), cx))
            x0 = int(round(cx - half_w))
        else:
            # No ball info: default to center crop of scaled frame
            bx_scaled = float(scaled_w) / 2.0
            by_scaled = float(scaled_h) / 2.0
            x0 = (scaled_w - out_w) // 2

        # Safety clamp for x0
        if x0 < 0:
            x0 = 0
        elif x0 > max_x0:
            x0 = max_x0

        x1 = x0 + out_w
        if x1 > scaled_w:
            x1 = scaled_w
            x0 = x1 - out_w

        # Extract crop from scaled frame
        crop = scaled[:, x0:x1, :]

        # Guard against empty / weird crops
        if crop is None or crop.size == 0:
            print(f"[ERROR] Empty crop at frame {frame_idx}, x0={x0}, x1={x1}, scaled_w={scaled_w}")
            break

        # Enforce exact size (out_h, out_w) before writing
        h, w = crop.shape[:2]
        if h != out_h or w != out_w:
            print(
                f"[WARN] Resizing crop at frame {frame_idx}: "
                f"got {w}x{h}, expected {out_w}x{out_h}"
            )
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            h, w = crop.shape[:2]

        if args.debug_overlay:
            # Ball position in portrait-crop coordinates
            bx_portrait = bx_scaled - float(x0)
            by_portrait = by_scaled

            bx_portrait = int(round(max(0.0, min(float(out_w - 1), bx_portrait))))
            by_portrait = int(round(max(0.0, min(float(out_h - 1), by_portrait))))

            # Red dot at ball
            cv2.circle(crop, (bx_portrait, by_portrait), 10, (0, 0, 255), -1)

            # Vertical center line (target follow center)
            cv2.line(crop, (half_w, 0), (half_w, out_h), (0, 255, 0), 1)

            # Frame index text
            cv2.putText(
                crop,
                f"f={frame_idx}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        if frame_idx in (0, 1, 2, 3, 4, 50, 100, 150, 200, 250, 300):
            print(
                f"[DEBUG] frame={frame_idx}, x0={x0}, x1={x1}, "
                f"crop_shape={crop.shape}, "
                f"bx_scaled={bx_scaled:.1f}, by_scaled={by_scaled:.1f}"
            )

        # Final safety: ensure contiguous uint8 for OpenCV
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGR)

        writer.write(crop)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[BALL-FOLLOW] Wrote portrait follow clip to: {args.out}")


if __name__ == "__main__":
    main()
