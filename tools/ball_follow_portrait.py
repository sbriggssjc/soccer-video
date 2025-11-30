#!/usr/bin/env python
import argparse
import json
import cv2
import os
import numpy as np


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
    Pick the best ball coordinate to use from the telemetry row.
    Prefer image/source coordinates if available.
    """
    # 1) ball_src: usually raw image coords
    if "ball_src" in row and row["ball_src"]:
        x, y = row["ball_src"][:2]
        return float(x), float(y)

    # 2) ball: sometimes same as ball_src, depending on pipeline
    if "ball" in row and row["ball"]:
        x, y = row["ball"][:2]
        return float(x), float(y)

    # 3) ball_out: if others missing; still image-like in your sample
    if "ball_out" in row and row["ball_out"]:
        x, y = row["ball_out"][:2]
        return float(x), float(y)

    return None, None


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
    width = out_w
    last_crop = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to scaled frame (landscape, same height as portrait)
        scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Get telemetry row (clamp at end if telemetry shorter)
        f = frame_idx
        row = tele_rows[min(f, len(tele_rows) - 1)]

        bx, by = get_ball_xy(row)
        if bx is None or by is None:
            # No ball; just use previous crop or skip overlay
            if last_crop is not None:
                crop = last_crop.copy()
                writer.write(crop)
            frame_idx += 1
            continue

        # Scale to the upscaled frame size
        bx_scaled = bx * scale
        by_scaled = by * scale

        # Center the portrait window on the ball horizontally
        cx = np.clip(bx_scaled, half_w, scaled_w - half_w)
        x0 = int(round(cx - half_w))
        x1 = x0 + width

        # Extract crop from scaled frame
        crop = scaled_frame[:, x0:x1]
        last_crop = crop

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

        f = frame_idx

        # --- DEBUG OVERLAY (ensure this replaces any previous circle/rect code) ---
        if args.debug_overlay:
            h, w = crop.shape[:2]

            # ball coords are in scaled-frame space; convert to crop-local space
            bx_local = int(round(bx_scaled - x0))
            by_local = int(round(by_scaled))

            # Draw the crop border so we can see the window
            cv2.rectangle(
                crop,
                (0, 0),
                (w - 1, h - 1),
                (0, 255, 0),
                2,
            )

            # Extra debug – log what we’re about to draw
            if f in (0, 50, 100, 150, 200, 250, 300):
                print(
                    f"[OVERLAY] frame={f}, bx_local={bx_local}, by_local={by_local}, "
                    f"w={w}, h={h}"
                )

            # Only draw if inside the visible crop
            if 0 <= bx_local < w and 0 <= by_local < h:
                cv2.circle(
                    crop,
                    (bx_local, by_local),
                    10,
                    (0, 0, 255),
                    2,
                )

        if frame_idx in (0, 1, 2, 3, 4, 50, 100, 150, 200, 250, 300):
            print(
                f"[DEBUG] frame={frame_idx}, x0={x0}, x1={x1}, "
                f"crop_shape={crop.shape}, "
                f"bx_scaled={bx_scaled:.1f}, by_scaled={by_scaled:.1f}"
            )

        # Final safety: ensure contiguous uint8 for OpenCV
        crop = np.ascontiguousarray(crop, dtype=np.uint8)

        writer.write(crop)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[BALL-FOLLOW] Wrote portrait follow clip to: {args.out}")


if __name__ == "__main__":
    main()
