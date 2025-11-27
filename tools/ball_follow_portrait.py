import cv2
import json
import argparse
import os


def load_telemetry(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No telemetry rows found in {path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="Path to source atomic clip (landscape 1920x1080)")
    ap.add_argument("--telemetry", required=True, help="Path to .ball.jsonl")
    ap.add_argument("--out", required=True, help="Output portrait mp4")
    ap.add_argument("--width", type=int, default=1080, help="Output width (portrait, default 1080)")
    ap.add_argument("--height", type=int, default=1920, help="Output height (portrait, default 1920)")
    ap.add_argument("--debug-overlay", action="store_true", help="Draw ball dot & crop box on output")
    args = ap.parse_args()

    clip_path = args.clip
    tele_path = args.telemetry
    out_path = args.out
    out_w = args.width
    out_h = args.height

    # --- Load telemetry ---
    rows = load_telemetry(tele_path)
    print(f"[BALL-FOLLOW] Loaded {len(rows)} telemetry rows")

    # --- Open video ---
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[BALL-FOLLOW] clip size: {src_w}x{src_h}, fps={fps:.3f}")

    if src_w <= 0 or src_h <= 0:
        raise SystemExit("Invalid source dimensions from video")

    # --- Compute scale to make height = out_h ---
    scale = out_h / src_h
    scaled_w = int(round(src_w * scale))
    print(f"[BALL-FOLLOW] scale={scale:.4f}, scaled size={scaled_w}x{out_h}")

    if scaled_w < out_w:
        raise SystemExit(
            f"Scaled width ({scaled_w}) < out_w ({out_w}); "
            "this script assumes landscape->portrait via scale-up then horizontal crop."
        )

    # --- Setup writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for output: {out_path}")

    half_w = out_w // 2
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= len(rows):
            break

        row = rows[frame_idx]

        # Ball in source coords
        ball_src = row.get("ball_src") or row.get("ball")
        if not ball_src:
            bx_src = src_w / 2.0
        else:
            bx_src, _ = ball_src

        # Scale ball x to the scaled frame
        bx_scaled = bx_src * scale

        # --- Resize frame to scaled size (scaled_w x out_h) ---
        frame_scaled = cv2.resize(frame, (scaled_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Clamp center_x so crop stays within [0, scaled_w]
        center_x = max(half_w, min(scaled_w - half_w, bx_scaled))

        # Convert to integer crop bounds with extra clamping
        x0 = int(round(center_x - half_w))
        # Make absolutely sure we don't go out of bounds due to rounding
        x0 = max(0, min(x0, scaled_w - out_w))
        x1 = x0 + out_w

        # --- Crop horizontal window following the ball ---
        crop = frame_scaled[:, x0:x1]

        # Safety: enforce exact size for the writer
        if crop.shape[1] != out_w or crop.shape[0] != out_h:
            print(f"[WARN] Bad crop shape {crop.shape}, fixing via resize")
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        if args.debug_overlay:
            # Draw ball dot in portrait space for debugging
            bx_portrait = bx_scaled - x0
            by_portrait = int(out_h * 0.55)  # roughly 55% down the frame
            cv2.circle(crop, (int(bx_portrait), by_portrait), 10, (0, 0, 255), -1)
            # vertical center line
            cv2.line(crop, (half_w, 0), (half_w, out_h), (0, 255, 0), 1)
            # frame index text
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

        writer.write(crop)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[BALL-FOLLOW] Wrote portrait follow clip to: {out_path}")


if __name__ == "__main__":
    main()
