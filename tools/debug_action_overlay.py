import argparse
import json
import os
import cv2


def load_action_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Overlay action_x/action_y on source clip")
    ap.add_argument("--clip", required=True, help="Input video (source 16:9 clip)")
    ap.add_argument("--telemetry", required=True, help="action.jsonl from build_action_telemetry.py")
    ap.add_argument("--out", required=True, help="Output debug mp4")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--min-conf", type=float, default=0.30, help="min overall confidence to draw dot")
    args = ap.parse_args()

    rows = load_action_rows(args.telemetry)
    if not rows:
        raise SystemExit(f"No telemetry rows in {args.telemetry}")

    # Index by frame number
    by_frame = {}
    for r in rows:
        f_idx = int(r.get("f", 0))
        by_frame[f_idx] = r

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"Could not open clip: {args.clip}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter for {args.out}")

    print(f"[ACTION-DEBUG] clip={args.clip}, size={width}x{height}, fps={fps:.3f}")
    print(f"[ACTION-DEBUG] telemetry rows={len(rows)}")

    frame_idx = 0
    last_valid = None  # (x, y)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        row = by_frame.get(frame_idx)
        if row is not None:
            is_valid = bool(row.get("is_valid", False))
            conf = float(row.get("confidence", 0.0))
            if is_valid and conf >= args.min_conf:
                ax = float(row["action_x"])
                ay = float(row["action_y"])
                last_valid = (ax, ay)

        if last_valid is not None:
            ax, ay = last_valid
            cx = int(round(ax))
            cy = int(round(ay))
            # Clamp just in case
            cx = max(0, min(width - 1, cx))
            cy = max(0, min(height - 1, cy))

            # Red circle + small crosshair
            cv2.circle(frame, (cx, cy), 12, (0, 0, 255), 2)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 1)

        if frame_idx % 50 == 0:
            print(f"[ACTION-DEBUG] frame={frame_idx}, last_valid={last_valid}")

        writer.write(frame)
        frame_idx += 1

        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"[ACTION-DEBUG] wrote debug overlay to: {args.out}")


if __name__ == "__main__":
    main()
