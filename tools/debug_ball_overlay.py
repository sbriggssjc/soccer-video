import cv2
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="path to source atomic clip")
    ap.add_argument("--telemetry", required=True, help="path to .ball.jsonl")
    ap.add_argument("--out", required=True, help="output debug mp4")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.clip}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DEBUG] clip size: {width}x{height}, fps={fps}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    # Load all telemetry rows
    rows = []
    with open(args.telemetry, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"[DEBUG] telemetry rows: {len(rows)}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(rows):
            break

        row = rows[frame_idx]

        # Ball position (prefer ball_src, fall back to ball)
        ball = row.get("ball_src") or row.get("ball")
        if ball is not None:
            bx, by = ball
            cv2.circle(frame, (int(bx), int(by)), 8, (0, 0, 255), -1)  # red dot

        # Crop rect: [x, y, w, h]
        crop = row.get("crop")
        if crop is not None and len(crop) == 4:
            x, y, w, h = crop
            x2 = x + w
            y2 = y + h
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)  # green box

        # Frame index text
        cv2.putText(
            frame,
            f"f={frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DEBUG] Wrote debug overlay to: {args.out}")


if __name__ == "__main__":
    main()
