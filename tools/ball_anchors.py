import argparse
import json
import os

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--frames",
        help="Comma-separated second marks like 0.8,5.2,9.7,13.0",
        required=True,
    )
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise SystemExit("Cannot open input")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    secs = [float(s.strip()) for s in args.frames.split(",") if s.strip()]
    points = []
    for s in secs:
        idx = max(0, int(round(s * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.namedWindow("Click ball (press ENTER)", cv2.WINDOW_NORMAL)
        cv2.imshow("Click ball (press ENTER)", frame)
        clicked = []

        def on_mouse(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked[:] = [(x, y)]
                img = frame.copy()
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                cv2.imshow("Click ball (press ENTER)", img)

        cv2.setMouseCallback("Click ball (press ENTER)", on_mouse)
        while True:
            key = cv2.waitKey(20)
            if key in (13, 10, 32):  # Enter or Space
                break
            if key == 27:  # Esc to skip
                clicked[:] = []
                break
        cv2.destroyWindow("Click ball (press ENTER)")
        if clicked:
            points.append(
                {
                    "t": float(idx) / float(fps),
                    "frame": idx,
                    "bx": float(clicked[0][0]),
                    "by": float(clicked[0][1]),
                }
            )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for point in points:
            f.write(json.dumps(point) + "\n")
    print(f"Wrote {len(points)} anchors -> {args.out}")


if __name__ == "__main__":
    main()
