import argparse
import json
import os
import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--outdir", default=r"out\diag_path")
    ap.add_argument("--every", type=int, default=60, help="save every N frames")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load path
    pts = []
    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pts.append((float(d.get("bx", 0)), float(d.get("by", 0))))

    cap = cv2.VideoCapture(args.input)
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    for f_idx in range(0, min(n, len(pts)), args.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, img = cap.read()
        if not ok:
            break
        bx, by = pts[f_idx]
        cv2.circle(img, (int(round(bx)), int(round(by))), 10, (0, 0, 255), 3)
        cv2.putText(
            img,
            f"f={f_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (50, 220, 50),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(os.path.join(args.outdir, f"{f_idx:06d}.png"), img)

    print("Wrote:", args.outdir)


if __name__ == "__main__":
    main()
