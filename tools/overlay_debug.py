#!/usr/bin/env python
# tools/overlay_debug.py — self-contained overlay from JSONL telemetry
import argparse, json, os
from pathlib import Path
import cv2

def read_jsonl(p):
    recs = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input mp4 used for render")
    ap.add_argument("--telemetry", required=True, help="JSONL written by unified renderer")
    ap.add_argument("--out", default=None, help="output debug mp4")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--ball-radius", type=int, default=6)
    args = ap.parse_args()

    inp = os.path.abspath(args.inp)
    tel = os.path.abspath(args.telemetry)
    out = os.path.abspath(args.out) if args.out else os.path.splitext(inp)[0] + ".__DEBUG.mp4"

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    recs = read_jsonl(tel)
    if not recs:
        print("[ERROR] no telemetry records found"); return 2

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        print(f"[ERROR] cannot open input video: {inp}"); return 3

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out, fourcc, fps, (W, H), True)
    if not vw.isOpened():
        print(f"[ERROR] cannot open writer: {out}"); return 4

    idx = 0
    ok = True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded from input clip")
    return frames


def _sample_frame(frames: List[np.ndarray], fps_in: float, fps_out: float, index: int) -> np.ndarray:
    t = float(index) / float(fps_out)
    src_idx = int(round(t * fps_in))
    src_idx = max(0, min(src_idx, len(frames) - 1))
    return frames[src_idx]


def _draw(frame: np.ndarray, telemetry: dict, label_point: Optional[np.ndarray]) -> np.ndarray:
    output = frame.copy()

    crop = telemetry.get("crop")
    if crop and len(crop) >= 4:
        x0, y0, crop_w, crop_h = crop[:4]
    else:
        cx = telemetry.get("cx", 0.0)
        cy = telemetry.get("cy", 0.0)
        crop_w = telemetry.get("crop_w", 0.0)
        crop_h = telemetry.get("crop_h", 0.0)
        x0 = cx - crop_w / 2.0
        y0 = cy - crop_h / 2.0

    cv2.rectangle(
        output,
        (int(x0), int(y0)),
        (int(x0 + crop_w), int(y0 + crop_h)),
        (0, 255, 0),
        2,
    )

    ball = telemetry.get("ball")
    if ball and len(ball) >= 2:
        bx, by = ball[:2]
        cv2.circle(output, (int(bx), int(by)), 6, (0, 0, 255), -1)
    elif label_point is not None and not np.isnan(label_point).any():
        cv2.circle(output, (int(label_point[0]), int(label_point[1])), 8, (0, 0, 255), -1)

    used = telemetry.get("used_label", False)
    clamp = telemetry.get("clamp_flags", [])
    text = f"used_label={used} zoom={telemetry.get('zoom', 0.0):.2f}"
    if clamp:
        text += f" clamps={','.join(clamp)}"
    cv2.putText(output, text, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return output


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    telemetry_path = Path(args.telemetry).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")
    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {telemetry_path}")

    records = _read_telemetry(telemetry_path)
    fps_in = ffprobe_fps(input_path)
    if len(records) > 1:
        dt = np.median(np.diff([rec["t"] for rec in records]))
        fps_out = 1.0 / max(dt, 1e-6)
    else:
        fps_out = fps_in

    frames = _load_frames(input_path, args.flip180)

    label_points = None
    if args.labels_root:
        labels = load_labels(
            find_label_files(input_path.stem, args.labels_root),
            frames[0].shape[1],
            frames[0].shape[0],
            fps_in,
        )
        positions, _ = interp_labels_to_fps(labels, len(frames), fps_in, fps_out)
        label_points = positions

    height, width = frames[0].shape[:2]
    output_size = (width, height)
    out_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".__DEBUG.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer for {out_path}")

    for idx, telemetry in enumerate(records):
        frame = _sample_frame(frames, fps_in, fps_out, idx)
        point = None
        if label_points is not None and idx < len(label_points):
            point = label_points[idx]
        rendered = _draw(frame, telemetry, point)
        writer.write(rendered)

    writer.release()
    print(f"Wrote debug overlay to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay telemetry on a clip for QC")
    parser.add_argument("--in", dest="input", required=True, help="Source MP4")
    parser.add_argument("--telemetry", dest="telemetry", required=True, help="Telemetry JSONL path")
    parser.add_argument("--out", dest="output", help="Output MP4 path")
    parser.add_argument("--labels-root", dest="labels_root", help="Optional labels root for ball point rendering")
    parser.add_argument("--flip180", dest="flip180", action="store_true", help="Apply 180-degree rotation before drawing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    raise SystemExit(main())
