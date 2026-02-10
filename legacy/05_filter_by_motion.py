import argparse
import csv
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover - librosa optional
    librosa = None


def parse_triplet(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected comma separated triplet like 100,25,25")
    nums = []
    for p in parts:
        try:
            nums.append(int(float(p)))
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(f"invalid HSV component '{p}'") from exc
    return tuple(max(0, min(255, n)) for n in nums)  # type: ignore[return-value]


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def read_candidates(csv_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing highlights CSV: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        field_map = []
        if reader.fieldnames:
            field_map = [fn.strip().lower().lstrip("\ufeff") for fn in reader.fieldnames]
            reader.fieldnames = field_map
        idx = 0
        for row in reader:
            idx += 1
            lowered = {k.lower(): v for k, v in row.items()}
            start = lowered.get("start") or lowered.get("t0") or lowered.get("clip_start")
            end = lowered.get("end") or lowered.get("t1") or lowered.get("clip_end")
            center = lowered.get("center") or lowered.get("mid") or lowered.get("time")
            if start is None or end is None:
                if center is not None:
                    try:
                        mid = float(str(center).replace(",", "."))
                    except ValueError:
                        continue
                    start = str(mid - 2.0)
                    end = str(mid + 2.0)
                else:
                    continue
            try:
                s_val = float(str(start).replace(",", "."))
                e_val = float(str(end).replace(",", "."))
            except ValueError:
                continue
            if math.isfinite(s_val) and math.isfinite(e_val) and e_val > s_val:
                rows.append({
                    "start": s_val,
                    "end": e_val,
                    "kind": str(lowered.get("kind") or lowered.get("label") or lowered.get("event") or "action"),
                })
    return rows


def hsv_mask(frame_bgr: np.ndarray, low: Sequence[int], high: Sequence[int]) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    low_arr = np.array(low, dtype=np.uint8)
    high_arr = np.array(high, dtype=np.uint8)
    mask = cv2.inRange(hsv, low_arr, high_arr)
    return mask


def center_lane_mask(height: int, width: int, inner_ratio: float = 0.6) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x0 = int((1.0 - inner_ratio) * 0.5 * width)
    x1 = max(x0 + 1, int(width - x0))
    mask[:, x0:x1] = 1
    return mask


def calc_action_metrics(
    cap: cv2.VideoCapture,
    start_s: float,
    end_s: float,
    team_low: Sequence[int],
    team_high: Sequence[int],
    presence_bias: float,
    step_frames: int = 3,
    downsample: int = 2,
) -> Dict[str, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    start_f = max(0, int(start_s * fps))
    end_f = max(start_f + 1, int(end_s * fps))
    total_frames = end_f - start_f
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    gray_frames: List[np.ndarray] = []
    color_frames: List[np.ndarray] = []
    for idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step_frames != 0:
            continue
        color_frames.append(frame.copy())
        if downsample > 1:
            frame = cv2.resize(frame, None, fx=1.0 / downsample, fy=1.0 / downsample, interpolation=cv2.INTER_AREA)
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if len(gray_frames) < 2:
        return {
            "flow_mean": 0.0,
            "flow_std": 0.0,
            "center_ratio": 0.0,
            "contig_frames": 0.0,
            "ball_speed": 0.0,
            "ball_pitch_ratio": 0.0,
            "navy_presence": 0.0,
            "motion_ratio": 0.0,
            "samples": float(len(gray_frames)),
        }

    height, width = gray_frames[0].shape[:2]
    lane_mask = center_lane_mask(height, width, inner_ratio=0.62).astype(bool)

    flow_means: List[float] = []
    center_samples: List[float] = []
    motion_area_samples: List[float] = []
    presence_samples: List[float] = []
    ball_points: List[Tuple[float, float]] = []

    prev = gray_frames[0]
    for idx, frame in enumerate(gray_frames[1:], start=1):
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            frame,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        mag = cv2.magnitude(flow[..., 0], flow[..., 1])
        flow_mean = float(np.mean(mag))
        flow_means.append(flow_mean)

        motion_threshold = float(np.percentile(mag, 85))
        motion_mask = mag > motion_threshold
        motion_ratio = float(np.mean(motion_mask))
        motion_area_samples.append(motion_ratio)

        center_ratio = float((motion_mask & lane_mask).sum() / (motion_mask.sum() + 1e-6))
        center_samples.append(center_ratio)

        color_idx = min(idx, len(color_frames) - 1)
        team_mask = hsv_mask(color_frames[color_idx], team_low, team_high)
        if downsample > 1:
            team_mask = cv2.resize(team_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        team_overlap = float(((motion_mask.astype(np.uint8) > 0) & (team_mask > 0)).sum())
        motion_pixels = float(motion_mask.sum() + 1e-6)
        presence_samples.append(team_overlap / motion_pixels)

        bright_threshold = float(np.percentile(mag, 99.5))
        bright_mask = mag >= bright_threshold
        ys, xs = np.where(bright_mask)
        if len(xs) > 0:
            ball_points.append((float(xs.mean()), float(ys.mean())))
        prev = frame

    flow_mean = float(np.mean(flow_means)) if flow_means else 0.0
    flow_std = float(np.std(flow_means)) if flow_means else 0.0
    center_ratio = float(np.mean(center_samples)) if center_samples else 0.0
    navy_raw = float(np.mean(presence_samples)) if presence_samples else 0.0
    navy_presence = float(np.clip(navy_raw * presence_bias, 0.0, 1.0))
    motion_ratio = float(np.mean(motion_area_samples)) if motion_area_samples else 0.0

    if flow_means:
        flow_gate = max(0.6 * flow_mean, float(np.percentile(flow_means, 50)))
    else:
        flow_gate = 0.0
    center_gate = max(0.18, center_ratio * 0.6)
    contig = 0
    best = 0
    for f_val, c_val in zip(flow_means, center_samples):
        if f_val >= flow_gate and c_val >= center_gate:
            contig += step_frames
            best = max(best, contig)
        else:
            contig = 0
    contig_frames = float(best)

    ball_speed = 0.0
    if len(ball_points) >= 2:
        distances = [
            math.hypot(ball_points[i + 1][0] - ball_points[i][0], ball_points[i + 1][1] - ball_points[i][1])
            for i in range(len(ball_points) - 1)
        ]
        if distances:
            ball_speed = float(np.median(distances))

    if ball_points:
        margin_x = 0.08 * width
        margin_y = 0.06 * height
        inside = 0
        for x, y in ball_points:
            if margin_x <= x <= (width - margin_x) and margin_y <= y <= (height - margin_y):
                inside += 1
        ball_pitch_ratio = float(inside / len(ball_points))
    else:
        ball_pitch_ratio = 0.0

    return {
        "flow_mean": flow_mean,
        "flow_std": flow_std,
        "center_ratio": center_ratio,
        "contig_frames": contig_frames,
        "ball_speed": ball_speed,
        "ball_pitch_ratio": ball_pitch_ratio,
        "navy_presence": navy_presence,
        "motion_ratio": motion_ratio,
        "samples": float(len(gray_frames)),
    }


def maybe_audio_db(video_path: str, start_s: float, end_s: float) -> float:
    if librosa is None:
        return 0.0
    duration = max(0.05, end_s - start_s)
    try:
        y, sr = librosa.load(video_path, sr=None, mono=True, offset=max(0.0, start_s), duration=duration)
    except Exception:
        return 0.0
    if y.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(y))))
    if rms <= 1e-6:
        return -60.0
    return float(20.0 * math.log10(rms + 1e-9))


def time_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    if inter <= 0:
        return 0.0
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    if union <= 0:
        return 0.0
    return inter / union


def violates_min_sep(a: Dict[str, float], b: Dict[str, float], gap: float) -> bool:
    if gap <= 0:
        return False
    if a["start"] >= b["end"] + gap:
        return False
    if b["start"] >= a["end"] + gap:
        return False
    return True


def load_team_config(json_path: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if json_path and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank soccer actions by motion and Navy presence")
    parser.add_argument("--video", required=True, help="Input stabilized match video")
    parser.add_argument("--csv", "--highlights", dest="csv", required=True, help="Candidate highlight CSV")
    parser.add_argument("--out", default=os.path.join("out", "plays.csv"), help="Output ranked plays CSV")
    parser.add_argument("--goal-resets", dest="goal_resets", help="Legacy compat; ignored")
    parser.add_argument("--top-dir", dest="legacy_top_dir", help="Legacy compat directory (ignored)")
    parser.add_argument("--goals-dir", dest="legacy_goals_dir", help="Legacy compat directory (ignored)")
    parser.add_argument("--min-flow-mean", dest="min_flow_mean", type=float, default=1.6)
    parser.add_argument("--min-flow", dest="min_flow_mean", type=float, help="Alias for --min-flow-mean")
    parser.add_argument("--min-ball-speed", type=float, default=1.2)
    parser.add_argument("--min-center-ratio", type=float, default=0.1)
    parser.add_argument("--min-contig-frames", type=float, default=12)
    parser.add_argument("--min-duration", type=float, default=2.4)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--need-ball", type=int, default=0, help="Require ball on pitch (1=yes)")
    parser.add_argument("--ball-on-pitch-required", dest="need_ball", action="store_const", const=1)
    parser.add_argument("--min-team-pres", dest="min_team_pres", type=float)
    parser.add_argument("--min-navy-pres", dest="min_navy_pres", type=float)
    parser.add_argument("--team-hsv-low", type=parse_triplet, dest="team_hsv_low")
    parser.add_argument("--team-hsv-high", type=parse_triplet, dest="team_hsv_high")
    parser.add_argument("--navy-json", default=os.path.join("config", "team_navy.json"))
    parser.add_argument("--rank-top", type=int, default=20)
    parser.add_argument("--min-sep", type=float, default=4.0, help="Minimum separation between kept plays (seconds)")
    parser.add_argument("--audio-boost", type=float, default=1.0, help="Scaling applied to audio z-score")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--step-frames", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--min-moving-players", dest="legacy_min_players", type=float, help="Legacy compat (ignored)")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    team_cfg = load_team_config(args.navy_json)
    default_low = tuple(team_cfg.get("hsv_low", (100, 25, 25)))  # type: ignore[arg-type]
    default_high = tuple(team_cfg.get("hsv_high", (140, 255, 255)))  # type: ignore[arg-type]
    team_low = args.team_hsv_low or default_low
    team_high = args.team_hsv_high or default_high
    presence_bias = float(team_cfg.get("presence_bias", 1.0))

    min_navy = args.min_navy_pres
    if min_navy is None:
        min_navy = team_cfg.get("presence_min")
    if min_navy is None:
        min_navy = args.min_team_pres if args.min_team_pres is not None else 0.35
    min_navy = float(min_navy)

    candidates = read_candidates(args.csv)
    if not candidates:
        raise RuntimeError("No candidate highlights found in CSV")

    ensure_dir(args.out)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    metrics_list: List[Dict[str, float]] = []
    audio_values: List[float] = []
    enriched_candidates: List[Dict[str, float]] = []

    for cand in candidates:
        start = float(cand["start"])
        end = float(cand["end"])
        if end - start < args.min_duration or end - start > args.max_duration:
            continue
        metrics = calc_action_metrics(
            cap,
            start,
            end,
            team_low,
            team_high,
            float(presence_bias),
            step_frames=max(1, int(args.step_frames)),
            downsample=max(1, int(args.downsample)),
        )
        navy_presence = metrics.get("navy_presence", 0.0)
        contig_frames = metrics.get("contig_frames", 0.0)
        ball_pitch_ratio = metrics.get("ball_pitch_ratio", 0.0)
        ball_speed = metrics.get("ball_speed", 0.0)
        if contig_frames < args.min_contig_frames:
            continue
        if navy_presence < min_navy:
            continue
        if args.need_ball and ball_pitch_ratio < 0.5:
            continue
        if metrics.get("flow_mean", 0.0) < args.min_flow_mean:
            continue
        if metrics.get("ball_speed", 0.0) < args.min_ball_speed:
            continue
        if metrics.get("center_ratio", 0.0) < args.min_center_ratio:
            continue

        audio_db = maybe_audio_db(args.video, start, end)
        audio_values.append(audio_db)

        enriched = {
            "start": start,
            "end": end,
            "kind": cand.get("kind", "action"),
            "flow_mean": metrics.get("flow_mean", 0.0),
            "flow_std": metrics.get("flow_std", 0.0),
            "center_ratio": metrics.get("center_ratio", 0.0),
            "contig_frames": contig_frames,
            "ball_speed": ball_speed,
            "ball_pitch_ratio": ball_pitch_ratio,
            "navy_presence": navy_presence,
            "motion_ratio": metrics.get("motion_ratio", 0.0),
            "audio_db": audio_db,
        }
        metrics_list.append(metrics)
        enriched_candidates.append(enriched)

    cap.release()

    if not enriched_candidates:
        raise RuntimeError("No plays passed the motion filters")

    audio_mean = float(np.mean(audio_values)) if audio_values else 0.0
    audio_std = float(np.std(audio_values)) if audio_values else 0.0

    scored: List[Dict[str, float]] = []
    for item in enriched_candidates:
        audio_db = item["audio_db"]
        audio_z = 0.0
        if audio_std > 1e-5:
            audio_z = (audio_db - audio_mean) / audio_std
        flow_mean = item["flow_mean"]
        ball_speed = item["ball_speed"]
        center_ratio = item["center_ratio"]
        navy_presence = item["navy_presence"]

        penalty_low_motion = max(0.0, args.min_flow_mean - flow_mean) * 1.5
        penalty_low_motion += max(0.0, args.min_ball_speed - ball_speed) * 1.2
        penalty_static = 0.0
        if item["flow_std"] < 0.12:
            penalty_static += 0.8
        if item["motion_ratio"] < 0.015:
            penalty_static += 0.7
        penalty_pitch = 0.0
        if center_ratio < args.min_center_ratio:
            penalty_pitch += (args.min_center_ratio - center_ratio) * 2.0
        if navy_presence < min_navy:
            penalty_pitch += (min_navy - navy_presence) * 3.0
        if item["ball_pitch_ratio"] < 0.4:
            penalty_pitch += 0.6

        score = (
            1.5 * flow_mean
            + 2.0 * ball_speed
            + 1.5 * center_ratio
            + 2.0 * navy_presence
            + 0.5 * audio_z * args.audio_boost
            - penalty_low_motion
            - penalty_static
            - penalty_pitch
        )

        flow_norm = flow_mean / max(args.min_flow_mean, 1e-3)
        ball_norm = ball_speed / max(args.min_ball_speed, 1e-3)
        center_norm = center_ratio / max(args.min_center_ratio, 1e-3)
        navy_norm = navy_presence / max(min_navy, 1e-3)
        confidence = max(0.0, min(1.0, (flow_norm + ball_norm + center_norm + navy_norm) / 4.0))

        scored.append({
            **item,
            "score": score,
            "audio_z": audio_z,
            "confidence": confidence,
        })

    scored.sort(key=lambda r: r["score"], reverse=True)

    selected: List[Dict[str, float]] = []
    for cand in scored:
        if len(selected) >= args.rank_top:
            break
        reject = False
        for prev in selected:
            if time_iou(cand, prev) > 0.5:
                reject = True
                break
            if violates_min_sep(cand, prev, args.min_sep):
                reject = True
                break
        if reject:
            continue
        selected.append(cand)

    selected.sort(key=lambda r: r["start"])

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "start",
            "end",
            "score",
            "kind",
            "confidence",
            "ball_speed",
            "flow_mean",
            "center_ratio",
            "navy_presence",
            "audio_db",
        ])
        for row in selected:
            writer.writerow([
                f"{row['start']:.3f}",
                f"{row['end']:.3f}",
                f"{row['score']:.4f}",
                row.get("kind", "action"),
                f"{row['confidence']:.3f}",
                f"{row['ball_speed']:.3f}",
                f"{row['flow_mean']:.3f}",
                f"{row['center_ratio']:.3f}",
                f"{row['navy_presence']:.3f}",
                f"{row['audio_db']:.3f}",
            ])

    if args.verbose:
        print(f"[filter] kept {len(selected)} plays from {len(scored)} scored candidates")
        print(f"[filter] output -> {args.out}")


if __name__ == "__main__":
    main()
