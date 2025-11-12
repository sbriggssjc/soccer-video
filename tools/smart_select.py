"""CLI entry point for the Smart Soccer Highlight Selector pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Support invocation both as module and script.
    from .ffconcat import Clip, ensure_duration_cap, total_duration, write_ffconcat
    from .metrics import compute_metrics, write_reports
    from .selector import SelectorConfig, SmartSelector
    from .signal_features import (
        AudioFeatureConfig,
        MotionFeatureConfig,
        compute_audio_features,
        compute_motion_features,
        derive_in_play_mask,
    )
except ImportError:  # pragma: no cover - executed when run as a script.
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from tools.ffconcat import Clip, ensure_duration_cap, total_duration, write_ffconcat
    from tools.metrics import compute_metrics, write_reports
    from tools.selector import SelectorConfig, SmartSelector
    from tools.signal_features import (
        AudioFeatureConfig,
        MotionFeatureConfig,
        compute_audio_features,
        compute_motion_features,
        derive_in_play_mask,
    )


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )


def _read_spans_csv(path: Path) -> List[Tuple[float, float]]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    start_col = cols.get("start") or cols.get("t0") or cols.get("begin")
    end_col = cols.get("end") or cols.get("t1") or cols.get("stop")
    if not start_col or not end_col:
        raise ValueError(f"Could not find start/end columns in {path}")
    spans: List[Tuple[float, float]] = []
    for row in df.itertuples(index=False):
        start = float(getattr(row, start_col))
        end = float(getattr(row, end_col))
        if end > start:
            spans.append((start, end))
    return spans


def _read_events_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    t0 = cols.get("start") or cols.get("t0") or cols.get("begin")
    t1 = cols.get("end") or cols.get("t1") or cols.get("stop")
    label_col = cols.get("label") or cols.get("event") or cols.get("type")
    if t0 and "time" not in cols:
        df["time"] = (df[t0] + df[t1]) / 2.0 if t1 else df[t0]
    elif "time" not in df.columns and cols.get("t"):
        df["time"] = df[cols.get("t")]
    if label_col:
        df["label"] = df[label_col].astype(str).str.upper()
    if t0:
        df["t0"] = df[t0]
    if t1:
        df["t1"] = df[t1]
    return df


def _read_goal_resets(path: Path) -> List[float]:
    df = pd.read_csv(path)
    values: List[float] = []
    for column in df.columns:
        try:
            numeric = pd.to_numeric(df[column], errors="coerce")
        except Exception:
            continue
        values.extend(float(v) for v in numeric.dropna())
        if values:
            break
    return sorted(values)


def _spans_from_events(events: Sequence[Dict[str, float]]) -> List[Tuple[float, float]]:
    spans = []
    for event in events:
        spans.append((float(event["start"]), float(event["end"])))
    return spans


def _clips_from_spans(spans: Sequence[Dict[str, float]]) -> List[Clip]:
    return [Clip(float(row["start"]), float(row["end"]), row.get("label", ""), float(row.get("score", 0.0))) for row in spans]


def _write_ffconcat_sets(video_path: Path, spans_df: pd.DataFrame, reels_dir: Path, target_cap: float) -> Dict[str, float]:
    reels_dir.mkdir(parents=True, exist_ok=True)
    records = spans_df.to_dict("records")
    clips = _clips_from_spans(records)
    write_ffconcat(clips, video_path, reels_dir / "all.ffconcat")

    goal_clips = [clip for clip in clips if clip.label == "GOAL"]
    if goal_clips:
        write_ffconcat(goal_clips, video_path, reels_dir / "goals.ffconcat")
    else:
        (reels_dir / "goals.ffconcat").write_text("ffconcat version 1.0\n", encoding="utf-8")

    shot_labels = {"SHOT", "WOODWORK", "SAVE"}
    shot_clips = [clip for clip in clips if clip.label in shot_labels]
    if shot_clips:
        write_ffconcat(shot_clips, video_path, reels_dir / "shots.ffconcat")
    else:
        (reels_dir / "shots.ffconcat").write_text("ffconcat version 1.0\n", encoding="utf-8")

    capped = ensure_duration_cap(clips, target_cap)
    if capped:
        write_ffconcat(capped, video_path, reels_dir / "all_capped.ffconcat")

    return {
        "total_all": total_duration(clips),
        "total_goals": total_duration(goal_clips),
        "total_shots": total_duration(shot_clips),
        "total_capped": total_duration(capped),
    }


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log_smart_select.txt"
    _configure_logging(log_path)
    logger = logging.getLogger("smart_select")

    video_path = Path(args.video)
    logger.info("Video: %s", video_path)

    audio_df = compute_audio_features(video_path, AudioFeatureConfig())
    logger.info("Audio frames: %d", len(audio_df))
    motion_df = compute_motion_features(video_path, MotionFeatureConfig())
    logger.info("Motion samples: %d", len(motion_df))

    in_play_spans: List[Tuple[float, float]]
    if args.in_play:
        in_play_spans = _read_spans_csv(Path(args.in_play))
        logger.info("Loaded in-play mask from %s (%d spans)", args.in_play, len(in_play_spans))
    else:
        inferred = derive_in_play_mask(audio_df, motion_df)
        in_play_spans = _spans_from_events(inferred.to_dict("records"))
        logger.info("Derived in-play mask (%d spans)", len(in_play_spans))

    prior_frames = []
    goal_resets: List[float] = []
    forced_goals_df = pd.DataFrame()
    if args.use_priors:
        for path in args.use_priors:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning("Prior path missing: %s", path)
                continue
            if "reset" in path_obj.stem:
                goal_resets.extend(_read_goal_resets(path_obj))
                logger.info("Loaded %d goal resets from %s", len(goal_resets), path)
            elif "forced" in path_obj.stem:
                forced_goals_df = _read_events_csv(path_obj)
                logger.info("Loaded forced goal marks: %d", len(forced_goals_df))
            else:
                prior_frames.append(_read_events_csv(path_obj))
        if prior_frames:
            prior_events = pd.concat(prior_frames, ignore_index=True, sort=False)
        else:
            prior_events = pd.DataFrame()
    else:
        prior_events = pd.DataFrame()

    goal_resets = sorted(set(goal_resets))
    logger.info("Goal resets considered: %d", len(goal_resets))

    config = SelectorConfig(min_clip=args.min_clip, max_clip=args.max_clip)
    selector = SmartSelector(
        audio_df,
        motion_df,
        in_play_spans,
        config,
        prior_events=prior_events,
        goal_resets=goal_resets,
        forced_goal_marks=forced_goals_df,
    )
    events_df, spans, ratios = selector.run()
    combined_threshold = (
        selector.last_combined_threshold if selector.last_combined_threshold is not None else float("nan")
    )
    logger.info("Selector thresholds: combined>=%.3f audio_z>=%.2f motion_z>=%.2f", combined_threshold, config.audio_peak_threshold, config.motion_peak_threshold)
    logger.info(
        "Window ranges goal=%s/%s shot=%s/%s save=%s/%s buildup=%s/%s",
        config.goal_pre_range,
        config.goal_post_range,
        config.shot_pre_range,
        config.shot_post_range,
        config.save_pre_range,
        config.save_post_range,
        config.buildup_pre_range,
        config.buildup_post_range,
    )
    events_path = out_dir / "events_final.csv"
    events_df.to_csv(events_path, index=False)
    logger.info("Exported %d events to %s", len(events_df), events_path)

    reels_dir = out_dir / "reels"
    totals = _write_ffconcat_sets(video_path, events_df, reels_dir, args.target_cap_sec)
    logger.info("ffconcat totals: %s", totals)

    metrics = compute_metrics(events_df, goal_resets, ratios)
    write_reports(metrics, out_dir)
    logger.info("Metrics: %s", metrics)

    summary = {
        "audio_frames": len(audio_df),
        "motion_samples": len(motion_df),
        "combined_threshold": selector.last_combined_threshold,
        "merge_gap": config.merge_gap,
        "goal_resets": goal_resets,
        "totals": totals,
        "metrics": metrics.__dict__,
    }
    (out_dir / "log_smart_select_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Self tests


def _build_synthetic_features(duration: float = 120.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(12345)
    times = np.linspace(0, duration, int(duration * 4))
    audio = pd.DataFrame({
        "time": times,
        "rms": 0.1 + 0.02 * rng.standard_normal(len(times)),
        "rms_smooth": 0.1 + 0.02 * rng.standard_normal(len(times)),
        "centroid": 0.2 + 0.03 * rng.standard_normal(len(times)),
        "centroid_smooth": 0.2 + 0.03 * rng.standard_normal(len(times)),
        "zcr": np.zeros_like(times),
        "crowd_score": 0.1 * np.ones_like(times),
        "whistle_score": np.zeros_like(times),
    })
    motion = pd.DataFrame({
        "time": times,
        "motion": 0.05 + 0.02 * rng.standard_normal(len(times)),
        "motion_smooth": 0.05 + 0.02 * rng.standard_normal(len(times)),
        "pan_score": np.zeros_like(times),
    })
    return audio, motion


def _self_test_windows() -> None:
    audio, motion = _build_synthetic_features()
    audio.loc[(audio["time"] > 40) & (audio["time"] < 41), "rms_smooth"] += 4.0
    motion.loc[(motion["time"] > 40) & (motion["time"] < 41), "motion_smooth"] += 3.0
    selector = SmartSelector(audio, motion, [(0.0, 120.0)], SelectorConfig(), goal_resets=[70.0])
    events_df, spans, ratios = selector.run()
    assert not events_df.empty, "Expected at least one event"
    goal = events_df.iloc[0]
    assert goal["end"] <= 70.0 and goal["start"] <= 45.0, "Goal window should end before reset"


def _self_test_goal_coverage() -> None:
    audio, motion = _build_synthetic_features()
    audio.loc[(audio["time"] > 20) & (audio["time"] < 21), "rms_smooth"] += 3.5
    selector = SmartSelector(audio, motion, [(0.0, 120.0)], SelectorConfig(), goal_resets=[55.0])
    events_df, _, _ = selector.run()
    assert (events_df["label"] == "GOAL").any(), "Goal coverage expected"


def _self_test_in_play_clamp() -> None:
    audio, motion = _build_synthetic_features()
    audio.loc[(audio["time"] > 60) & (audio["time"] < 61), "rms_smooth"] += 4.0
    in_play = [(0.0, 59.0), (62.0, 120.0)]
    selector = SmartSelector(audio, motion, in_play, SelectorConfig(), goal_resets=[90.0])
    events_df, spans, _ = selector.run()
    assert not events_df.empty
    for span in spans:
        if span.label != "GOAL":
            assert span.start >= 62.0 or span.end <= 59.0, "Non-goal span must be clamped to in-play"


def run_self_tests() -> None:
    _self_test_windows()
    _self_test_goal_coverage()
    _self_test_in_play_clamp()
    print("Self-tests passed")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Soccer Highlight Selector")
    parser.add_argument("--video", type=Path, help="Path to stabilized match video")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--use-priors", nargs="*", default=[], help="CSV priors to use when scoring")
    parser.add_argument("--in-play", type=Path, help="Optional in-play CSV")
    parser.add_argument("--target-cap-sec", type=float, default=360.0)
    parser.add_argument("--min-clip", type=float, default=3.0)
    parser.add_argument("--max-clip", type=float, default=16.0)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_tests()
        return
    if not args.video:
        parser.error("--video is required unless --self-test is used")
    run_pipeline(args)


if __name__ == "__main__":
    main()


