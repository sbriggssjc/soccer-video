"""Console entry point for the soccer highlights suite."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from ._loguru import logger

from .clips import run_clips
from .config import AppConfig, load_config
from .detect import run_detect
from .rank import run_topk
from .reels import render_reel
from .shrink import run_shrink
from .utils import load_report, summary_stats


def _configure_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def _load_config(path: Path) -> AppConfig:
    if not path.exists():
        logger.info("Using default configuration; no %s found", path)
    return load_config(path)


def _resolve_video_path(config: AppConfig, override: str | None) -> Path:
    return Path(override) if override else Path(config.paths.video)


def cmd_detect(config: AppConfig, args: argparse.Namespace) -> None:
    if args.pre is not None:
        config.detect.pre = args.pre
    if args.post is not None:
        config.detect.post = args.post
    if args.max_count is not None:
        config.detect.max_count = args.max_count
    if args.min_gap is not None:
        config.detect.min_gap = args.min_gap
    if args.audio_weight is not None:
        config.detect.audio_weight = args.audio_weight
    video_path = _resolve_video_path(config, args.video)
    config.paths.video = video_path
    out_csv = Path(args.out or (config.output_dir / "highlights.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result = run_detect(config, video_path, out_csv)
    stats = summary_stats(result.windows)
    report = load_report(config.output_dir)
    report.update(
        "detect",
        {
            **stats,
            "adaptive_threshold": round(result.adaptive_threshold, 3),
            "low_threshold": round(result.low_threshold, 3),
            "mean_score": round(result.mean_score, 3),
            "std_score": round(result.std_score, 3),
            "output": out_csv.as_posix(),
        },
    )


def cmd_shrink(config: AppConfig, args: argparse.Namespace) -> None:
    if args.mode:
        config.shrink.mode = args.mode
    if args.pre is not None:
        config.shrink.pre = args.pre
    if args.post is not None:
        config.shrink.post = args.post
    if args.aspect:
        config.shrink.aspect = args.aspect
    if args.zoom is not None:
        config.shrink.zoom = args.zoom
    if args.bias_blue:
        config.shrink.bias_blue = True
    if args.write_clips:
        config.shrink.write_clips = Path(args.write_clips)
    video_path = _resolve_video_path(config, args.video)
    config.paths.video = video_path
    csv_in = Path(args.csv or (config.output_dir / "highlights.csv"))
    csv_out = Path(args.out or (config.output_dir / "highlights_tight.csv"))
    refined = run_shrink(config, video_path, csv_in, csv_out)
    stats = summary_stats(refined)
    report = load_report(config.output_dir)
    report.update(
        "shrink",
        {**stats, "mode": config.shrink.mode, "output": csv_out.as_posix()},
    )


def cmd_clips(config: AppConfig, args: argparse.Namespace) -> None:
    if args.min_dur is not None:
        config.clips.min_duration = args.min_dur
    if args.workers is not None:
        config.clips.workers = args.workers
    if args.overwrite:
        config.clips.overwrite = True
    video_path = _resolve_video_path(config, args.video)
    config.paths.video = video_path
    csv_path = Path(args.csv or (config.output_dir / "highlights_tight.csv"))
    out_dir = Path(args.outdir or (config.output_dir / "clips"))
    exported = run_clips(config, video_path, csv_path, out_dir)
    report = load_report(config.output_dir)
    report.update(
        "clips",
        {
            "count": len(exported),
            "output_dir": out_dir.as_posix(),
        },
    )


def cmd_topk(config: AppConfig, args: argparse.Namespace) -> None:
    csv_out = Path(args.csv or (config.output_dir / "smart_top10.csv"))
    concat_out = Path(args.list or (config.output_dir / "smart_top10_concat.txt"))
    if args.k is not None:
        config.rank.k = args.k
    if args.max_len is not None:
        config.rank.max_len = args.max_len
    dirs: List[Path] = []
    if args.candirs:
        dirs = [Path(p.strip()) for p in args.candirs.split(",") if p.strip()]
    else:
        dirs = [config.output_dir / "clips", config.output_dir / "clips_acc"]
    ranked = run_topk(config, dirs, csv_out, concat_out)
    report = load_report(config.output_dir)
    top_score = max((clip.score for clip in ranked), default=0.0)
    report.update(
        "topk",
        {
            "count": len(ranked),
            "top_score": round(top_score, 3),
            "csv": csv_out.as_posix(),
            "concat": concat_out.as_posix(),
        },
    )


def cmd_reel(config: AppConfig, args: argparse.Namespace) -> None:
    list_path = Path(args.list or (config.output_dir / "smart_top10_concat.txt"))
    out_path = Path(args.out or (config.output_dir / "reels" / "top10.mp4"))
    title = args.title or config.reels.topk_title
    profile = args.profile or config.reels.profile
    duration = render_reel(config, list_path, out_path, title, profile)
    report = load_report(config.output_dir)
    report.update(
        "reel",
        {
            "profile": profile,
            "output": out_path.as_posix(),
            "duration": round(duration, 2),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soccerhl", description="Soccer highlights end-to-end pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    detect_p = sub.add_parser("detect", help="Run motion/audio detection")
    detect_p.add_argument("--video")
    detect_p.add_argument("--out")
    detect_p.add_argument("--pre", type=float)
    detect_p.add_argument("--post", type=float)
    detect_p.add_argument("--min-gap", dest="min_gap", type=float)
    detect_p.add_argument("--max-count", type=int)
    detect_p.add_argument("--audio-weight", type=float)
    detect_p.set_defaults(func=cmd_detect)

    shrink_p = sub.add_parser("shrink", help="Refine highlight windows")
    shrink_p.add_argument("--video")
    shrink_p.add_argument("--csv")
    shrink_p.add_argument("--out")
    shrink_p.add_argument("--mode", choices=["simple", "smart"])
    shrink_p.add_argument("--pre", type=float)
    shrink_p.add_argument("--post", type=float)
    shrink_p.add_argument("--aspect", choices=["horizontal", "vertical"])
    shrink_p.add_argument("--zoom", type=float)
    shrink_p.add_argument("--bias-blue", action="store_true")
    shrink_p.add_argument("--write-clips")
    shrink_p.set_defaults(func=cmd_shrink)

    clips_p = sub.add_parser("clips", help="Export clips with ffmpeg")
    clips_p.add_argument("--video")
    clips_p.add_argument("--csv")
    clips_p.add_argument("--outdir")
    clips_p.add_argument("--min-dur", type=float)
    clips_p.add_argument("--workers", type=int)
    clips_p.add_argument("--overwrite", action="store_true")
    clips_p.set_defaults(func=cmd_clips)

    topk_p = sub.add_parser("topk", help="Score and pick top clips")
    topk_p.add_argument("--candirs", help="Comma separated directories of candidate clips")
    topk_p.add_argument("--csv")
    topk_p.add_argument("--list")
    topk_p.add_argument("--k", type=int)
    topk_p.add_argument("--max-len", type=float)
    topk_p.set_defaults(func=cmd_topk)

    reel_p = sub.add_parser("reel", help="Render final reel video")
    reel_p.add_argument("--list")
    reel_p.add_argument("--out")
    reel_p.add_argument("--title")
    reel_p.add_argument("--profile")
    reel_p.set_defaults(func=cmd_reel)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    config_path = Path(args.config)
    config = _load_config(config_path)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(config, args)


if __name__ == "__main__":
    main()
