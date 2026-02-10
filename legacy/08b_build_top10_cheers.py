import argparse
from pathlib import Path

from soccer_highlights.top10_cheers import build_cheers_top10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Top-10 clips using cheers as optional anchors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default="out/full_game_stabilized.mp4", help="Input match video")
    parser.add_argument(
        "--filtered",
        default="out/highlights_filtered.csv",
        help="CSV of filtered highlight windows with action_score",
    )
    parser.add_argument("--cheers", help="CSV of cheer timestamps to force include", default=None)
    parser.add_argument("--clips-dir", default="out/clips_top10", help="Directory for exported clips")
    parser.add_argument("--concat", default="out/concat_top10.txt", help="Concat file for ffmpeg")
    parser.add_argument("--out", default="out/top10.mp4", help="Concatenated Top-10 video output")
    parser.add_argument("--csv-out", help="Optional CSV summary of selected windows")
    parser.add_argument("--n", type=int, default=10, help="Maximum number of clips to keep")
    parser.add_argument("--pre", type=float, default=2.0, help="Seconds to pad before each window")
    parser.add_argument("--post", type=float, default=3.0, help="Seconds to pad after each window")
    parser.add_argument("--min-len", type=float, default=0.8, help="Minimum clip duration after padding")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Drop clips that overlap an accepted one by more than this fraction",
    )
    parser.add_argument("--cheer-max", type=int, default=4, help="Maximum number of cheer anchors")
    parser.add_argument("--cheer-gap", type=float, default=45.0, help="Minimum spacing between cheers in seconds")
    parser.add_argument("--cheer-pre", type=float, default=7.5, help="Seconds of context before cheer peak")
    parser.add_argument("--cheer-post", type=float, default=2.5, help="Seconds after cheer peak")
    parser.add_argument("--cheer-min-len", type=float, default=0.8, help="Minimum cheer clip length before padding")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe executable")
    parser.add_argument(
        "--fallback-duration",
        type=float,
        help="Use this duration if ffprobe cannot determine it",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Compute selections but skip ffmpeg rendering (useful for tests)",
    )
    args = parser.parse_args()

    selected = build_cheers_top10(
        video_path=Path(args.video),
        filtered_csv=Path(args.filtered),
        cheers_csv=Path(args.cheers) if args.cheers else None,
        clips_dir=Path(args.clips_dir),
        concat_path=Path(args.concat),
        output_video=Path(args.out),
        max_count=args.n,
        pad_pre=args.pre,
        pad_post=args.post,
        min_length=args.min_len,
        overlap_threshold=args.overlap,
        cheer_max=args.cheer_max,
        cheer_spacing=args.cheer_gap,
        cheer_pre=args.cheer_pre,
        cheer_post=args.cheer_post,
        cheer_min_length=args.cheer_min_len,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        fallback_duration=args.fallback_duration,
        csv_out=Path(args.csv_out) if args.csv_out else None,
        skip_render=args.skip_render,
    )

    if args.skip_render:
        print(f"Selected {len(selected)} clips (render skipped)")
    else:
        print(f"DONE: {args.out}")


if __name__ == "__main__":
    main()

