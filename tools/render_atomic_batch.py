import argparse
import subprocess
from pathlib import Path
import sys

from tools.path_naming import build_output_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preset",
        default="segment_smooth",
        help="e.g., segment_smooth (default) or wide_follow",
    )
    p.add_argument("--src-dir", default="out/atomic_clips", help="Root to scan for mp4s")
    p.add_argument(
        "--out-dir", default="out/portrait_reels/clean", help="Where renders go"
    )
    p.add_argument("--pattern", default="*.mp4", help="Glob pattern for clips")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    p.add_argument(
        "--skip-existing", action="store_true", help="Skip if output already exists"
    )
    p.add_argument(
        "--no-clobber",
        action="store_true",
        help="Skip rendering if output exists and is non-empty.",
    )
    p.add_argument("--portrait", default="1080x1920", help="Portrait output size")
    p.add_argument(
        "--debug-ball",
        action="store_true",
        help="Enable ball debug overlays for render_follow_unified.",
    )
    p.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep scratch artifacts under out/_scratch after rendering.",
    )
    p.add_argument(
        "--scratch-root",
        help="Root directory for scratch artifacts (default: out/_scratch).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(src_root.rglob(args.pattern))
    if args.limit and args.limit > 0:
        clips = clips[: args.limit]

    if not clips:
        print(f"[WARN] No clips found under {src_root} matching {args.pattern}")
        return 2

    ok = 0
    skipped = 0
    failed = 0

    for i, clip in enumerate(clips, 1):
        stem = clip.stem
        preset_label = args.preset.upper()
        # Deterministic naming: reruns overwrite.
        output_name = build_output_name(
            input_path=str(clip),
            preset=preset_label,
            portrait=args.portrait,
            follow=None,
            is_final=True,
            extra_tags=[],
        )
        out_path = out_dir / output_name

        skip_existing = args.skip_existing or args.no_clobber
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            print(f"[SKIP] {i}/{len(clips)} exists: {out_path.name}")
            skipped += 1
            continue

        cmd = [
            sys.executable,
            str(Path("tools") / "render_follow_unified.py"),
            "--preset",
            args.preset,
            "--in",
            str(clip),
            "--out",
            str(out_path),
            "--portrait",
            args.portrait,
        ]
        if args.no_clobber:
            cmd.append("--no-clobber")
        if args.keep_scratch:
            cmd.append("--keep-scratch")
        if args.scratch_root:
            cmd.extend(["--scratch-root", args.scratch_root])
        if args.debug_ball:
            cmd.extend(["--draw-ball", "--debug-ball-overlay", "--use-ball-telemetry"])
        else:
            cmd.append("--no-draw-ball")

        print(f"[RUN] {i}/{len(clips)} {clip.name}")
        print("[CMD]", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {clip} -> {e}")
            failed += 1

    print(f"\n[DONE] ok={ok} skipped={skipped} failed={failed} total={len(clips)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
