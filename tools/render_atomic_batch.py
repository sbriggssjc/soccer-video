import argparse
import subprocess
from pathlib import Path
import sys


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
    p.add_argument("--portrait", default="1080x1920", help="Portrait output size")
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
        out_path = out_dir / f"{stem}__{args.preset.upper()}_portrait_FINAL.mp4"

        if args.skip_existing and out_path.exists():
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
            "--draw-ball",
        ]

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
