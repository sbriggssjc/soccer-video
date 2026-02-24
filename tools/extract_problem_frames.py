"""Extract problem frames from a rendered clip for manual ball annotation.

Reads the diagnostic CSV from a rendered clip, identifies frames where
the spline camera path has gaps (no real ball detection), and extracts
those frames as annotated images showing:
  - The current crop window (green box)
  - The fused ball position (red dot — may be wrong)
  - Frame number and source label

Also generates a template CSV for manual anchors that can be filled in
and passed to --manual-anchors on re-render.

Usage:
    python tools/extract_problem_frames.py \\
        --diag  out/portrait_reels/.../001__fix_v8_spline.diag.csv \\
        --video out/atomic_clips/.../001__....mp4 \\
        --out   out/portrait_reels/.../001__problem_frames/

    This creates:
        001__problem_frames/
            frame_280.jpg      # annotated source frame
            frame_285.jpg
            ...
            manual_anchors_template.csv   # fill in ball_x, ball_y
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def load_diag_csv(path: Path) -> list[dict]:
    """Load diagnostic CSV into list of dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_problem_frames(
    rows: list[dict],
    *,
    gap_threshold: int = 10,
    trailing_threshold: int = 20,
    anchor_sources: set[str] = None,
) -> list[dict]:
    """Find frames that likely need manual anchors.

    Returns list of dicts with 'frame', 'reason', and existing data.
    """
    if anchor_sources is None:
        anchor_sources = {"yolo", "blended", "tracker"}

    n = len(rows)
    if n == 0:
        return []

    # Find anchor frame indices
    anchor_indices = []
    for i, row in enumerate(rows):
        src = row.get("source", "").strip().lower()
        if src in anchor_sources:
            anchor_indices.append(i)

    if not anchor_indices:
        # No anchors at all — flag every 30th frame
        return [
            {"frame": i, "reason": "no_anchors", **rows[i]}
            for i in range(0, n, 30)
        ]

    problems = []

    # 1) Interior gaps between anchors
    for gi in range(len(anchor_indices) - 1):
        gap_start = anchor_indices[gi]
        gap_end = anchor_indices[gi + 1]
        gap_len = gap_end - gap_start
        if gap_len > gap_threshold:
            # Sample frames within the gap (midpoint + quartiles)
            mid = (gap_start + gap_end) // 2
            q1 = (gap_start + mid) // 2
            q3 = (mid + gap_end) // 2
            for fi in sorted(set([q1, mid, q3])):
                if gap_start < fi < gap_end:
                    problems.append({
                        "frame": fi,
                        "reason": f"gap_{gap_len}f (f{gap_start}-f{gap_end})",
                        **rows[fi],
                    })

    # 2) Leading hold (before first anchor)
    first_anchor = anchor_indices[0]
    if first_anchor > trailing_threshold:
        mid = first_anchor // 2
        problems.insert(0, {
            "frame": mid,
            "reason": f"leading_hold_{first_anchor}f",
            **rows[mid],
        })

    # 3) Trailing hold (after last anchor)
    last_anchor = anchor_indices[-1]
    trailing = n - 1 - last_anchor
    if trailing > trailing_threshold:
        # Sample several frames in trailing region
        trail_frames = np.linspace(last_anchor + 10, n - 10, min(5, trailing // 20), dtype=int)
        for fi in trail_frames:
            fi = int(fi)
            if 0 <= fi < n:
                problems.append({
                    "frame": fi,
                    "reason": f"trailing_hold_{trailing}f",
                    **rows[fi],
                })

    # 4) Frames where ball escapes crop (ball_in_crop == "0") near anchors
    for i, row in enumerate(rows):
        if row.get("ball_in_crop", "1") == "0":
            src = row.get("source", "").strip().lower()
            if src in anchor_sources:
                problems.append({
                    "frame": i,
                    "reason": "anchor_escape",
                    **rows[i],
                })

    # De-duplicate and sort
    seen = set()
    unique = []
    for p in sorted(problems, key=lambda x: x["frame"]):
        if p["frame"] not in seen:
            seen.add(p["frame"])
            unique.append(p)

    return unique


def extract_frames(
    video_path: Path,
    problem_frames: list[dict],
    out_dir: Path,
):
    """Extract annotated frames from video and save as images."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {src_w}x{src_h}, {total} frames")

    for pf in problem_frames:
        fi = int(pf["frame"])
        if fi >= total:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw crop window (green rectangle)
        try:
            cx0 = float(pf.get("crop_x0", 0))
            cy0 = float(pf.get("crop_y0", 0))
            cw = float(pf.get("crop_w", 0))
            ch = float(pf.get("crop_h", 0))
            if cw > 0 and ch > 0:
                cv2.rectangle(
                    frame,
                    (int(cx0), int(cy0)),
                    (int(cx0 + cw), int(cy0 + ch)),
                    (0, 255, 0), 3,
                )
        except (ValueError, TypeError):
            pass

        # Draw fused ball position (red dot — may be wrong)
        try:
            bx = float(pf.get("ball_x", 0))
            by = float(pf.get("ball_y", 0))
            if bx > 0 and by > 0:
                cv2.circle(frame, (int(bx), int(by)), 12, (0, 0, 255), -1)
                cv2.circle(frame, (int(bx), int(by)), 14, (255, 255, 255), 2)
        except (ValueError, TypeError):
            pass

        # Draw camera center (blue crosshair)
        try:
            cam_cx = float(pf.get("cam_cx", 0))
            cam_cy = float(pf.get("cam_cy", 0))
            if cam_cx > 0 and cam_cy > 0:
                cv2.drawMarker(
                    frame, (int(cam_cx), int(cam_cy)),
                    (255, 180, 0), cv2.MARKER_CROSS, 30, 2,
                )
        except (ValueError, TypeError):
            pass

        # Text overlay
        source = pf.get("source", "?")
        conf = pf.get("confidence", "?")
        reason = pf.get("reason", "?")
        label = f"f{fi} | src={source} conf={conf} | {reason}"
        cv2.putText(
            frame, label, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3,
        )
        cv2.putText(
            frame, "RED=fused ball (may be wrong)  GREEN=crop  BLUE=camera center",
            (20, src_h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )

        out_path = out_dir / f"frame_{fi:04d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Saved {out_path.name}: {reason}")

    cap.release()


def generate_template(problem_frames: list[dict], out_dir: Path):
    """Generate a template CSV for manual anchors."""
    template_path = out_dir / "manual_anchors_template.csv"
    with open(template_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "ball_x", "ball_y"])
        for pf in problem_frames:
            fi = pf["frame"]
            # Pre-fill with fused position as starting estimate
            bx = pf.get("ball_x", "")
            by = pf.get("ball_y", "")
            writer.writerow([fi, bx, by])

    print(f"\nTemplate: {template_path}")
    print("Instructions:")
    print("  1. Open the extracted frame images to see the actual video")
    print("  2. Find the ball in each frame image")
    print("  3. Update ball_x and ball_y in the template CSV")
    print("  4. Delete rows where you can't confidently identify the ball")
    print("  5. Re-render with: --manual-anchors <path_to_csv>")


def main():
    parser = argparse.ArgumentParser(
        description="Extract problem frames for manual ball annotation"
    )
    parser.add_argument(
        "--diag", required=True,
        help="Path to diagnostic CSV from rendered clip",
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to source video (atomic clip)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for extracted frames and template",
    )
    parser.add_argument(
        "--gap-threshold", type=int, default=10,
        help="Minimum gap size (frames) to flag as problem (default: 10)",
    )
    parser.add_argument(
        "--trailing-threshold", type=int, default=20,
        help="Minimum trailing hold (frames) to flag (default: 20)",
    )
    args = parser.parse_args()

    diag_path = Path(args.diag)
    video_path = Path(args.video)
    out_dir = Path(args.out)

    if not diag_path.exists():
        print(f"ERROR: Diagnostic CSV not found: {diag_path}")
        sys.exit(1)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print(f"Loading diagnostics: {diag_path.name}")
    rows = load_diag_csv(diag_path)
    print(f"  {len(rows)} frames")

    print(f"\nFinding problem frames (gap>{args.gap_threshold}f, trailing>{args.trailing_threshold}f)...")
    problems = find_problem_frames(
        rows,
        gap_threshold=args.gap_threshold,
        trailing_threshold=args.trailing_threshold,
    )
    print(f"  Found {len(problems)} problem frames")

    if not problems:
        print("\nNo problem frames found — clip looks clean!")
        return

    print(f"\nExtracting frames from: {video_path.name}")
    extract_frames(video_path, problems, out_dir)

    generate_template(problems, out_dir)
    print(f"\nDone! Review images in {out_dir}")


if __name__ == "__main__":
    main()
