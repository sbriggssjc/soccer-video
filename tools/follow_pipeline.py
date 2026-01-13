#!/usr/bin/env python
"""Simple follow → portrait → branding pipeline wrapper.

This script is a thin orchestrator around ``render_follow_unified.py`` and an
optional PowerShell branding script (tsc_brand.ps1).

Behavior:

- For each ``--clip`` path, derive a clip ID from the filename (without the
  extension), e.g. ``001__SHOT__t155.50-t166.40``.
- Call ``tools/render_follow_unified.py`` with the requested preset and
  portrait geometry to produce a portrait master under:

    out/portrait_reels/clean/<ClipID>__<VARIANT>_portrait_FINAL.mp4

- If ``--brand-script`` is supplied, call that PowerShell script with
  ``-In`` and ``-Out`` to (re)brand the portrait reel.
- If ``--cleanup`` is given, run ``tools/Cleanup-Intermediates.ps1`` at the
  end to sweep known intermediate folders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Ensure ``tools`` can be imported when running the script directly (``python tools/follow_pipeline.py``)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ball telemetry helper
try:
    # If running as a package: python -m tools.follow_pipeline
    from tools.ball_telemetry import telemetry_path_for_video
except ImportError:  # pragma: no cover - fallback for direct script execution
    # If running as a plain script: python tools\follow_pipeline.py
    from ball_telemetry import telemetry_path_for_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run follow → portrait → branding pipeline for one or more clips."
    )
    parser.add_argument(
        "--clip",
        action="append",
        required=True,
        help="Path to an atomic clip to process (can be passed multiple times)",
    )
    parser.add_argument(
        "--preset",
        default="wide_follow",
        help="render_follow_unified preset name (default: wide_follow)",
    )
    parser.add_argument(
        "--portrait",
        default="1080x1920",
        help="Portrait geometry WxH passed through to render_follow_unified (default: 1080x1920)",
    )
    parser.add_argument(
        "--variant",
        default="WIDE",
        help="Variant label used only in the output filename (default: WIDE)",
    )
    parser.add_argument(
        "--brand-script",
        help="Optional PowerShell branding script (e.g. tools/tsc_brand.ps1)",
    )
    parser.add_argument(
        "--aspect",
        default="9x16",
        help="Aspect ratio string passed to the branding script (default: 9x16)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="If set, run tools/Cleanup-Intermediates.ps1 at the end.",
    )
    parser.add_argument(
        "--use-ball-telemetry",
        action="store_true",
        help="Generate and use ball telemetry for portrait planning",
    )
    parser.add_argument(
        "--telemetry",
        help="Optional telemetry path (overrides discovery/detection)",
    )
    parser.add_argument("--w", type=int, default=None, help="Override source width")
    parser.add_argument("--h", type=int, default=None, help="Override source height")
    parser.add_argument(
        "--follow-override",
        type=str,
        default=None,
        help="Path to offline follow trajectory override (.jsonl).",
    )
    parser.add_argument(
        "--follow-exact",
        action="store_true",
        help="Use override follow telemetry exactly with no smoothing or follow controller.",
    )
    # Accept (but ignore) some legacy flags so existing commands don't break.
    parser.add_argument("--extra", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--init-manual", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--init-t", type=float, default=None, help=argparse.SUPPRESS)

    return parser.parse_args(argv)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _smoothstep(x: float) -> float:
    # 0..1 -> smooth 0..1
    x = _clamp(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _load_first_video_meta(args: argparse.Namespace) -> tuple[int, int]:
    """
    Prefer explicit --w/--h; otherwise fall back to defaults.
    This prevents portrait crops from being computed against a stale/incorrect default.
    """
    if args.w and args.h:
        return args.w, args.h
    # Last resort defaults (will usually be overridden by telemetry)
    w = args.w or 1920
    h = args.h or 1080
    return w, h


def _derive_crop_for_preset(preset_name: str, src_w: int, src_h: int) -> tuple[int, int]:
    """
    Fix: wide_follow rectangle sizing was incorrect when generating portrait follow output.
    We derive a crop box that is portrait by default (9:16) but never exceeds src bounds.
    """
    # "wide_follow" produces portrait reels; keep that consistent.
    if preset_name.lower() in ("wide_follow", "wide_follow_smooth", "portrait_follow", "portrait"):
        target_ar = 9 / 16  # width/height for portrait crop
        # Prefer a crop height that uses most of vertical resolution; keep within src.
        crop_h = int(round(src_h * 0.80))
        crop_h = int(_clamp(crop_h, 360, src_h))
        crop_w = int(round(crop_h * target_ar))
        if crop_w > src_w:
            crop_w = src_w
            crop_h = int(round(crop_w / target_ar))
        # Ensure even numbers (ffmpeg friendliness)
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        return crop_w, crop_h

    # Fallback: keep existing behavior (full frame)
    return src_w, src_h


def _read_ball_track(telemetry_path: Path | None) -> list[tuple[float, float, float, float]]:
    """
    Read ball detections from telemetry jsonl.
    Expected keys in each row: t, ball_x, ball_y, ball_conf, or ball=(x,y).
    Returns list of (t, x, y, conf).
    """
    track: list[tuple[float, float, float, float]] = []
    if not telemetry_path or not os.path.exists(telemetry_path):
        return track
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r.get("t") or r.get("time") or r.get("ts")
            if t is None:
                continue
            conf = r.get("ball_conf", r.get("conf", None))
            bx = r.get("ball_x", None)
            by = r.get("ball_y", None)
            if bx is None or by is None:
                b = r.get("ball", None)
                if isinstance(b, (list, tuple)) and len(b) >= 2:
                    bx, by = b[0], b[1]
            if bx is None or by is None:
                continue
            # If confidence is missing, assume usable
            if conf is None:
                conf = 1.0
            track.append((float(t), float(bx), float(by), float(conf)))
    track.sort(key=lambda z: z[0])
    return track


def _interpolate_ball(
    track: list[tuple[float, float, float, float]],
    t: float,
    *,
    min_conf: float = 0.10,
) -> tuple[float, float, float, bool]:
    """
    Simple linear interpolation between track points. Returns (x,y,conf,found_bool).
    If no usable points, found_bool=False.
    """
    if not track:
        return 0.0, 0.0, 0.0, False
    # Find rightmost <= t
    lo = None
    hi = None
    for i in range(len(track)):
        if track[i][0] <= t:
            lo = track[i]
        if track[i][0] >= t:
            hi = track[i]
            break
    if lo is None:
        lo = track[0]
    if hi is None:
        hi = track[-1]
    t0, x0, y0, c0 = lo
    t1, x1, y1, c1 = hi
    if max(c0, c1) < min_conf:
        return x0, y0, max(c0, c1), False
    if t1 == t0:
        return x0, y0, max(c0, c1), True
    a = (t - t0) / (t1 - t0)
    a = _clamp(a, 0.0, 1.0)
    x = x0 + (x1 - x0) * a
    y = y0 + (y1 - y0) * a
    c = c0 + (c1 - c0) * a
    return x, y, c, True


def _is_inside_crop(
    x: float,
    y: float,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    *,
    margin: float = 0.0,
) -> bool:
    return (
        x >= crop_x + margin
        and x <= crop_x + crop_w - margin
        and y >= crop_y + margin
        and y <= crop_y + crop_h - margin
    )


def _update_camera_center(
    cur_cx: float,
    cur_cy: float,
    target_cx: float,
    target_cy: float,
    dt: float,
    *,
    responsiveness: float = 6.0,
) -> tuple[float, float]:
    """Exponential smoothing toward target center. responsiveness higher -> snappier."""
    # Convert responsiveness to alpha per dt
    alpha = 1.0 - math.exp(-responsiveness * max(dt, 1e-3))
    cur_cx = cur_cx + (target_cx - cur_cx) * alpha
    cur_cy = cur_cy + (target_cy - cur_cy) * alpha
    return cur_cx, cur_cy


def write_follow_keyframes(
    clip: Path,
    preset_name: str,
    args: argparse.Namespace,
    telemetry_path: Path | None,
) -> Path:
    """Write follow keyframes to out/camera for downstream ffmpeg steps."""
    src_w, src_h = _load_first_video_meta(args)
    crop_w, crop_h = _derive_crop_for_preset(preset_name, src_w, src_h)

    # Ball track from telemetry
    ball_track = _read_ball_track(telemetry_path)

    # Parameters tuned for behavior: quick edge-follow, sensible fallback.
    min_ball_conf = 0.10
    edge_margin = 0.03 * min(crop_w, crop_h)  # when ball near edge, bias panning
    offscreen_grace_sec = 0.75  # keep moving briefly when ball disappears
    fallback_pan_speed = 0.65  # how hard we drift in fallback (0..1)

    # Camera center init: frame center
    cam_cx = src_w / 2.0
    cam_cy = src_h / 2.0

    last_t = None
    last_seen_t = None
    # Fallback drift direction based on last ball velocity
    vx = 0.0
    vy = 0.0

    # Generate per-frame camera centers (or keyframes) used downstream by ffmpeg crop/filter
    keyframes: list[dict[str, float]] = []

    # If telemetry has explicit frame times you already use, keep that;
    # otherwise sample a reasonable time grid from ball track.
    if ball_track:
        t_start = ball_track[0][0]
        t_end = ball_track[-1][0]
    else:
        # No telemetry: still emit one static keyframe
        t_start, t_end = 0.0, 0.0

    # Sample at ~30hz unless you already have frame stamps elsewhere.
    step = 1.0 / 30.0
    t = t_start
    prev_ball = None
    while t <= t_end + 1e-6:
        bx, by, _, found = _interpolate_ball(ball_track, t, min_conf=min_ball_conf)

        if last_t is None:
            dt = step
        else:
            dt = max(1e-3, t - last_t)

        if found:
            # Estimate velocity for fallback drift
            if prev_ball is not None:
                pbx, pby = prev_ball
                vx = (bx - pbx) / dt
                vy = (by - pby) / dt
            prev_ball = (bx, by)
            last_seen_t = t

            # Target camera center = ball, but softly biased if ball is offscreen/near-edge
            # Compute current crop placement
            crop_x = cam_cx - crop_w / 2.0
            crop_y = cam_cy - crop_h / 2.0
            # Clamp crop origin to src bounds
            crop_x = _clamp(crop_x, 0.0, src_w - crop_w)
            crop_y = _clamp(crop_y, 0.0, src_h - crop_h)

            if _is_inside_crop(bx, by, crop_x, crop_y, crop_w, crop_h, margin=edge_margin):
                target_cx, target_cy = bx, by
            else:
                # If ball is outside or near edge, move toward it more aggressively
                target_cx, target_cy = bx, by

            cam_cx, cam_cy = _update_camera_center(
                cam_cx,
                cam_cy,
                target_cx,
                target_cy,
                dt,
                responsiveness=7.5,
            )
        else:
            # Ball missing: continue panning in last known direction for a bit
            if last_seen_t is not None and (t - last_seen_t) <= offscreen_grace_sec:
                target_cx = cam_cx + vx * dt
                target_cy = cam_cy + vy * dt
                # Drift strength
                cam_cx, cam_cy = _update_camera_center(
                    cam_cx,
                    cam_cy,
                    target_cx,
                    target_cy,
                    dt,
                    responsiveness=3.0 * fallback_pan_speed,
                )
            else:
                # No ball for a while: slowly recenter
                cam_cx, cam_cy = _update_camera_center(
                    cam_cx,
                    cam_cy,
                    src_w / 2.0,
                    src_h / 2.0,
                    dt,
                    responsiveness=0.8,
                )

        # Keep camera center within bounds so crop stays valid
        cam_cx = _clamp(cam_cx, crop_w / 2.0, src_w - crop_w / 2.0)
        cam_cy = _clamp(cam_cy, crop_h / 2.0, src_h - crop_h / 2.0)

        keyframes.append(
            {
                "t": round(t, 5),
                "cx": cam_cx,
                "cy": cam_cy,
                "crop_w": crop_w,
                "crop_h": crop_h,
            }
        )
        last_t = t
        t += step

    # Write keyframes for downstream ffmpeg step(s)
    out_dir = REPO_ROOT / "out" / "camera"
    out_dir.mkdir(parents=True, exist_ok=True)
    kf_path = out_dir / f"{clip.stem}__{preset_name}__keyframes.json"
    with open(kf_path, "w", encoding="utf-8") as f:
        json.dump({"src_w": src_w, "src_h": src_h, "keyframes": keyframes}, f, indent=2)
    print(f"[OK] wrote keyframes: {kf_path}")
    return kf_path


def run_render(
    clip: Path,
    preset: str,
    portrait: str,
    variant: str,
    *,
    use_ball_telemetry: bool,
    telemetry_path: Path | None,
    follow_override: str | None = None,
    follow_exact: bool = False,
) -> Path:
    """Invoke render_follow_unified.py for a single clip and return portrait path."""

    clip = clip.resolve()
    if not clip.is_file():
        raise FileNotFoundError(f"Clip not found: {clip}")

    clip_id = clip.stem  # e.g. 001__SHOT__t155.50-t166.40

    out_dir = REPO_ROOT / "out" / "portrait_reels" / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{clip_id}__{variant}_portrait_FINAL.mp4"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "render_follow_unified.py"),
        "--preset",
        preset,
        "--in",
        str(clip),
        "--out",
        str(out_path),
        "--portrait",
        portrait,
    ]

    if use_ball_telemetry:
        telemetry_path = (
            Path(telemetry_path).resolve() if telemetry_path else Path(telemetry_path_for_video(clip)).resolve()
        )
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--use-ball-telemetry", "--telemetry", str(telemetry_path)])

    # === FOLLOW OVERRIDE SUPPORT (NEW) ===
    follow_override_path = None
    if follow_override:
        follow_override_path = follow_override

    if follow_override_path:
        cmd.extend(["--follow-override", follow_override_path])

    if follow_exact:
        cmd.append("--follow-exact")

    print(f"[INFO] Rendering {clip_id}")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not out_path.is_file():
        raise RuntimeError(f"Expected portrait output not created: {out_path}")

    return out_path


def run_brand(brand_script: Path, in_path: Path, aspect: str) -> Path:
    """Run the PowerShell branding script on the portrait reel.

    We brand into a temporary ``.__BRANDTMP`` file and, on success, atomically
    replace the original.
    """

    brand_script = brand_script.resolve()
    if not brand_script.is_file():
        raise FileNotFoundError(f"Brand script not found: {brand_script}")

    in_path = in_path.resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Portrait input for branding not found: {in_path}")

    # Preserve the original extension so ffmpeg/PowerShell can infer the
    # container type.  ``Path.with_suffix`` would drop the ``.mp4`` suffix, so we
    # build the filename manually: ``<stem>__BRANDTMP<suffix>``.
    tmp_path = in_path.with_name(in_path.stem + "__BRANDTMP" + in_path.suffix)

    cmd = [
        "pwsh.EXE",
        "-NoProfile",
        "-File",
        str(brand_script),
        "-In",
        str(in_path),
        "-Out",
        str(tmp_path),
        "-Aspect",
        aspect,
    ]

    print(f"[INFO] Branding {in_path.name} → {tmp_path.name}")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not tmp_path.is_file():
        raise RuntimeError(f"Branding script did not produce expected file: {tmp_path}")

    # Swap tmp → final
    in_path.unlink(missing_ok=True)
    tmp_path.rename(in_path)

    return in_path


def run_cleanup() -> None:
    ps1 = REPO_ROOT / "tools" / "Cleanup-Intermediates.ps1"
    if not ps1.is_file():
        print(f"[WARN] Cleanup script not found: {ps1}")
        return
    cmd = ["pwsh.EXE", "-NoProfile", "-File", str(ps1), "-Root", str(REPO_ROOT)]
    print("[INFO] Cleaning intermediates …")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)

    clips = [Path(c) for c in ns.clip]
    ok = 0
    for clip in clips:
        try:
            preset = ns.preset
            telemetry_path: Path | None = Path(ns.telemetry) if ns.telemetry else Path(telemetry_path_for_video(clip))
            use_ball_telemetry = ns.use_ball_telemetry
            if ns.use_ball_telemetry:
                if not telemetry_path.is_file() and ns.telemetry:
                    print(f"[BALL] Telemetry path missing ({telemetry_path}); falling back to reactive follow")
                    use_ball_telemetry = False
                elif not telemetry_path.is_file():
                    detect_cmd = [
                        sys.executable,
                        str(REPO_ROOT / "tools" / "ball_telemetry.py"),
                        "detect",
                        "--video",
                        str(clip),
                        "--out",
                        str(telemetry_path),
                        "--sport",
                        "soccer",
                    ]
                    print(f"[BALL] Detecting telemetry for {clip}")
                    print("[CMD]", " ".join(detect_cmd))
                    result = subprocess.run(detect_cmd)
                    if result.returncode != 0:
                        print(
                            f"[WARN] Telemetry detection failed for {clip}; falling back to reactive follow",
                            file=sys.stderr,
                        )
                        use_ball_telemetry = False

                meta_path = telemetry_path.with_suffix(telemetry_path.suffix + ".meta.json")
                if use_ball_telemetry and meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        ball_cov = float(meta.get("ball_conf_coverage", 0.0))
                        action_cov = float(meta.get("action_valid_coverage", 0.0))
                        if ball_cov < 0.3 and action_cov < 0.6:
                            print(
                                f"[BALL] Weak telemetry (ball_cov={ball_cov:.3f}, action_cov={action_cov:.3f}); "
                                "switching to jerk_follow_fast without telemetry",
                                file=sys.stderr,
                            )
                            use_ball_telemetry = False
                            preset = "jerk_follow_fast"
                    except Exception as exc:  # noqa: BLE001
                        print(f"[WARN] Failed to parse telemetry meta {meta_path}: {exc}")

                if use_ball_telemetry and telemetry_path.is_file():
                    print(f"[BALL] Using telemetry from {telemetry_path}")
                elif ns.use_ball_telemetry:
                    print(f"[BALL] No telemetry; reactive follow only for {clip}")

            if use_ball_telemetry and telemetry_path and telemetry_path.is_file():
                write_follow_keyframes(clip, preset, ns, telemetry_path)

            portrait = run_render(
                clip,
                preset,
                ns.portrait,
                ns.variant,
                use_ball_telemetry=use_ball_telemetry,
                telemetry_path=telemetry_path if use_ball_telemetry else None,
                follow_override=ns.follow_override,
                follow_exact=ns.follow_exact,
            )
            if ns.brand_script:
                run_brand(Path(ns.brand_script), portrait, ns.aspect)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed for {clip}: {exc}", file=sys.stderr)

    print(f"[INFO] {ok}/{len(clips)} clips rendered")

    if ns.cleanup:
        try:
            run_cleanup()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
