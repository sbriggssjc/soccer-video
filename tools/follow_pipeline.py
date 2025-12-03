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

