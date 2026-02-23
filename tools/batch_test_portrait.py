#!/usr/bin/env python3
"""Batch test portrait reels across diverse clip types.

Runs render_follow_unified.py for a selection of clips with --diagnostics,
capturing ball-in-crop metrics and fusion diagnostics to verify that the
shot-hold / pan-hold / goal-infer changes generalise.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLIPS_DIR = REPO / "out" / "atomic_clips" / "2026-02-23__TSC_vs_Greenwood"
OUT_DIR = REPO / "out" / "portrait_reels" / "batch_test"
RENDERER = REPO / "tools" / "render_follow_unified.py"

# Map event tags in filenames -> --event-type values
EVENT_MAP = {
    "GOAL": "GOAL",
    "CROSS": "CROSS",
    "SHOT": None,       # no special event type
    "DEFENSE": None,
    "BUILD": None,
    "PRESSURE": None,
    "SAVE": None,
    "LONG_BALL": None,
    "NUTMEG": None,
    "SKILL": None,
    "CORNER": None,
    "COUNTER": None,
    "FREE_KICK": "FREE_KICK",
}


def extract_event_type(clip_name: str) -> str | None:
    """Extract the most specific event type from a clip filename."""
    upper = clip_name.upper()
    # Priority: GOAL > CROSS > FREE_KICK > None
    if "GOAL" in upper:
        return "GOAL"
    if "CROSS" in upper:
        return "CROSS"
    if "FREE_KICK" in upper:
        return "FREE_KICK"
    return None


# Diverse selection of clips to test
TEST_CLIPS = [
    "001__2026-02-23__TSC_vs_Greenwood__NUTMEG_CROSS_AND_SHOT__t78.73-t90.70.mp4",
    "004__2026-02-23__TSC_vs_Greenwood__LONG_BALL__t547.00-t553.00.mp4",
    "012__2026-02-23__TSC_vs_Greenwood__CORNER_AND_GOAL__t1373.00-t1381.00.mp4",
    "014__2026-02-23__TSC_vs_Greenwood__PRESSURE_AND_GOAL__t1649.00-t1671.00.mp4",
    "016__2026-02-23__TSC_vs_Greenwood__PRESSURE_AND_CROSS__t1880.00-t1911.00.mp4",
    "019__2026-02-23__TSC_vs_Greenwood__DEFENSE_COUNTER_AND_GOAL__t2039.00-t2059.00.mp4",
    "020__2026-02-23__TSC_vs_Greenwood__SAVE__t2287.00-t2295.00.mp4",
    "021__2026-02-23__TSC_vs_Greenwood__BUILD_PRESSURE_AND_SHOT__t2319.00-t2359.00.mp4",
]


def run_clip(clip_name: str) -> dict:
    """Render one clip and return metrics parsed from stdout."""
    clip_path = CLIPS_DIR / clip_name
    stem = Path(clip_name).stem
    out_path = OUT_DIR / f"{stem}__portrait_test.mp4"

    event_type = extract_event_type(clip_name)

    cmd = [
        sys.executable,
        str(RENDERER),
        "--in", str(clip_path),
        "--src", str(clip_path),
        "--out", str(out_path),
        "--portrait", "1080x1920",
        "--preset", "cinematic",
        "--fps", "24",
        "--diagnostics",
    ]
    if event_type:
        cmd.extend(["--event-type", event_type])

    print(f"\n{'='*70}")
    print(f"CLIP: {clip_name}")
    print(f"EVENT TYPE: {event_type or '(none)'}")
    print(f"{'='*70}")

    result = {
        "clip": clip_name,
        "stem": stem,
        "event_type": event_type,
        "ball_in_crop_pct": None,
        "ball_in_crop_frac": None,
        "max_escape_px": None,
        "yolo_confirmed_pct": None,
        "validation": None,
        "shot_hold": False,
        "pan_hold": False,
        "goal_infer": False,
        "hold_exit": False,
        "warnings": [],
        "exit_code": None,
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            encoding="utf-8",
            errors="replace",
        )
        result["exit_code"] = proc.returncode
        output = proc.stdout + "\n" + proc.stderr

        # Print relevant diagnostic lines
        for line in output.split("\n"):
            line_s = line.strip()
            if any(kw in line_s for kw in [
                "[DIAG]", "[FUSION]", "[GOAL", "[TRIM]",
                "Ball in crop", "Max escape", "Validation",
                "YOLO-confirmed", "WARNING", "PASSED", "FAILED",
                "Shot-hold", "Pan-hold", "hold_exit",
            ]):
                print(f"  {line_s}")

        # Parse metrics from output
        m = re.search(r"Ball in crop:\s*([\d.]+)%\s*\((\d+)/(\d+)\)", output)
        if m:
            result["ball_in_crop_pct"] = float(m.group(1))
            result["ball_in_crop_frac"] = f"{m.group(2)}/{m.group(3)}"

        m = re.search(r"Max escape:\s*([\d.]+)\s*px", output)
        if m:
            result["max_escape_px"] = float(m.group(1))

        m = re.search(r"YOLO-confirmed in crop:\s*([\d.]+)%", output)
        if m:
            result["yolo_confirmed_pct"] = float(m.group(1))

        if "PASSED" in output:
            result["validation"] = "PASSED"
        elif "FAILED" in output:
            result["validation"] = "FAILED"

        if "Shot-hold:" in output:
            result["shot_hold"] = True
        if "Pan-hold:" in output:
            result["pan_hold"] = True
        if "GOAL-INFER" in output or "goal_event_inferred" in output:
            result["goal_infer"] = True
        if "hold_exit=" in output:
            result["hold_exit"] = True

        for line in output.split("\n"):
            if "WARNING" in line:
                result["warnings"].append(line.strip())

    except subprocess.TimeoutExpired:
        result["exit_code"] = -1
        print("  TIMEOUT after 600s!")
    except Exception as e:
        result["exit_code"] = -2
        print(f"  ERROR: {e}")

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for clip_name in TEST_CLIPS:
        r = run_clip(clip_name)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*90}")
    print("BATCH TEST SUMMARY")
    print(f"{'='*90}")
    print(f"{'Clip':<12} {'Event':<8} {'BallInCrop':>10} {'MaxEsc':>8} {'YOLOConf':>10} {'Valid':>8} {'Flags'}")
    print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*25}")

    pass_count = 0
    fail_count = 0
    for r in results:
        flags = []
        if r["shot_hold"]:
            flags.append("SH")
        if r["pan_hold"]:
            flags.append("PH")
        if r["goal_infer"]:
            flags.append("GI")
        if r["hold_exit"]:
            flags.append("HE")

        bic = f"{r['ball_in_crop_pct']:.1f}%" if r['ball_in_crop_pct'] is not None else "N/A"
        esc = f"{r['max_escape_px']:.0f}px" if r['max_escape_px'] is not None else "N/A"
        yc = f"{r['yolo_confirmed_pct']:.1f}%" if r['yolo_confirmed_pct'] is not None else "N/A"
        v = r['validation'] or "N/A"

        clip_id = r['stem'][:11]
        evt = r['event_type'] or "-"

        if v == "PASSED":
            pass_count += 1
        elif v == "FAILED":
            fail_count += 1

        print(f"{clip_id:<12} {evt:<8} {bic:>10} {esc:>8} {yc:>10} {v:>8} {', '.join(flags)}")

    print(f"\nTotal: {len(results)} clips | PASSED: {pass_count} | FAILED: {fail_count}")

    if fail_count > 0:
        print("\nFAILED clips need investigation!")
        for r in results:
            if r['validation'] == 'FAILED':
                print(f"  - {r['stem']}: ball_in_crop={r['ball_in_crop_pct']}%, max_escape={r['max_escape_px']}px")
                for w in r['warnings']:
                    print(f"    {w}")

    # Exit code = number of failures
    sys.exit(fail_count)


if __name__ == "__main__":
    main()
