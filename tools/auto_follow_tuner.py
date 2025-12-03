import json
import numpy as np
import subprocess
import os
import argparse
from pathlib import Path


def auto_tune_follow(ball_path, out_prefix, width=1920, height=1080, fps=24):
    ball_path = Path(ball_path)

    if not ball_path.exists():
        raise FileNotFoundError(f"Ball telemetry file not found:\n{ball_path}")

    # ---- Load ball telemetry ----
    rows = []
    with open(ball_path, "r") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)

    if not rows:
        raise ValueError("Ball telemetry file contained no rows.")

    xs = np.array([r["cx"] for r in rows])
    ys = np.array([r["cy"] for r in rows])

    # ---- Derivatives ----
    vx = np.diff(xs) * fps
    vy = np.diff(ys) * fps
    v = np.sqrt(vx * vx + vy * vy)

    if len(v) < 3:
        raise ValueError("Telemetry too short to compute velocity/acceleration.")

    ax = np.diff(vx) * fps
    ay = np.diff(vy) * fps
    a = np.sqrt(ax * ax + ay * ay)

    # ---- Stats ----
    max_v = np.percentile(v, 95)    # stable velocity estimate
    max_a = np.percentile(a, 95)    # stable acceleration estimate

    # ---- Auto tuned follow parameters ----
    wn = float(np.clip((max_v / 500), 4.0, 14.0))
    zeta = float(np.clip((1200 / max_v), 0.9, 2.5))
    jitter = np.median(np.abs(np.diff(xs)))
    deadzone = float(np.clip(jitter * 1.8, 2, 12))
    zoom_max = float(np.clip(1.0 + (max_v / 2000), 1.1, 2.2))
    plan_lookahead = float(np.clip(max_v / 300, 2, 8))

    print("\n=== AUTO TUNED FOLLOW PARAMETERS ===")
    print(f"  WN             : {wn:.2f}")
    print(f"  Zeta           : {zeta:.2f}")
    print(f"  Deadzone       : {deadzone:.2f}")
    print(f"  Zoom-max       : {zoom_max:.2f}")
    print(f"  Lookahead      : {plan_lookahead:.2f}")
    print(f"  Output Prefix  : {out_prefix}")
    print("====================================\n")

    # ---- Run offline planner ----
    out_json = f"{out_prefix}.jsonl"
    cmd = [
        "python", "tools/offline_portrait_planner.py",
        "--width", str(width),
        "--height", str(height),
        "--fps", str(fps),
        "--smooth", "0.18",
        "--zoom-max", str(zoom_max),
        "--out", out_json,
        str(ball_path),
    ]

    print("Running offline portrait planner:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    # ---- Flatten output ----
    flat = f"{out_prefix}.flat.jsonl"

    with open(out_json, "r") as f:
        root = json.load(f)

    keyframes = root.get("keyframes", [])
    if not keyframes:
        raise ValueError("Planner output had no keyframes!")

    with open(flat, "w") as f:
        for kf in keyframes:
            f.write(json.dumps(kf) + "\n")

    print(f"\nWrote tuned follow file:")
    print(f"  {flat}\n")

    return flat


# ----------------------------------------------------------
# CLI ENTRYPOINT
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune follow parameters from ball telemetry and generate a follow override path."
    )

    parser.add_argument("ball_path",
                        help="Path to ball telemetry .jsonl file")

    parser.add_argument("out_prefix",
                        help="Output prefix (no extension). "
                             "Script writes prefix.jsonl and prefix.flat.jsonl")

    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)

    args = parser.parse_args()

    auto_tune_follow(
        args.ball_path,
        args.out_prefix,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
