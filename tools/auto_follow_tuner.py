import json
import numpy as np
import subprocess
import os
from pathlib import Path


def auto_tune_follow(ball_path, out_prefix, width=1920, height=1080, fps=24):

    # ---- Load ball telemetry ----
    rows = []
    with open(ball_path, "r") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)

    xs = np.array([r["cx"] for r in rows])
    ys = np.array([r["cy"] for r in rows])

    # ---- Derivatives ----
    vx = np.diff(xs) * fps
    vy = np.diff(ys) * fps
    v = np.sqrt(vx*vx + vy*vy)

    ax = np.diff(vx) * fps
    ay = np.diff(vy) * fps
    a = np.sqrt(ax*ax + ay*ay)

    # ---- Stats ----
    max_v = np.percentile(v, 95)    # peak velocity but robust
    max_a = np.percentile(a, 95)    # peak acceleration

    # ---- Compute tuned follow parameters ----
    # Responsiveness (wn): higher if ball moves fast
    wn = np.clip((max_v / 500), 4.0, 14.0)

    # Damping (zeta): smooth but responsive
    zeta = np.clip((1200 / max_v), 0.9, 2.5)

    # Deadzone: based on jitter
    jitter = np.median(np.abs(np.diff(xs)))
    deadzone = np.clip(jitter * 1.8, 2, 12)

    # Zoom-max: widen if ball is risky
    zoom_max = np.clip(1.0 + (max_v / 2000), 1.1, 2.2)

    # Lookahead
    plan_lookahead = np.clip(max_v / 300, 2, 8)

    # ---- Print chosen values ----
    print("AUTO TUNED PARAMS")
    print(f"  WN: {wn:.2f}")
    print(f"  Zeta: {zeta:.2f}")
    print(f"  Deadzone: {deadzone:.2f}")
    print(f"  Zoom-max: {zoom_max:.2f}")
    print(f"  Lookahead: {plan_lookahead:.2f}")

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
        ball_path,
    ]
    subprocess.run(cmd, check=True)

    # ---- Flatten output ----
    flat = f"{out_prefix}.flat.jsonl"
    with open(out_json, "r") as f:
        root = json.load(f)

    with open(flat, "w") as f:
        for kf in root["keyframes"]:
            f.write(json.dumps(kf) + "\n")

    return flat
