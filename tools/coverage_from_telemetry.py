import argparse
import json
import math
from typing import List, Optional, Sequence, Tuple


# --- robust outward nudge (Ceiling) + slightly bigger slack ---
SLACK = 8          # was 6
HALF = 0.65        # widen local warp window (seconds); was ~0.50–0.60


def load_jsonl(path: str) -> List[object]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


def build_hann_window(radius_frames: int) -> List[float]:
    size = max(1, radius_frames * 2 + 1)
    if size == 1:
        return [1.0]
    return [
        0.5 * (1.0 - math.cos((2.0 * math.pi * idx) / (size - 1)))
        for idx in range(size)
    ]


def build_local_corrections(
    need_dx: Sequence[float],
    need_dy: Sequence[float],
    fps: float,
    half_seconds: float = HALF,
) -> Tuple[List[float], List[float]]:
    total = max(len(need_dx), len(need_dy))
    if total == 0:
        return [], []

    radius = int(round(max(0.0, half_seconds) * max(1.0, fps)))
    window = build_hann_window(radius)

    accum_dx = [0.0] * total
    accum_dy = [0.0] * total
    accum_w = [0.0] * total

    for idx in range(total):
        dx = float(need_dx[idx]) if idx < len(need_dx) else 0.0
        dy = float(need_dy[idx]) if idx < len(need_dy) else 0.0
        if dx == 0.0 and dy == 0.0:
            continue

        for offset in range(-radius, radius + 1):
            j = idx + offset
            if j < 0 or j >= total:
                continue
            weight = window[offset + radius]
            if dx != 0.0:
                accum_dx[j] += dx * weight
            if dy != 0.0:
                accum_dy[j] += dy * weight
            accum_w[j] += weight

    corr_dx = [0.0] * total
    corr_dy = [0.0] * total
    for idx in range(total):
        w = accum_w[idx]
        if w != 0.0:
            corr_dx[idx] = accum_dx[idx] / w
            corr_dy[idx] = accum_dy[idx] / w

    return corr_dx, corr_dy


def _apply_delta(row: dict, key: str, delta: float) -> Optional[float]:
    if key not in row:
        return None
    val = row.get(key)
    if val is None:
        return None
    try:
        new_val = float(val) + float(delta)
    except (TypeError, ValueError):
        return None
    row[key] = new_val
    return new_val


def _mirror_ball_fields(row: dict) -> None:
    has_plain = "bx" in row and "by" in row
    has_stab = "bx_stab" in row and "by_stab" in row
    if not (has_plain and has_stab):
        return

    bx_plain = row.get("bx")
    by_plain = row.get("by")
    bx_stab = row.get("bx_stab")
    by_stab = row.get("by_stab")

    def _choose(primary, secondary):
        if primary is None and secondary is None:
            return None
        if primary is None:
            try:
                return float(secondary)
            except (TypeError, ValueError):
                return None
        try:
            return float(primary)
        except (TypeError, ValueError):
            try:
                return float(secondary)
            except (TypeError, ValueError):
                return None

    bx_val = _choose(bx_plain, bx_stab)
    by_val = _choose(by_plain, by_stab)

    if bx_val is not None:
        row["bx"] = float(bx_val)
        row["bx_stab"] = float(bx_val)
    if by_val is not None:
        row["by"] = float(by_val)
        row["by_stab"] = float(by_val)


def write_plan(
    path: str,
    plan_rows: Sequence[object],
    need_dx: Sequence[float],
    need_dy: Sequence[float],
    fps: float,
) -> None:
    corr_dx, corr_dy = build_local_corrections(need_dx, need_dy, fps)
    total_dx = len(corr_dx)
    total_dy = len(corr_dy)

    with open(path, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(plan_rows):
            if isinstance(rec, dict):
                row = dict(rec)
            else:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                continue

            dx = corr_dx[idx] if idx < total_dx else 0.0
            dy = corr_dy[idx] if idx < total_dy else 0.0

            if dx != 0.0 or dy != 0.0:
                _apply_delta(row, "bx", dx)
                _apply_delta(row, "by", dy)
                _apply_delta(row, "bx_stab", dx)
                _apply_delta(row, "by_stab", dy)

            _mirror_ball_fields(row)

            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--crop-w", type=int, default=486)
    ap.add_argument("--crop-h", type=int, default=864)
    ap.add_argument("--margin", type=int, default=90)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--plan", dest="plan_in", default=None)
    ap.add_argument("--plan-out", dest="plan_out", default=None)
    args = ap.parse_args()

    recs = load_jsonl(args.telemetry)
    inside = 0
    total = 0
    misses = []

    crop_w = args.crop_w
    crop_h = args.crop_h
    margin = args.margin
    W = args.w
    H = args.h

    for i, r in enumerate(recs):
        if r is None:
            continue
        if not all(k in r for k in ("cx", "cy", "bx", "by")):
            continue
        total += 1
        cx = float(r["cx"])
        cy = float(r["cy"])
        bx = float(r["bx"])
        by = float(r["by"])
        z_out = float(r.get("zoom_out", 1.0))
        z_in = float(r.get("zoom", 1.0))
        z_eff = max(z_out, 1.0 / max(z_in, 1e-6))
        z = max(1.0, z_eff)

        eff_w = min(W, crop_w * z)
        eff_h = min(H, crop_h * z)

        x0 = max(0.0, min(W - eff_w, cx - eff_w * 0.5))
        y0 = max(0.0, min(H - eff_h, cy - eff_h * 0.5))
        x1 = x0 + eff_w
        y1 = y0 + eff_h

        left = x0 + margin
        right = x1 - margin
        top = y0 + margin
        bottom = y1 - margin

        ok = bx >= left and bx <= right and by >= top and by <= bottom
        if ok:
            inside += 1
        else:
            t = float(r.get("t", i / 30.0))
            need_dx = 0
            need_dy = 0
            if bx < left:
                need_dx = int(math.ceil((left + SLACK) - bx))
            elif bx > right:
                need_dx = int(math.ceil((right - SLACK) - bx))

            if by < top:
                need_dy = int(math.ceil((top + SLACK) - by))
            elif by > bottom:
                need_dy = int(math.ceil((bottom - SLACK) - by))

            misses.append((i, t, bx, by, need_dx, need_dy))

    need_dx_series = [0.0] * len(recs)
    need_dy_series = [0.0] * len(recs)
    for idx, _t, _bx, _by, dx, dy in misses:
        if 0 <= idx < len(need_dx_series):
            need_dx_series[idx] = float(dx)
        if 0 <= idx < len(need_dy_series):
            need_dy_series[idx] = float(dy)

    pct = 100.0 * inside / max(1, total)
    print(
        f"camera-coverage: {inside}/{total} = {pct:.2f}% (crop {args.crop_w}x{args.crop_h}, margin {args.margin}px; zoom-aware)"
    )
    if misses:
        i, t, bx, by, first_need_dx, first_need_dy = misses[0]
        print(
            "first miss @ frame",
            f" {i} t={t:.2f}s  ball=({bx:.1f},{by:.1f})",
            f" need_dx={first_need_dx} need_dy={first_need_dy}"
        )

    if args.plan_in or args.plan_out:
        if not args.plan_in or not args.plan_out:
            raise SystemExit("Both --plan and --plan-out must be provided to write a corrected plan.")

        plan_rows = load_jsonl(args.plan_in)
        if not plan_rows:
            raise SystemExit(f"Plan input {args.plan_in} is empty or unreadable.")

        write_plan(args.plan_out, plan_rows, need_dx_series, need_dy_series, args.fps)
        print(f"wrote corrected plan to {args.plan_out}")


if __name__ == "__main__":
    main()
