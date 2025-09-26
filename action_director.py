#!/usr/bin/env python3
"""Automatic action director for soccer highlight phases.

This script analyses tracking CSVs to estimate key event timestamps
(throw-in, shot and celebration window) and generates both a JSON
"recipe" and a ready-to-use ffmpeg filter graph tailored to those
phases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PRESETS = {
    "throw_in": {"w_forced": 720, "z_shot": 1.08, "z_cele": 1.18, "lead": 1.2, "trail": 0.8},
    "corner": {"w_forced": 680, "z_shot": 1.10, "z_cele": 1.20, "lead": 1.0, "trail": 1.2},
    "goal": {"w_forced": 700, "z_shot": 1.12, "z_cele": 1.22, "lead": 0.7, "trail": 1.1},
}


def smooth(series: pd.Series, window: int = 5) -> pd.Series:
    """Median smooth a series with a centered rolling window."""
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1, center=True).median()


def derive_vel(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Ensure velocity and speed columns exist on the dataframe."""
    df = df.sort_values("frame").reset_index(drop=True)
    dt = 1.0 / fps

    have_vx = "ball_vx" in df.columns
    have_vy = "ball_vy" in df.columns
    if not (have_vx and have_vy):
        for axis in ("x", "y"):
            vel = df[f"ball_{axis}"].diff().fillna(0) / dt
            df[f"ball_v{axis}"] = smooth(vel, 5)

    if "ball_speed" not in df.columns:
        df["ball_speed"] = np.hypot(df["ball_vx"], df["ball_vy"])

    return df


def even_floor(value: float) -> int:
    """Return the largest even integer not greater than value."""
    return int(np.floor(value / 2.0) * 2)


def estimate_vertical_band(
    df: pd.DataFrame,
    image_height: int,
    t_start: float,
    t_end: float,
    pad: float = 60.0,
    min_band: float = 420.0,
) -> tuple[float, float]:
    """Estimate a vertical band [y_min, y_max] that covers the action."""
    mask = (df.t >= t_start) & (df.t <= t_end)
    values = df.loc[mask, "ball_y"].dropna()
    if values.empty:
        values = df["ball_y"].dropna()
    if values.empty:
        return 0.0, float(image_height)

    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 90))
    y_min = max(0.0, low - pad)
    y_max = min(float(image_height), high + pad)

    band = y_max - y_min
    if band < min_band:
        center = float(values.median())
        half = min(float(image_height) / 2.0, max(min_band / 2.0, band / 2.0))
        y_min = max(0.0, center - half)
        y_max = min(float(image_height), center + half)

    if y_max <= y_min:
        center = float(values.median()) if not values.empty else image_height / 2.0
        y_min = max(0.0, center - image_height / 2.0)
        y_max = min(float(image_height), center + image_height / 2.0)

    return y_min, y_max


def median_ball_y(df: pd.DataFrame, t_start: float, t_end: float) -> Optional[float]:
    """Median y position of the ball within a time window."""
    mask = (df.t >= t_start) & (df.t <= t_end)
    values = df.loc[mask, "ball_y"].dropna()
    if values.empty:
        return None
    return float(values.median())


def detect_throw_in(
    df: pd.DataFrame,
    image_width: int,
    margin: float = 0.03,
    vx_inward: float = 150.0,
) -> Optional[float]:
    """Detect throw-in moment near touchlines with inward velocity spike."""
    left_bound = image_width * margin
    right_bound = image_width * (1 - margin)
    near_edge = (df.ball_x <= left_bound) | (df.ball_x >= right_bound)

    candidates = df.loc[near_edge].copy()
    if candidates.empty:
        return None

    hits: list[pd.Series] = []
    left_hits = candidates.loc[candidates.ball_x <= left_bound]
    if not left_hits.empty:
        inward = left_hits.loc[left_hits.ball_vx > vx_inward]
        if not inward.empty:
            hits.append(inward.iloc[0])

    right_hits = candidates.loc[candidates.ball_x >= right_bound]
    if not right_hits.empty:
        inward = right_hits.loc[right_hits.ball_vx < -vx_inward]
        if not inward.empty:
            hits.append(inward.iloc[0])

    if not hits:
        return None

    best = min(hits, key=lambda row: row.t)
    return float(best.t)


def detect_shot(
    df: pd.DataFrame,
    goal_x: float,
    goal_y: float,
    speed_mult: float = 1.7,
    dir_tol_deg: float = 20.0,
) -> Optional[float]:
    """Detect the shot moment from a speed surge aimed toward the goal."""
    if "ball_speed" not in df.columns:
        return None

    speed = smooth(df["ball_speed"], 7)
    baseline = smooth(speed, 25) + 1e-6
    surge = speed > (speed_mult * baseline)

    if "ball_vx" not in df.columns or "ball_vy" not in df.columns:
        idx = speed.idxmax()
        return float(df.loc[idx, "t"]) if pd.notna(idx) else None

    vec_goal = np.stack([goal_x - df.ball_x, goal_y - df.ball_y], axis=1)
    vec_vel = np.stack([df.ball_vx, df.ball_vy], axis=1)
    dot = (vec_goal * vec_vel).sum(axis=1)
    goal_norm = np.linalg.norm(vec_goal, axis=1) + 1e-6
    vel_norm = np.linalg.norm(vec_vel, axis=1) + 1e-6
    cos_ang = np.clip(dot / (goal_norm * vel_norm), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_ang))
    direction_ok = ang < dir_tol_deg

    candidates = df.index[surge & direction_ok]
    if len(candidates) == 0:
        idx = speed.idxmax()
        return float(df.loc[idx, "t"]) if pd.notna(idx) else None

    idx = candidates[0]
    return float(df.loc[idx, "t"])


def detect_cele_end(
    df: pd.DataFrame,
    t_start: float,
    max_ms: int = 2600,
) -> float:
    """Estimate celebration end via a simple capped window."""
    clip_end = float(df.t.max())
    t_end = min(clip_end, t_start + max_ms / 1000.0)
    return t_end


def build_ffmpeg_phase_filter(
    image_width: int,
    image_height: int,
    clip_start: float,
    clip_end: float,
    t_phase_begin: float,
    t_phase_end: float,
    midx: float,
    w_forced: int,
    z_shot: float,
    z_cele: float,
    shot_center_y: float,
    cele_center_y: float,
    y_min: float,
    y_max: float,
    fade_duration: float,
) -> str:
    """Construct an ffmpeg filter graph for the action phases."""

    scaled_height = 1080.0
    scaled_width = image_width * (scaled_height / float(image_height))

    pre_len = max(0.0, t_phase_begin - clip_start)
    mid_len = max(0.0, t_phase_end - t_phase_begin)
    post_len = max(0.0, clip_end - t_phase_end)

    positive_lengths = [length for length in (pre_len, mid_len, post_len) if length > 0]
    if len(positive_lengths) < 2:
        fade = 0.0
    else:
        fade_candidates = [fade_duration, *positive_lengths, *(length * 0.5 for length in positive_lengths)]
        fade = max(0.05, min(fade_candidates))

    shot_crop_h = max(16, even_floor(scaled_height / z_shot))
    shot_crop_w = max(16, even_floor((scaled_height * 9.0 / 16.0) / z_shot))
    cele_crop_h = max(16, even_floor(scaled_height / z_cele))
    cele_crop_w = max(16, even_floor((scaled_height * 9.0 / 16.0) / z_cele))

    pre_x = float(np.clip(midx - w_forced / 2.0, 0.0, scaled_width - w_forced))

    shot_lower = y_min + shot_crop_h / 2.0
    shot_upper = y_max - shot_crop_h / 2.0
    if shot_upper < shot_lower:
        center_default = float(np.clip((y_min + y_max) / 2.0, shot_crop_h / 2.0, scaled_height - shot_crop_h / 2.0))
        shot_lower = shot_upper = center_default
    shot_center = float(np.clip(shot_center_y, shot_lower, shot_upper))
    shot_y = float(np.clip(shot_center - shot_crop_h / 2.0, 0.0, scaled_height - shot_crop_h))
    shot_x = float(np.clip(midx - shot_crop_w / 2.0, 0.0, scaled_width - shot_crop_w))

    cele_lower = y_min + cele_crop_h / 2.0
    cele_upper = y_max - cele_crop_h / 2.0
    if cele_upper < cele_lower:
        center_default = float(np.clip((y_min + y_max) / 2.0, cele_crop_h / 2.0, scaled_height - cele_crop_h / 2.0))
        cele_lower = cele_upper = center_default
    cele_center = float(np.clip(cele_center_y, cele_lower, cele_upper))
    cele_y = float(np.clip(cele_center - cele_crop_h / 2.0, 0.0, scaled_height - cele_crop_h))
    cele_x = float(np.clip(midx - cele_crop_w / 2.0, 0.0, scaled_width - cele_crop_w))

    if fade <= 0:
        offset_first = pre_len
        offset_second = pre_len + mid_len
    else:
        offset_first = max(0.0, pre_len - fade)
        offset_second = max(0.0, pre_len + mid_len - 2 * fade)

    filter_graph = f"""
[0:v]split=3[v0][v1][v2];[0:a]asplit=3[a0][a1][a2];
[v0]trim=start=0:end={t_phase_begin:.3f},setpts=PTS-STARTPTS,
scale=w=-2:h=1080:flags=lanczos,setsar=1,
crop=w={w_forced}:h=1080:x={pre_x:.3f}:y='(ih-1080)/2',format=yuv420p[v0o];
[a0]atrim=start=0:end={t_phase_begin:.3f},asetpts=PTS-STARTPTS[a0o];
[v1]trim=start={t_phase_begin:.3f}:end={t_phase_end:.3f},setpts=PTS-STARTPTS,
scale=w=-2:h=1080:flags=lanczos,setsar=1,
crop=w={shot_crop_w}:h={shot_crop_h}:x={shot_x:.3f}:y={shot_y:.3f},format=yuv420p[v1o];
[a1]atrim=start={t_phase_begin:.3f}:end={t_phase_end:.3f},asetpts=PTS-STARTPTS[a1o];
[v2]trim=start={t_phase_end:.3f},setpts=PTS-STARTPTS,
scale=w=-2:h=1080:flags=lanczos,setsar=1,
crop=w={cele_crop_w}:h={cele_crop_h}:x={cele_x:.3f}:y={cele_y:.3f},format=yuv420p[v2o];
[a2]atrim=start={t_phase_end:.3f},asetpts=PTS-STARTPTS[a2o];
[v0o][v1o]xfade=transition=fade:duration={fade:.3f}:offset={offset_first:.3f}[v01];
[a0o][a1o]acrossfade=d={fade:.3f}:o={offset_first:.3f}[a01];
[v01][v2o]xfade=transition=fade:duration={fade:.3f}:offset={offset_second:.3f}[v];
[a01][a2o]acrossfade=d={fade:.3f}:o={offset_second:.3f}[a]
"""

    return "".join(line.strip() for line in filter_graph.strip().splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic action director")
    parser.add_argument(
        "--csv",
        required=True,
        help="Tracking CSV with columns: frame,t,ball_x,ball_y[,ball_vx,ball_vy]",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--iw", type=int, default=1920)
    parser.add_argument("--ih", type=int, default=1080)
    parser.add_argument("--goal_left", type=float, default=840)
    parser.add_argument("--goal_right", type=float, default=1080)
    parser.add_argument("--action", choices=sorted(PRESETS.keys()), default="goal")
    parser.add_argument("--out_recipe", required=True)
    parser.add_argument("--out_ffmpeg", required=True)
    args = parser.parse_args()

    df = pd.read_csv(
        args.csv,
        comment="#",
        sep=None,
        engine="python",
        on_bad_lines="skip",
    )
    df = derive_vel(df, args.fps)

    midx = (args.goal_left + args.goal_right) / 2.0

    preset = PRESETS[args.action]
    lead = float(preset["lead"])
    trail = float(preset["trail"])

    t_throw = detect_throw_in(df, args.iw)
    t_shot = detect_shot(df, goal_x=midx, goal_y=args.ih / 2.0)
    if t_shot is None:
        t_shot = float(df.t.min() + 13.0)

    clip_start = float(df.t.min())
    clip_end = float(df.t.max())

    t_phase_begin = max(clip_start, t_shot - lead)
    t_phase_end = min(clip_end, t_shot + trail)
    t_cele_end = detect_cele_end(df, t_phase_end)

    throw_preroll = float(np.clip(lead, 0.5, 1.0))
    if t_throw is not None:
        t_throw = max(clip_start, t_throw - throw_preroll)

    band_start = min(t_phase_begin, t_shot)
    band_end = max(t_cele_end, t_phase_end)
    y_min, y_max = estimate_vertical_band(df, args.ih, band_start, band_end)

    shot_window_start = max(clip_start, t_shot - 0.6)
    shot_window_end = min(clip_end, t_shot + 0.6)
    shot_center_y = median_ball_y(df, shot_window_start, shot_window_end)
    if shot_center_y is None:
        shot_center_y = (y_min + y_max) / 2.0

    cele_window_start = t_phase_end
    cele_window_end = min(clip_end, t_cele_end + 0.6)
    cele_center_y = median_ball_y(df, cele_window_start, cele_window_end)
    if cele_center_y is None:
        cele_center_y = shot_center_y

    fade_duration = 0.35

    w_forced = int(preset["w_forced"])
    z_shot = float(preset["z_shot"])
    z_cele = float(preset["z_cele"])

    recipe = {
        "throw_in_s": t_throw,
        "shot_s": t_shot,
        "phase_beg_s": t_phase_begin,
        "phase_end_s": t_phase_end,
        "cele_end_s": t_cele_end,
        "midx": int(round(midx)),
        "w_forced": w_forced,
        "z_shot": z_shot,
        "z_cele": z_cele,
        "lead_s": lead,
        "trail_s": trail,
        "y_min": y_min,
        "y_max": y_max,
        "shot_center_y": shot_center_y,
        "cele_center_y": cele_center_y,
        "fade_s": fade_duration,
        "action": args.action,
    }

    Path(args.out_recipe).write_text(json.dumps(recipe, indent=2))

    ffmpeg_filter = build_ffmpeg_phase_filter(
        image_width=args.iw,
        image_height=args.ih,
        clip_start=clip_start,
        clip_end=clip_end,
        t_phase_begin=t_phase_begin,
        t_phase_end=t_phase_end,
        midx=midx,
        w_forced=recipe["w_forced"],
        z_shot=recipe["z_shot"],
        z_cele=recipe["z_cele"],
        shot_center_y=shot_center_y,
        cele_center_y=cele_center_y,
        y_min=y_min,
        y_max=y_max,
        fade_duration=fade_duration,
    )
    Path(args.out_ffmpeg).write_text(ffmpeg_filter)


if __name__ == "__main__":
    main()
