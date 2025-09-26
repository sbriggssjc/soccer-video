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
    t_phase_begin: float,
    t_phase_end: float,
    midx: float,
    w_forced: int = 700,
    z_shot: float = 1.12,
    z_cele: float = 1.22,
) -> str:
    """Construct an ffmpeg filter graph for the action phases."""
    return (
        "[0:v]split=3[v0][v1][v2];[0:a]asplit=3[a0][a1][a2];"
        "[v0]trim=start=0:end={t_phase_begin},setpts=PTS-STARTPTS,"
        "scale=w=-2:h=1080:flags=lanczos,setsar=1,"
        "crop=w={w_forced}:h=1080:x='(iw-{w_forced})/2':y='(ih-1080)/2',"
        "format=yuv420p[v0o];"
        "[a0]atrim=start=0:end={t_phase_begin},asetpts=PTS-STARTPTS[a0o];"
        "[v1]trim=start={t_phase_begin}:end={t_phase_end},setpts=PTS-STARTPTS,"
        "scale=w=-2:h=1080:flags=lanczos,setsar=1,"
        "crop=w='max(16, floor(((ih*9/16)/{z_shot})/2)*2)':"
        "h='max(16, floor((ih/{z_shot})/2)*2)':"
        "x='({midx}) - (max(16, floor(((ih*9/16)/{z_shot})/2)*2))/2':"
        "y='(ih/2) - (max(16, floor((ih/{z_shot})/2)*2))/2',"
        "format=yuv420p[v1o];"
        "[a1]atrim=start={t_phase_begin}:end={t_phase_end},asetpts=PTS-STARTPTS[a1o];"
        "[v2]trim=start={t_phase_end},setpts=PTS-STARTPTS,"
        "scale=w=-2:h=1080:flags=lanczos,setsar=1,"
        "crop=w='max(16, floor(((ih*9/16)/{z_cele})/2)*2)':"
        "h='max(16, floor((ih/{z_cele})/2)*2)':"
        "x='({midx}) - (max(16, floor(((ih*9/16)/{z_cele})/2)*2))/2':"
        "y='(ih/2) - (max(16, floor((ih/{z_cele})/2)*2))/2',"
        "format=yuv420p[v2o];"
        "[a2]atrim=start={t_phase_end},asetpts=PTS-STARTPTS[a2o];"
        "[v0o][a0o][v1o][a1o][v2o][a2o]concat=n=3:v=1:a=1[v][a]"
    ).format(
        t_phase_begin=t_phase_begin,
        t_phase_end=t_phase_end,
        w_forced=w_forced,
        midx=midx,
        z_shot=z_shot,
        z_cele=z_cele,
    )


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
    parser.add_argument("--out_recipe", required=True)
    parser.add_argument("--out_ffmpeg", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = derive_vel(df, args.fps)

    midx = (args.goal_left + args.goal_right) / 2.0

    t_throw = detect_throw_in(df, args.iw)
    t_shot = detect_shot(df, goal_x=midx, goal_y=args.ih / 2.0)
    if t_shot is None:
        t_shot = float(df.t.min() + 13.0)

    t_phase_begin = max(float(df.t.min()), t_shot - 0.7)
    t_phase_end = min(float(df.t.max()), t_shot + 1.1)
    t_cele_end = detect_cele_end(df, t_phase_end)

    recipe = {
        "throw_in_s": t_throw,
        "shot_s": t_shot,
        "phase_beg_s": t_phase_begin,
        "phase_end_s": t_phase_end,
        "cele_end_s": t_cele_end,
        "midx": int(round(midx)),
        "w_forced": 700,
        "z_shot": 1.12,
        "z_cele": 1.22,
    }

    Path(args.out_recipe).write_text(json.dumps(recipe, indent=2))

    ffmpeg_filter = build_ffmpeg_phase_filter(
        image_width=args.iw,
        image_height=args.ih,
        t_phase_begin=t_phase_begin,
        t_phase_end=t_phase_end,
        midx=midx,
        w_forced=recipe["w_forced"],
        z_shot=recipe["z_shot"],
        z_cele=recipe["z_cele"],
    )
    Path(args.out_ffmpeg).write_text(ffmpeg_filter)


if __name__ == "__main__":
    main()
