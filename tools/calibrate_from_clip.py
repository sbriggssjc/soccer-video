"""Preset calibration helper for the unified renderer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import cv2  # type: ignore
import numpy as np
import yaml

from render_follow_unified import (
    CameraPlanner,
    PRESETS_PATH,
    ensure_presets_file,
    ffprobe_fps,
    find_label_files,
    interp_labels_to_fps,
    load_labels,
    load_presets,
    parse_portrait,
)


def _video_metadata(path: Path) -> Dict[str, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return {"width": width, "height": height, "frame_count": frame_count}


def _cost(states, positions: np.ndarray, used_mask: np.ndarray) -> float:
    if not states:
        return float("inf")

    vertical_errors: List[float] = []
    safety_penalties: List[float] = []

    for state, pos, used in zip(states, positions, used_mask):
        if not used or np.isnan(pos).any():
            continue
        crop_left = state.cx - state.crop_w / 2.0
        crop_top = state.cy - state.crop_h / 2.0
        rel_x = (pos[0] - crop_left) / max(state.crop_w, 1e-6)
        rel_y = (pos[1] - crop_top) / max(state.crop_h, 1e-6)
        vertical_errors.append((rel_y - 0.4) ** 2)
        edge_distance = min(rel_x, 1.0 - rel_x, rel_y, 1.0 - rel_y)
        safety_penalties.append(max(0.0, 0.12 - edge_distance) ** 2)

    if not vertical_errors:
        return float("inf")

    cx_values = np.array([s.cx for s in states], dtype=np.float32)
    zoom_values = np.array([s.zoom for s in states], dtype=np.float32)
    if len(cx_values) >= 3:
        lateral_acc = np.diff(cx_values, n=2)
    else:
        lateral_acc = np.array([0.0], dtype=np.float32)
    if len(zoom_values) >= 3:
        zoom_acc = np.diff(zoom_values, n=2)
    else:
        zoom_acc = np.array([0.0], dtype=np.float32)

    total = (
        float(np.mean(vertical_errors)) * 4.0
        + float(np.mean(np.square(lateral_acc))) * 0.6
        + float(np.mean(np.square(zoom_acc))) * 0.4
        + (float(np.mean(safety_penalties)) if safety_penalties else 0.0) * 2.5
    )
    return total


def _evaluate(config: Dict[str, float], width: int, height: int, fps: float, portrait, positions, used_mask) -> float:
    planner = CameraPlanner(
        width=width,
        height=height,
        fps=fps,
        lookahead=int(config["lookahead"]),
        smoothing=float(config["smoothing"]),
        pad=float(config["pad"]),
        speed_limit=float(config["speed_limit"]),
        zoom_min=float(config.get("zoom_min", 1.0)),
        zoom_max=float(config["zoom_max"]),
        portrait=portrait,
    )
    states = planner.plan(positions, used_mask)
    return _cost(states, positions, used_mask)


def _save_preset(preset_name: str, values: Dict[str, float]) -> None:
    ensure_presets_file()
    presets = load_presets()
    presets.setdefault(preset_name, {}).update(values)
    with PRESETS_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(presets, handle, sort_keys=True)


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")

    presets = load_presets()
    preset_name = args.preset.lower()
    if preset_name not in presets:
        raise ValueError(f"Preset '{preset_name}' not present in {PRESETS_PATH}")

    preset_config = presets[preset_name]
    portrait = parse_portrait(preset_config.get("portrait")) if preset_config.get("portrait") else None

    fps_in = ffprobe_fps(input_path)
    fps = float(preset_config.get("fps", fps_in))
    if fps <= 0:
        fps = fps_in

    meta = _video_metadata(input_path)

    labels_root = Path(args.labels_root or "out/yolo").expanduser()
    label_files = find_label_files(input_path.stem, labels_root)
    labels = load_labels(label_files, meta["width"], meta["height"])
    positions, used_mask = interp_labels_to_fps(labels, meta["frame_count"], fps_in, fps)

    base_config = {
        "lookahead": float(preset_config.get("lookahead", 18)),
        "smoothing": float(preset_config.get("smoothing", 0.65)),
        "pad": float(preset_config.get("pad", 0.22)),
        "speed_limit": float(preset_config.get("speed_limit", 480)),
        "zoom_max": float(preset_config.get("zoom_max", 2.2)),
        "zoom_min": float(preset_config.get("zoom_min", 1.0)),
    }

    best_config = dict(base_config)
    best_cost = _evaluate(best_config, meta["width"], meta["height"], fps, portrait, positions, used_mask)

    rng = random.Random(args.seed)
    for _ in range(args.iters):
        candidate = dict(base_config)
        candidate["lookahead"] = int(np.clip(rng.gauss(base_config["lookahead"], 3.0), 4, 36))
        candidate["smoothing"] = float(np.clip(rng.gauss(base_config["smoothing"], 0.05), 0.4, 0.85))
        candidate["pad"] = float(np.clip(rng.gauss(base_config["pad"], 0.02), 0.12, 0.30))
        candidate["speed_limit"] = float(np.clip(rng.gauss(base_config["speed_limit"], 60.0), 240, 680))
        candidate["zoom_max"] = float(np.clip(rng.gauss(base_config["zoom_max"], 0.1), 1.6, 2.6))
        cost_value = _evaluate(candidate, meta["width"], meta["height"], fps, portrait, positions, used_mask)
        if cost_value < best_cost:
            best_cost = cost_value
            best_config.update(candidate)

    if best_config != base_config:
        _save_preset(preset_name, best_config)

    log_dir = Path("out/render_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{input_path.stem}_calib.json"
    payload = {
        "preset": preset_name,
        "input": str(input_path),
        "iterations": args.iters,
        "base": base_config,
        "best": best_config,
        "cost": best_cost,
    }
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Updated preset '{preset_name}' with cost {best_cost:.4f}. Settings written to {PRESETS_PATH}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate unified renderer presets")
    parser.add_argument("--in", dest="input", required=True, help="Tester clip MP4 path")
    parser.add_argument("--labels-root", dest="labels_root", help="Root folder with YOLO labels")
    parser.add_argument("--preset", dest="preset", default="cinematic", help="Preset to adjust")
    parser.add_argument("--iters", dest="iters", type=int, default=200, help="Random search iterations")
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="Deterministic seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
