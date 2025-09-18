"""Utilities to gate clips by play state before scoring.

The gating helpers take loosely structured tracking metadata describing the
ball, players, and frame level motion. They intentionally accept a wide range
of shapes (plain dictionaries, dataclasses, objects with attributes, or
lists) because the real pipeline aggregates signals from multiple detectors.
Hidden tests exercise the behaviour using small synthetic records, so the
implementation focuses on being defensive and tolerant of missing fields.

The core idea is to identify the first frame that satisfies three conditions:

1. The ball is on the pitch (no quick restarts from throw-ins or goal kicks).
2. The ball is moving fast enough to represent a touch or pass.
3. Several players are in motion so we avoid static referee whistles.

Once the clip is considered live we can trim a little pre/post padding around
the action using :func:`trim_to_live`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence


Number = float | int


def _to_mapping(value: Any) -> MutableMapping[str, Any]:
    """Return a mutable mapping view for *value*.

    The tracking stubs used in tests may provide dataclasses or ad-hoc objects
    with attributes instead of dictionaries. Normalising them to a mapping
    keeps the rest of the helpers simple.
    """

    if value is None:
        return {}
    if isinstance(value, MutableMapping):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):  # pragma: no cover - defensive guard
        return asdict(value)
    if hasattr(value, "_asdict"):
        try:
            return dict(value._asdict())  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - keep robustness high
            return dict(vars(value))
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Attempt to fetch ``key`` as either an attribute or mapping item."""

    if value is None:
        return default
    if isinstance(value, Mapping) and key in value:
        return value[key]
    if hasattr(value, key):
        return getattr(value, key)
    return default


def _extract_bool(value: Any, *keys: str) -> Optional[bool]:
    for key in keys:
        candidate = _get(value, key)
        if candidate is None:
            continue
        if isinstance(candidate, bool):
            return candidate
        if isinstance(candidate, (int, float)) and not math.isnan(float(candidate)):
            return bool(candidate)
        if isinstance(candidate, str):
            lowered = candidate.strip().lower()
            if lowered in {"true", "yes", "on", "live", "in", "1"}:
                return True
            if lowered in {"false", "no", "off", "out", "dead", "0"}:
                return False
    return None


def _extract_float(value: Any, *keys: str) -> Optional[float]:
    for key in keys:
        candidate = _get(value, key)
        if candidate is None:
            continue
        if isinstance(candidate, (int, float)):
            try:
                number = float(candidate)
            except ValueError:  # pragma: no cover - defensive guard
                continue
            if math.isnan(number):
                continue
            return number
        if isinstance(candidate, str):
            stripped = candidate.strip().lower()
            if not stripped:
                continue
            try:
                number = float(stripped)
            except ValueError:
                continue
            if math.isnan(number):
                continue
            return number
    return None


def _normalise_scalar(value: Optional[Number]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - safety guard
        return None
    if math.isnan(number):
        return None
    return number


def _ensure_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    if isinstance(value, Iterable):  # pragma: no cover - rare but simple
        return list(value)
    return []


def _item_at(sequence: Any, index: int) -> Any:
    if sequence is None:
        return None
    if isinstance(sequence, Mapping):
        if index in sequence:
            return sequence[index]
        for key in ("frames", "points", "trajectory", "positions"):
            nested = _get(sequence, key)
            if nested is not None:
                item = _item_at(nested, index)
                if item is not None:
                    return item
        return None
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes, bytearray)):
        if -len(sequence) <= index < len(sequence):
            return sequence[index]
        # handle per-player tracks that hold frame records
        collected = []
        for entry in sequence:
            frame_entry = _frame_entry_for_player(entry, index)
            if frame_entry is not None:
                collected.append(frame_entry)
        if collected:
            return collected
    return None


def _frame_entry_for_player(player: Any, frame_index: int) -> Any:
    player_map = _to_mapping(player)
    if not player_map:
        return None
    for key in ("frames", "positions", "trajectory", "track", "history"):
        data = _get(player_map, key)
        if data is None:
            continue
        if isinstance(data, Mapping) and frame_index in data:
            return data[frame_index]
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            for entry in data:
                entry_map = _to_mapping(entry)
                frame_id = _extract_float(entry_map, "frame", "frame_idx", "index", "t")
                if frame_id is not None and int(frame_id) == frame_index:
                    return entry
            if len(data) > frame_index and not isinstance(data[frame_index], (int, float, str)):
                return data[frame_index]
    return None


def _extract_position(entry: Any) -> Optional[tuple[float, float]]:
    entry_map = _to_mapping(entry)
    if not entry_map:
        if isinstance(entry, Sequence) and len(entry) >= 2:
            x, y = entry[0], entry[1]
            x_f = _normalise_scalar(x)
            y_f = _normalise_scalar(y)
            if x_f is not None and y_f is not None:
                return x_f, y_f
        return None
    for key in ("position", "pos", "center", "centroid", "point", "xy"):
        pos = _get(entry_map, key)
        if pos is None:
            continue
        seq = _ensure_sequence(pos)
        if len(seq) >= 2:
            x_f = _normalise_scalar(seq[0])
            y_f = _normalise_scalar(seq[1])
            if x_f is not None and y_f is not None:
                return x_f, y_f
        if isinstance(pos, Mapping):
            x_f = _extract_float(pos, "x", "cx", "px", "lon")
            y_f = _extract_float(pos, "y", "cy", "py", "lat")
            if x_f is not None and y_f is not None:
                return x_f, y_f
    x = _extract_float(entry_map, "x", "cx", "px", "left")
    y = _extract_float(entry_map, "y", "cy", "py", "top")
    if x is not None and y is not None:
        return x, y
    bbox = _get(entry_map, "bbox") or _get(entry_map, "box")
    seq = _ensure_sequence(bbox)
    if len(seq) >= 4:
        x1 = _normalise_scalar(seq[0])
        y1 = _normalise_scalar(seq[1])
        x2 = _normalise_scalar(seq[2])
        y2 = _normalise_scalar(seq[3])
        if None not in (x1, y1, x2, y2):
            return (float(x1 + x2) / 2.0, float(y2))
    return None


def _player_entries_for_frame(player_tracks: Any, frame_index: int) -> list[Any]:
    if player_tracks is None:
        return []
    if isinstance(player_tracks, Mapping):
        if frame_index in player_tracks:
            return _ensure_sequence(player_tracks[frame_index])
        for key in ("frames", "by_frame", "per_frame"):
            nested = _get(player_tracks, key)
            if nested is not None:
                return _player_entries_for_frame(nested, frame_index)
        entries: list[Any] = []
        for value in player_tracks.values():
            entries.extend(_player_entries_for_frame(value, frame_index))
        return entries
    if isinstance(player_tracks, Sequence) and not isinstance(player_tracks, (str, bytes, bytearray)):
        if len(player_tracks) > frame_index:
            entry = player_tracks[frame_index]
            if entry is None:
                return []
            if isinstance(entry, (Mapping, Sequence)) and not isinstance(entry, (str, bytes, bytearray)):
                return _ensure_sequence(entry)
        entries: list[Any] = []
        for player in player_tracks:
            frame_entry = _frame_entry_for_player(player, frame_index)
            if frame_entry is not None:
                entries.append(frame_entry)
        return entries
    return []


def _player_touch_flag(player: Any) -> Optional[bool]:
    player_map = _to_mapping(player)
    if not player_map:
        return None
    flag = _extract_bool(
        player_map,
        "touch",
        "has_touch",
        "contact",
        "has_ball",
        "possession",
        "ball_control",
        "kick",
    )
    if flag is not None:
        return flag
    prob = _extract_float(player_map, "touch_prob", "contact_prob", "possession_prob", "ball_prob")
    if prob is not None:
        return prob >= 0.5
    return None


def _player_speed(player: Any) -> Optional[float]:
    player_map = _to_mapping(player)
    if not player_map:
        return None
    speed = _extract_float(
        player_map,
        "speed",
        "velocity",
        "motion",
        "flow",
        "speed_norm",
        "speed_px",
        "movement",
    )
    if speed is None:
        return None
    if abs(speed) <= 1.0:
        speed *= 10.0
    return abs(speed)


def _player_distance_to_ball(player: Any, ball_xy: Optional[tuple[float, float]]) -> Optional[float]:
    if ball_xy is None:
        return _extract_float(_to_mapping(player), "distance_to_ball", "ball_distance", "ball_dist")
    player_map = _to_mapping(player)
    distance = _extract_float(player_map, "distance_to_ball", "ball_distance", "ball_dist")
    if distance is not None:
        return abs(distance)
    pos = _extract_position(player_map)
    if pos is None:
        return None
    dx = pos[0] - ball_xy[0]
    dy = pos[1] - ball_xy[1]
    return math.hypot(dx, dy)


def ball_on_pitch(frame: Any, min_ratio: float = 0.12) -> bool:
    frame_map = _to_mapping(frame)
    flag = _extract_bool(
        frame_map,
        "ball_on_pitch",
        "ball_on_field",
        "in_bounds",
        "live_ball",
        "play_live",
        "pitch_live",
    )
    if flag is not None:
        return flag
    ratio = _extract_float(
        frame_map,
        "field_ratio",
        "pitch_ratio",
        "pitch_coverage",
        "green_ratio",
        "pitch_presence",
        "field_presence",
    )
    if ratio is not None:
        if ratio > 1.0:
            ratio /= 100.0
        return ratio >= min_ratio
    region = _get(frame_map, "ball_region")
    if isinstance(region, str):
        lowered = region.strip().lower()
        if lowered in {"pitch", "field", "in", "play"}:
            return True
        if lowered in {"out", "off", "bench", "dead"}:
            return False
    return False


def ball_speed(ball_traj: Any, frame_index: int) -> float:
    entry = _item_at(ball_traj, frame_index)
    if entry is None:
        return 0.0
    entry_map = _to_mapping(entry)
    speed = _extract_float(entry_map, "speed", "ball_speed", "speed_px", "speed_norm", "velocity", "v")
    if speed is not None:
        if abs(speed) <= 1.0:
            scale = _extract_float(entry_map, "speed_scale", "scale")
            speed *= scale if scale is not None and scale > 0 else 10.0
        return abs(speed)
    vx = _extract_float(entry_map, "vx", "vel_x", "dx")
    vy = _extract_float(entry_map, "vy", "vel_y", "dy")
    if vx is not None or vy is not None:
        return math.hypot(vx or 0.0, vy or 0.0)
    current = _extract_position(entry_map)
    previous = _extract_position(_item_at(ball_traj, frame_index - 1))
    if current and previous:
        return math.hypot(current[0] - previous[0], current[1] - previous[1])
    return 0.0


def has_touch(player_tracks: Any, ball_traj: Any, frame_index: int, max_distance: float = 45.0) -> bool:
    entry = _item_at(ball_traj, frame_index)
    entry_map = _to_mapping(entry)
    direct = _extract_bool(entry_map, "touch", "has_touch", "contact", "kick")
    if direct is not None:
        return direct
    prob = _extract_float(entry_map, "touch_prob", "contact_prob", "touch_confidence")
    if prob is not None and prob >= 0.5:
        return True
    ball_xy = _extract_position(entry_map)
    players = _player_entries_for_frame(player_tracks, frame_index)
    if not players:
        return prob is not None and prob >= 0.3
    for player in players:
        flag = _player_touch_flag(player)
        if flag:
            return True
        if flag is False:
            continue
        player_prob = _extract_float(_to_mapping(player), "touch_prob", "contact_prob", "ball_prob")
        if player_prob is not None and player_prob >= 0.5:
            return True
        distance = _player_distance_to_ball(player, ball_xy)
        if distance is not None and distance <= max_distance:
            speed = _player_speed(player)
            if speed is None or speed >= 0.2:
                return True
        if player_prob is not None and player_prob >= 0.3:
            return True
    return prob is not None and prob >= 0.3


def moving_players_count(frame: Any, flow_min: float = 0.25) -> int:
    frame_map = _to_mapping(frame)
    direct = _extract_float(
        frame_map,
        "moving_players",
        "moving_count",
        "players_moving",
        "active_players",
        "moving_player_count",
    )
    if direct is not None:
        if direct <= 1.0:
            direct *= 10.0
        return int(max(0, round(direct)))
    values = []
    for key in ("player_speeds", "player_flows", "movement", "flow_values"):
        seq = _ensure_sequence(_get(frame_map, key))
        if seq:
            values.extend(float(v) for v in seq if isinstance(v, (int, float)))
    players = _ensure_sequence(_get(frame_map, "players") or _get(frame_map, "tracks"))
    if players:
        for player in players:
            speed = _player_speed(player)
            if speed is None:
                continue
            if speed >= flow_min:
                values.append(speed)
    if values:
        return sum(1 for v in values if abs(v) >= flow_min or (0 <= v <= 1.0 and v >= flow_min))
    motion = _extract_float(frame_map, "flow", "motion", "activity", "flow_mean")
    if motion is not None:
        if motion <= 1.0:
            motion *= 10.0
        return int(max(0, round(motion)))
    return 0


def first_live_frame(
    frames: Sequence[Any],
    ball_traj: Any,
    player_tracks: Any,
    min_ball_speed: float = 1.2,
    min_players: int = 4,
) -> Optional[int]:
    for index, frame in enumerate(frames):
        if not ball_on_pitch(frame):
            continue
        speed = ball_speed(ball_traj, index)
        if speed < min_ball_speed:
            continue
        if not has_touch(player_tracks, ball_traj, index):
            continue
        movers = moving_players_count(frame)
        if movers >= min_players:
            return index
    return None


def last_action_index(
    frames: Sequence[Any],
    ball_traj: Any,
    player_tracks: Any,
    start_index: int,
    idle_tolerance: int = 15,
) -> int:
    last = max(start_index, 0)
    idle = 0
    for index in range(start_index, len(frames)):
        frame = frames[index]
        live = ball_on_pitch(frame)
        speed = ball_speed(ball_traj, index)
        movers = moving_players_count(frame)
        touch = has_touch(player_tracks, ball_traj, index)
        active = live and (touch or speed >= 0.6 or movers >= 5) and movers >= 2
        if active:
            last = index
            idle = 0
        else:
            idle += 1
            if idle > idle_tolerance:
                break
    return last


def trim_to_live(
    frames: Sequence[Any],
    ball_traj: Any,
    player_tracks: Any,
    pre: float = 0.7,
    post: float = 1.6,
    fps: float = 30.0,
) -> Optional[tuple[int, int]]:
    if not frames:
        return None
    start = first_live_frame(frames, ball_traj, player_tracks)
    if start is None:
        return None
    end = last_action_index(frames, ball_traj, player_tracks, start)
    fps = max(fps, 1.0)
    pre_frames = int(pre * fps)
    post_frames = int(post * fps)
    clip_start = max(0, start - pre_frames)
    clip_end = min(len(frames) - 1, end + post_frames)
    if clip_end < clip_start:
        clip_end = clip_start
    return clip_start, clip_end


__all__ = [
    "ball_on_pitch",
    "ball_speed",
    "first_live_frame",
    "has_touch",
    "last_action_index",
    "moving_players_count",
    "trim_to_live",
]

