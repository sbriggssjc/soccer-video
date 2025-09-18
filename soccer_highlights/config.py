"""Configuration models and loader for the soccer highlights suite."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ._pydantic_compat import BaseModel, Field, validator
import yaml


class PathsConfig(BaseModel):
    video: Path = Field(Path("full_game_stabilized.mp4"), description="Primary input video path.")
    output_dir: Path = Field(Path("out"), description="Base directory for derived outputs.")
    work_dir: Optional[Path] = Field(None, description="Optional cache/work directory.")


class DetectConfig(BaseModel):
    min_gap: float = Field(2.0, description="Minimum gap between merged segments in seconds.")
    pre: float = Field(1.0, description="Seconds of pre-roll when expanding detections.")
    post: float = Field(2.0, description="Seconds of post-roll when expanding detections.")
    max_count: int = Field(40, description="Maximum number of highlight windows to keep.")
    audio_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight of audio score in blended detection score.")
    threshold_std: float = Field(0.5, description="Multiplier for std-dev when computing adaptive threshold.")
    hysteresis: float = Field(0.3, description="Fraction of threshold used as low hysteresis.")
    sustain: float = Field(1.0, description="Seconds a detection must sustain to be accepted.")
    merge_hysteresis: float = Field(0.75, description="Merge overlapping windows when overlap exceeds this fraction.")
    exclude_events: list[str] = Field(
        default_factory=lambda: ["restart", "setup"],
        description="Event categories that should be dropped before merging windows.",
    )


class ShrinkConfig(BaseModel):
    mode: str = Field("smart", description="Shrink mode: 'simple' or 'smart'.")
    pre: float = Field(3.0)
    post: float = Field(5.0)
    aspect: str = Field("horizontal", description="Crop aspect for tracked clips.")
    zoom: float = Field(1.2, description="Zoom factor when aspect is horizontal.")
    bias_blue: bool = Field(False, description="Bias motion scoring toward blue jerseys.")
    write_clips: Optional[Path] = Field(None, description="When set, tracked clips are written here.")

    @validator("mode")
    def validate_mode(cls, value: str) -> str:
        if value not in {"simple", "smart"}:
            raise ValueError("mode must be 'simple' or 'smart'")
        return value

    @validator("aspect")
    def validate_aspect(cls, value: str) -> str:
        if value not in {"horizontal", "vertical"}:
            raise ValueError("aspect must be 'horizontal' or 'vertical'")
        return value


class ClipConfig(BaseModel):
    min_duration: float = Field(3.0, description="Skip clips shorter than this.")
    preset: str = Field("veryfast")
    crf: int = Field(20)
    audio_bitrate: str = Field("160k")
    workers: int = Field(2, description="Number of parallel workers for clip export.")
    overwrite: bool = Field(False, description="Overwrite existing clips when true.")


class RankConfig(BaseModel):
    k: int = Field(10, description="Number of clips to keep for Top-K.")
    max_len: float = Field(18.0, description="Maximum duration per clip in seconds.")
    sustain: float = Field(1.25, description="Seconds of sustained activity for inpoint selection.")
    min_tail: float = Field(6.0, description="Minimum seconds remaining after inpoint.")


class ReelProfile(BaseModel):
    name: str
    width: int
    height: int
    fps: float = Field(30.0)
    crossfade_frames: int = Field(6)
    audio_duck_db: float = Field(6.0, description="Audio ducking applied during transitions.")
    title_duration: float = Field(1.0, description="Seconds for intro title slate.")
    label_position: str = Field("0.05*W:0.1*H", description="drawtext position expression.")


class ReelConfig(BaseModel):
    profile: str = Field("broadcast")
    topk_title: str = Field("Top Plays")
    full_title: str = Field("Full Highlights")


class HSVRange(BaseModel):
    h: float
    s: float
    v: float


class ColorsConfig(BaseModel):
    pitch_hsv: HSVRange = Field(HSVRange(h=90, s=80, v=80))
    team_primary: HSVRange = Field(HSVRange(h=210, s=70, v=70))
    team_secondary: HSVRange = Field(HSVRange(h=30, s=70, v=70))
    calibrate: bool = Field(False)


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    shrink: ShrinkConfig = Field(default_factory=ShrinkConfig)
    clips: ClipConfig = Field(default_factory=ClipConfig)
    rank: RankConfig = Field(default_factory=RankConfig)
    reels: ReelConfig = Field(default_factory=ReelConfig)
    colors: ColorsConfig = Field(default_factory=ColorsConfig)
    profiles: Dict[str, ReelProfile] = Field(
        default_factory=lambda: {
            "broadcast": ReelProfile(name="broadcast", width=1920, height=1080, fps=30.0, crossfade_frames=6),
            "social-vertical": ReelProfile(
                name="social-vertical",
                width=1080,
                height=1920,
                fps=30.0,
                crossfade_frames=8,
                audio_duck_db=8.0,
                label_position="0.08*W:0.12*H",
            ),
            "coach-review": ReelProfile(
                name="coach-review",
                width=1280,
                height=720,
                fps=30.0,
                crossfade_frames=4,
                audio_duck_db=4.0,
                label_position="0.05*W:0.05*H",
            ),
        }
    )

    def resolve_profile(self, name: Optional[str] = None) -> ReelProfile:
        target = name or self.reels.profile
        if target not in self.profiles:
            raise KeyError(f"Unknown reel profile '{target}'. Available: {sorted(self.profiles)}")
        return self.profiles[target]

    @property
    def output_dir(self) -> Path:
        return self.paths.output_dir


def load_config(path: Path | str) -> AppConfig:
    """Load configuration from YAML, falling back to defaults when missing."""

    cfg_path = Path(path)
    if not cfg_path.exists():
        return AppConfig()
    data = yaml.safe_load(cfg_path.read_text()) or {}
    return AppConfig.parse_obj(data)
