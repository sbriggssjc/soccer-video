"""Signal feature extraction for Smart Soccer Highlight Selector."""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Optional dependency used only when available.
    import cv2
except Exception as exc:  # pragma: no cover - handled gracefully at runtime
    cv2 = None
    logging.getLogger(__name__).warning("OpenCV is unavailable: %s", exc)

try:  # librosa loads audio straight from video containers via audioread.
    import librosa
except Exception as exc:  # pragma: no cover - handled gracefully
    librosa = None
    logging.getLogger(__name__).warning("librosa is unavailable: %s", exc)


@dataclasses.dataclass
class AudioFeatureConfig:
    sr: int = 11_025
    hop_length: int = 512
    frame_length: int = 2_048
    smooth_window_seconds: float = 0.75
    whistle_band: Tuple[float, float] = (3_500.0, 4_500.0)


@dataclasses.dataclass
class MotionFeatureConfig:
    sample_stride: int = 3
    roi: Tuple[float, float, float, float] = (0.2, 0.8, 0.25, 0.75)  # xmin,xmax,ymin,ymax in normalized coords
    smooth_window_seconds: float = 1.2


def _rolling(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series
    pad = window // 2
    padded = np.pad(series, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _time_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if "time" in df:
        return df["time"]
    if "t" in df:
        return df["t"]
    return None


def compute_audio_features(video_path: Path, config: AudioFeatureConfig) -> pd.DataFrame:
    """
    Robust audio feature extraction that guarantees equal-length columns.
    Returns a DataFrame with columns: ['t', 'rms', 'onset', 'flux', 'hf', 'zcr'].
    """

    if librosa is None:
        logging.warning("Audio features unavailable because librosa could not be imported.")
        return pd.DataFrame(columns=["t", "rms", "onset", "flux", "hf", "zcr"])

    try:
        # Load mono audio (librosa can read audio from mp4 via audioread/ffmpeg)
        signal, sr = librosa.load(str(video_path), sr=config.sr, mono=True)
    except Exception as e:
        logging.warning("Audio load failed for %s: %s", video_path, e)
        return pd.DataFrame(columns=["t", "rms", "onset", "flux", "hf", "zcr"])

    if signal is None or len(signal) == 0:
        logging.warning("No audio samples in %s", video_path)
        return pd.DataFrame(columns=["t", "rms", "onset", "flux", "hf", "zcr"])

    # Feature frames
    frame_length = config.frame_length
    hop_length = config.hop_length

    # Core features (each returns shape (n_frames,) or (n_freq, n_frames))
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    onset = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
    # Spectral flux (difference of power spectra)
    S = np.abs(librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)) ** 2
    flux = np.sqrt(np.sum(np.diff(S, axis=1, prepend=S[:, :1]) ** 2, axis=0))
    # High-frequency energy ratio (above ~2 kHz)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    hf_mask = freqs >= 2000.0
    total_energy = np.sum(S, axis=0) + 1e-9
    hf_energy = np.sum(S[hf_mask, :], axis=0)
    hf_ratio = hf_energy / total_energy
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=hop_length, center=True)[0]

    # Make all vectors the SAME length
    lengths = [len(rms), len(onset), len(flux), len(hf_ratio), len(zcr)]
    n_frames = int(min(lengths))
    rms = rms[:n_frames]
    onset = onset[:n_frames]
    flux = flux[:n_frames]
    hf_ratio = hf_ratio[:n_frames]
    zcr = zcr[:n_frames]

    # Time vector from frame indices (avoids float drift)
    frames = np.arange(n_frames)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    df = pd.DataFrame(
        {
            "t": times,
            "rms": rms,
            "onset": onset,
            "flux": flux,
            "hf": hf_ratio,
            "zcr": zcr,
        }
    )
    return df


def compute_motion_features(video_path: Path, config: Optional[MotionFeatureConfig] = None) -> pd.DataFrame:
    """Compute coarse motion descriptors from the video frames."""

    config = config or MotionFeatureConfig()
    if cv2 is None:
        logging.warning("Motion features unavailable because OpenCV could not be imported.")
        return pd.DataFrame(columns=["time", "motion", "motion_smooth", "pan_score"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning("Failed to open video for motion features: %s", video_path)
        return pd.DataFrame(columns=["time", "motion", "motion_smooth", "pan_score"])

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps):
        fps = 25.0

    stride = max(1, config.sample_stride)
    roi = config.roi
    records: List[dict] = []
    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        x0 = int(w * roi[0])
        x1 = int(w * roi[1])
        y0 = int(h * roi[2])
        y1 = int(h * roi[3])
        roi_slice = gray[y0:y1, x0:x1]

        if prev_gray is None:
            motion_mag = 0.0
            pan_score = 0.0
        else:
            diff = cv2.absdiff(roi_slice, prev_gray[y0:y1, x0:x1])
            motion_mag = float(diff.mean())
            global_diff = cv2.absdiff(gray, prev_gray)
            pan_score = float(np.percentile(global_diff, 95))
        records.append(
            {
                "time": frame_idx / fps,
                "motion": motion_mag,
                "pan_score": pan_score,
            }
        )
        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not records:
        return pd.DataFrame(columns=["time", "motion", "motion_smooth", "pan_score"])

    df = pd.DataFrame.from_records(records)
    window = max(1, int(config.smooth_window_seconds * fps / stride))
    df["motion_smooth"] = _rolling(df["motion"].to_numpy(), window)
    df["pan_score_smooth"] = _rolling(df["pan_score"].to_numpy(), window)
    df.rename(columns={"pan_score_smooth": "pan_score"}, inplace=True)
    return df[["time", "motion", "motion_smooth", "pan_score"]]


def interpolate_to(times: np.ndarray, samples_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate ``values`` sampled at ``samples_time`` to ``times``."""

    if samples_time.size == 0 or values.size == 0:
        return np.zeros_like(times)
    return np.interp(times, samples_time, values, left=float(values[0]), right=float(values[-1]))


def derive_in_play_mask(
    audio_df: pd.DataFrame,
    motion_df: pd.DataFrame,
    whistle_weight: float = 0.6,
    min_run: float = 2.0,
) -> pd.DataFrame:
    """Infer in-play spans from audio/motion features.

    The method combines smoothed RMS and motion levels and removes sections with
    dominant whistle energy. It returns a dataframe with ``start`` and ``end``
    columns marking contiguous in-play intervals.
    """

    if audio_df.empty and motion_df.empty:
        return pd.DataFrame(columns=["start", "end"])

    if audio_df.empty:
        motion_times_series = _time_series(motion_df)
        base_times = motion_times_series.to_numpy() if motion_times_series is not None else np.array([], dtype=float)
        rms = np.zeros_like(base_times)
        whistle = np.zeros_like(base_times)
        motion_interp = motion_df.get("motion_smooth", pd.Series(0)).to_numpy()
    else:
        audio_times_series = _time_series(audio_df)
        base_times = audio_times_series.to_numpy() if audio_times_series is not None else np.array([], dtype=float)
        rms = audio_df.get("rms_smooth", audio_df.get("rms", pd.Series(0))).to_numpy()
        whistle = audio_df.get("whistle_score", pd.Series(0)).to_numpy()
        motion_times_series = _time_series(motion_df)
        motion_times = motion_times_series.to_numpy() if motion_times_series is not None else np.array([], dtype=float)
        motion_interp = interpolate_to(
            base_times,
            motion_times,
            motion_df.get("motion_smooth", pd.Series(0)).to_numpy(),
        )

    if base_times.size == 0:
        return pd.DataFrame(columns=["start", "end"])

    rms_thr = np.percentile(rms, 35) if rms.size else 0.0
    motion_thr = np.percentile(motion_interp, 35) if motion_interp.size else 0.0
    whistle_thr = np.percentile(whistle, 75) if whistle.size else np.inf

    active = (rms >= rms_thr) | (motion_interp >= motion_thr)
    quiet_whistle = whistle < whistle_thr
    mask = active & quiet_whistle

    spans: List[Tuple[float, float]] = []
    if mask.any():
        start_idx = None
        for i, flag in enumerate(mask):
            if flag and start_idx is None:
                start_idx = i
            elif not flag and start_idx is not None:
                t0 = float(base_times[start_idx])
                t1 = float(base_times[i - 1])
                if t1 - t0 >= min_run:
                    spans.append((t0, t1))
                start_idx = None
        if start_idx is not None:
            t0 = float(base_times[start_idx])
            t1 = float(base_times[-1])
            if t1 - t0 >= min_run:
                spans.append((t0, t1))

    return pd.DataFrame(spans, columns=["start", "end"])


if __name__ == "__main__":  # pragma: no cover - smoke test
    import argparse

    parser = argparse.ArgumentParser(description="Quick feature extraction smoke test")
    parser.add_argument("video", type=Path)
    args = parser.parse_args()

    audio = compute_audio_features(args.video, AudioFeatureConfig())
    motion = compute_motion_features(args.video)
    print(audio.head())
    print(motion.head())

