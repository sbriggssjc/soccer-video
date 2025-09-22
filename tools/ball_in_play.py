"""Estimate ball-in-play windows using lightweight audiovisual heuristics."""
from __future__ import annotations

import argparse
import io
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
        return audio, sr
    except Exception:  # pragma: no cover - fallback path
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "wav",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-",
        ]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
            data = proc.stdout.read() if proc.stdout else b""
        if not data:
            raise RuntimeError("Unable to decode audio track")
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr


def _robust_normalise(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = np.quantile(values, 0.1)
    hi = np.quantile(values, 0.9)
    if math.isclose(hi, lo):
        hi = lo + 1e-6
    normed = (values - lo) / (hi - lo)
    return np.clip(normed, 0.0, 1.0)


def _compute_flow(path: Path, hop: float) -> Tuple[np.ndarray, np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(hop * fps)))
    prev_gray = None
    times = []
    mags = []
    frame_index = 0
    ok, frame = cap.read()
    while ok:
        if frame_index % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray_small,
                    None,
                    0.5,
                    1,
                    15,
                    3,
                    5,
                    1.1,
                    0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mags.append(float(np.mean(mag)))
            else:
                mags.append(0.0)
            times.append(frame_index / fps)
            prev_gray = gray_small
        frame_index += 1
        ok, frame = cap.read()
    cap.release()
    if not times:
        return np.array([0.0]), np.array([0.0]), fps
    return np.asarray(times), np.asarray(mags), float(fps)


def estimate_in_play(
    video: Path,
    hop: float = 0.10,
    audio_weight: float = 0.4,
    flow_weight: float = 0.6,
    combined_threshold: float = 0.45,
    reset_hf_threshold: float = 0.55,
    reset_drop_threshold: float = 0.25,
) -> pd.DataFrame:
    """Estimate ball-in-play and reset signals for the provided video."""

    audio, sr = _load_audio(video)
    hop_length = max(1, int(sr * hop))
    frame_length = max(1024, hop_length * 4)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    n_fft = 2048
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hf_mask = freqs >= 3000
    hf_energy = spec[hf_mask].mean(axis=0) if hf_mask.any() else np.zeros(spec.shape[1])
    total_energy = spec.sum(axis=0) + 1e-6
    hf_ratio = hf_energy / total_energy

    flow_times, flow_values, fps = _compute_flow(video, hop)
    flow_interp = np.interp(times, flow_times, flow_values, left=flow_values[0], right=flow_values[-1])

    rms_norm = _robust_normalise(rms)
    flow_norm = _robust_normalise(flow_interp)
    hf_norm = _robust_normalise(hf_ratio)

    combined = (audio_weight * rms_norm) + (flow_weight * flow_norm)
    win = max(1, int(round(0.6 / hop)))
    combined_smooth = pd.Series(combined).rolling(window=win, min_periods=1, center=True).mean().to_numpy()
    in_play = combined_smooth > combined_threshold

    rms_series = pd.Series(rms_norm, index=times)
    hf_series = pd.Series(hf_norm, index=times)
    combined_series = pd.Series(in_play, index=times)

    rms_diff = rms_series.diff().fillna(0.0)
    reset = (hf_series > reset_hf_threshold) & (rms_series < 0.3) & (rms_diff < -reset_drop_threshold)

    result = pd.DataFrame(
        {
            "in_play": combined_series.astype(bool),
            "reset": reset.astype(bool),
            "audio_rms": rms_series,
            "hf_ratio": hf_series,
            "flow": pd.Series(flow_interp, index=times),
        }
    )
    result.index.name = "time"
    result.attrs["fps"] = fps
    result.attrs["sr"] = sr
    return result


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate ball-in-play segments")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hop", type=float, default=0.10)
    parser.add_argument("--audio-weight", type=float, default=0.4)
    parser.add_argument("--flow-weight", type=float, default=0.6)
    parser.add_argument("--combined-threshold", type=float, default=0.45)
    parser.add_argument("--reset-hf-threshold", type=float, default=0.55)
    parser.add_argument("--reset-drop-threshold", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    df = estimate_in_play(
        args.video,
        hop=args.hop,
        audio_weight=args.audio_weight,
        flow_weight=args.flow_weight,
        combined_threshold=args.combined_threshold,
        reset_hf_threshold=args.reset_hf_threshold,
        reset_drop_threshold=args.reset_drop_threshold,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

