# Soccer Highlights Suite

`soccerhl` is a Windows-friendly end-to-end toolkit for turning a full match
recording into polished highlight reels. The pipeline combines motion analysis,
audio excitement, peak tightening, clip exports, smart ranking, and final reels
with transitions.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -e .
```

The toolkit depends on FFmpeg/FFprobe and the libraries listed in
`pyproject.toml`. Install FFmpeg separately and ensure it is on your `PATH`.

## Configuration

All commands read settings from `config.yaml` (or another YAML file passed via
`--config`). Configuration includes paths, detection thresholds, jersey colors,
output profiles, and encoding parameters. Profiles are provided for broadcast,
social-vertical, and coach-review outputs.

## CLI Overview

Each pipeline stage is available as a subcommand of `soccerhl`:

```powershell
soccerhl detect --video .\out\full_game_stabilized.mp4 --pre 1.0 --post 2.0 --max-count 40
soccerhl shrink --video .\out\full_game_stabilized.mp4 --csv .\out\highlights.csv --out .\out\highlights_smart.csv --aspect vertical --pre 3 --post 5 --bias-blue
soccerhl clips --video .\out\full_game_stabilized.mp4 --csv .\out\highlights_smart.csv --outdir .\out\clips --workers 4
soccerhl topk --candirs .\out\clips,.\out\clips_acc --k 10 --max-len 18
soccerhl reel --list .\out\smart_top10_concat.txt --out .\out\reels\top10.mp4 --profile social-vertical
```

All examples use Windows PowerShell syntax without line continuations. The
commands are idempotent—rerun them to resume processing, and add `--overwrite`
to regenerate clips when needed.

## Pipeline Summary

1. **detect** – merges motion and audio scores per second, applies hysteresis
   and sustain filters, and writes `out/highlights.csv`.
2. **shrink** – tightens highlight windows around motion peaks (optionally
   jersey-biased) and can write tracked social clips.
3. **clips** – exports frame-accurate MP4 clips with small audio fades using
   parallel FFmpeg workers.
4. **topk** – scores candidate clips, trims slow starts, and writes both CSV and
   concat list files for the best plays.
5. **reel** – renders finished reels with title slates, numbered overlays,
   crossfades, and gentle audio ducking.

Each command updates `out/report.md` with summary statistics so you always know
how many windows, clips, and reel duration were produced.


### Cheer-anchored Top-10

When you have a CSV of cheer timestamps (see `09_detect_cheers.py`), run
`08b_build_top10_cheers.py` to guarantee those big moments appear in the final
Top-10 reel. The script keeps up to four well-spaced cheers, pads each window
with additional context, fills remaining slots using the highest
`action_score` clips from `out/highlights_filtered.csv`, then renders
`out/top10.mp4` along with the individual clips and concat list.

## Motion Filter Tuning Notes

The optional `05_filter_by_motion.py` stage keeps the most action-packed windows
by combining motion strength, a rough ball-speed proxy, and a navy jersey mask.
You can adjust several parameters when your footage or uniforms differ from the
defaults:

* **Jersey mask** – `--team-hsv-low`/`--team-hsv-high` accept `H,S,V` triplets in
  OpenCV ranges. The dark-navy defaults are `105,70,20` through `130,255,160`.
  Raise the upper V toward `190` when clips look too dim, or increase the lower
  S to ~`90` if the mask is catching sky and bleachers.
* **Sensitivity knobs** – Lower `--min-flow-mean` (alias `--min-flow`) when fast
  plays are being dropped, or reduce `--min-ball-speed` when the ball appears
  small or far from the camera. Raising `--min-center-ratio` nudges the filter
  toward activity in the attacking thirds.

The filter now drops windows with low residual motion (such as pure camera pans)
and those lacking a moving ball or visible pitch, so expect filler "in-between"
clips to disappear as thresholds tighten.


## Examples & Tests

`examples/generate_sample.py` creates a small synthetic match clip that drives a
regression test. After running the generator, process the clip with the
PowerShell commands shown in `examples/README.md`.

Run the automated test to verify the entire pipeline on the synthetic clip:

```powershell
pytest -k pipeline
```

This produces a short reel under `examples/out/reels/` as part of the test run.

## Goal-aware Autoframe

`autoframe.py` now includes goal anchoring and context-aware zoom to keep plays
framed even when the camera drifts. Run it with `--roi goal` to auto-detect the
goal mouth, predict the motion center with dense optical flow, and blend toward
the net whenever the crop slips away. New flags like `--lead_ms`,
`--deadband_xy`, `--goal_side`, and `--anchor_weight` make the behaviour easy to
dial in, while `--preview` / `--compare` render debug overlays showing the crop
box, crosshair, goal outline, and per-frame IoU. See `README_AUTOFAME.md` for a
complete rundown of the tuning options.
