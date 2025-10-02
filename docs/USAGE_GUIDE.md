# Soccer Highlights Suite – Operator Handbook

This guide explains how to take a full-length match video and turn it into finished highlight reels with the `soccerhl` toolkit. It walks through environment setup, configuration, the step-by-step pipeline, and the supporting utilities that ship with the repository so that an editor can confidently run the workflow end to end.

## 1. Install and Prepare the Environment

1. **Create a virtual environment and install the package** using editable mode so CLI entry points are available:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   pip install -e .
   ```
   FFmpeg/FFprobe must be installed separately and present on your `PATH`, because every video and audio operation relies on them.【F:README.md†L8-L18】

2. **Optional dependencies.** The detection and shrink steps opportunistically import `librosa` for audio guidance and fall back gracefully when it is missing. Install it (already listed in `pyproject.toml`) if you want audio-aware behaviour.【F:soccer_highlights/detect.py†L28-L45】【F:soccer_highlights/shrink.py†L27-L43】

3. **Project layout.** All pipeline commands default to writing under `out/` next to your source video (`full_game_stabilized.mp4` by default). You can override paths in `config.yaml` or with CLI flags at runtime.【F:soccer_highlights/config.py†L120-L149】【F:soccer_highlights/cli.py†L24-L69】

## 2. Configure the Run

1. **Start from `config.yaml`.** Every CLI command accepts `--config` and loads defaults when the file is missing, so you can begin with the built-in settings and progressively customise.【F:soccer_highlights/cli.py†L17-L23】【F:soccer_highlights/config.py†L253-L266】

2. **Key sections to review:**
   * `paths.video` / `paths.output_dir` – input video and output root.【F:soccer_highlights/config.py†L120-L124】
   * `detect` – adaptive-threshold parameters, max highlight count, hysteresis, and which event labels to exclude before merging.【F:soccer_highlights/config.py†L126-L139】
   * `shrink` – whether to use `smart` optical-flow tightening, default pre/post roll, optional jersey bias, and tracked clip export location.【F:soccer_highlights/config.py†L142-L161】
   * `clips` – FFmpeg preset, CRF, audio bitrate, worker count, and overwrite toggle.【F:soccer_highlights/config.py†L164-L170】
   * `rank` – Top-K size, maximum clip length, motion sustain requirement, and required tail padding after the chosen in-point.【F:soccer_highlights/config.py†L173-L177】
   * `reels`/`profiles` – default reel title plus render profiles for broadcast, social-vertical, and coach-review formats (dimensions, fps, crossfade frames, label placement).【F:soccer_highlights/config.py†L180-L246】
   * `colors` – HSV presets for pitch and jersey masks and whether to auto-calibrate before smart shrinking.【F:soccer_highlights/config.py†L197-L208】

3. **Report files.** The pipeline keeps a rolling JSON/Markdown report in the output directory so you can track what has already run; this is automatically maintained by each stage.【F:soccer_highlights/utils.py†L76-L110】

## 3. Run the Core Pipeline

The `soccerhl` CLI exposes one subcommand per stage. Commands are idempotent, so rerunning a step will pick up where it left off unless you pass an overwrite flag.【F:README.md†L26-L41】

### Step 1 – Detect candidate highlights

```
soccerhl detect --video path\to\full_game.mp4 --out out\highlights.csv
```

* Combines per-second motion energy from Farneback optical flow with audio RMS, blended according to `detect.audio_weight`. Scores are normalised and compared against an adaptive threshold (mean + std-dev multiplier) with hysteresis and sustain filters to avoid flicker.【F:soccer_highlights/detect.py†L48-L156】
* Automatically pads each detection by `detect.pre` / `detect.post`, merges overlapping windows when the gap is below `detect.min_gap`, and limits the total to `detect.max_count`.【F:soccer_highlights/detect.py†L99-L171】
* Goal-specific detections from the dedicated module are appended unless excluded, and banned event labels are filtered out before merging.【F:soccer_highlights/detect.py†L160-L170】
* Results are written to CSV and summarised in the pipeline report.【F:soccer_highlights/detect.py†L173-L188】【F:soccer_highlights/utils.py†L41-L48】

**Operational tips**
* If the video lacks audio or `librosa` fails, the system logs a warning and continues with motion-only scores. This is normal for silent footage.【F:soccer_highlights/detect.py†L28-L45】
* Use `--pre`, `--post`, `--max-count`, or `--audio-weight` to quickly override configuration defaults during experimentation.【F:soccer_highlights/cli.py†L46-L71】

### Step 2 – Tighten windows around the action

```
soccerhl shrink --csv out\highlights.csv --out out\highlights_tight.csv --mode smart
```

* `smart` mode re-tracks each window using dense optical flow (optionally biasing toward the configured jersey colour) and blends in audio onset strength to find the true peak moment.【F:soccer_highlights/shrink.py†L205-L235】
* Goal events automatically expand to cover the full ball-in-play segment surrounding the peak so you do not miss celebration or setup context.【F:soccer_highlights/shrink.py†L108-L147】【F:soccer_highlights/shrink.py†L229-L235】
* Setting `--write-clips PATH` records motion-tracked vertical or horizontal crops directly from this stage using the specified aspect ratio and zoom factor.【F:soccer_highlights/shrink.py†L218-L244】
* `simple` mode provides deterministic padding around the original window when you prefer a quicker approximation.【F:soccer_highlights/shrink.py†L248-L254】

### Step 3 – Export polished clips

```
soccerhl clips --csv out\highlights_tight.csv --outdir out\clips --workers 4
```

* Reads every highlight that meets the minimum duration and constructs an FFmpeg command with frame-accurate in/out points, soft audio fades, and H.264/AAC encoding tuned by your clip settings.【F:soccer_highlights/clips.py†L24-L70】【F:soccer_highlights/clips.py†L73-L103】
* Uses a thread pool to parallelise exports and writes a `clips_metadata.csv` alongside the rendered files (event label, start/end, score) for downstream ranking.【F:soccer_highlights/clips.py†L73-L125】
* Respect existing files unless `clips.overwrite` or `--overwrite` is set, which keeps re-runs fast.【F:soccer_highlights/clips.py†L64-L70】【F:soccer_highlights/config.py†L164-L170】

### Step 4 – Rank for the Top-K reel

```
soccerhl topk --candirs out\clips --k 10 --max-len 18
```

* Aggregates candidate clips from the provided directories, merging any metadata exported earlier. Event keywords (e.g., “goal”, “pass”) classify plays into shot or buildup categories to determine trimming heuristics.【F:soccer_highlights/rank.py†L73-L111】
* Samples each clip at 6 fps to measure motion coverage and magnitude, finds sustained activity windows, and anchors the peak moment. Category-specific pre/post padding yields natural-feeling edits while enforcing a minimum live-play tail.【F:soccer_highlights/rank.py†L114-L193】【F:soccer_highlights/rank.py†L390-L405】
* Writes both a scoring CSV and an FFmpeg concat list with in/out points capped by `rank.max_len`, plus a combined list that appends any separate goal reel when present.【F:soccer_highlights/rank.py†L364-L405】

### Step 5 – Render the reel

```
soccerhl reel --list out\smart_top10_concat.txt --out out\reels\top10.mp4 --profile social-vertical
```

* Parses the concat list, prepends a title slate, numbers each clip, and renders using the selected profile’s dimensions, frame rate, and crossfade duration.【F:soccer_highlights/reels.py†L25-L148】
* Video transitions rely on `xfade` while audio uses `acrossfade` with gentle ducking; outputs inherit the clip encoding settings (preset, CRF, audio bitrate).【F:soccer_highlights/reels.py†L107-L151】
* Returns the total play length so you can log deliverable durations. Results are written to the configured output directory and logged in the report.【F:soccer_highlights/reels.py†L128-L156】【F:soccer_highlights/utils.py†L76-L84】

## 4. Monitor Outputs and Reports

* **Highlight CSVs** (`highlights.csv`, `highlights_tight.csv`, `smart_top10.csv`) contain the canonical in/out/score data for each stage.【F:soccer_highlights/utils.py†L25-L48】
* **Clip metadata** enables ranking heuristics to recognise event types and prior trimming decisions.【F:soccer_highlights/clips.py†L110-L125】
* **`out/report.json` + `out/report.md`** are updated after every command with counts, durations, thresholds, and output paths so production staff can confirm completion at a glance.【F:soccer_highlights/utils.py†L76-L110】

## 5. Extended Workflows

1. **Motion filtering before ranking.** The optional `05_filter_by_motion.py` script culls low-action windows via motion and ball-speed heuristics; tune jersey HSV bounds and sensitivity flags to match your footage.【F:README.md†L85-L103】

2. **Cheer-anchored top tens.** When you have a CSV of crowd-cheer detections, run `08b_build_top10_cheers.py` to force those moments into the final list before filling remaining slots with the highest `action_score` clips.【F:README.md†L76-L83】

3. **Brand overlays.** Use `tools/tsc_brand.ps1` to wrap a finished reel with Tulsa Soccer Club graphics, watermarks, lower thirds, and optional end cards. The script automatically swaps between 16×9 and 9×16 art, verifies ribbon/watermark assets, and renders drawtext overlays for title/subtitle copy when the Montserrat font family is installed under `fonts/`.【F:README.md†L59-L74】【F:tools/tsc_brand.ps1†L1-L213】

4. **Autoframe for social cuts.**
   * Track motion and goal anchors with `autoframe.py`, which now outputs per-frame centers, zoom, and diagnostics while exposing detailed tuning flags (`--lead`, `--deadband_xy`, `--goal_side`, etc.).【F:README_AUTOFAME.md†L13-L65】
   * Fit smooth cubic expressions with `fit_expr.py` and feed them into the PowerShell reel scripts to drive FFmpeg’s `crop`/`scale` filters without thousands of raw numbers.【F:README_AUTOFAME.md†L68-L99】
   * Batch processing is available via `scripts/batch_autoframe.py`, mirroring the recommended directory layout and exposing planner/tracker flags from the CLI.【F:README.md†L120-L145】

5. **Social-first reel packaging.** The `05_make_social_reel.sh` helper sorts the detection CSV by score, grabs the top `N` clips, and concatenates them into a vertical or square deliverable with optional music bed and configurable audio offset. Override `TARGET_AR`, `MAX_LEN`, `BITRATE`, `MUSIC`, or `audioOffset` in the environment to match each platform before the script pads/crops and titles the mix.【F:05_make_social_reel.sh†L1-L47】

6. **Image polish pass.** Run `.\enhance.ps1` (a thin wrapper over `tools/auto_enhance/auto_enhance.ps1`) to batch-normalise highlights into broadcast-safe Rec.709 levels. Profiles such as `rec709_smart`, `rec709_basic`, and `punchy` adjust contrast/saturation curves, retry without `normalize` when FFmpeg lacks the filter, and emit `_ENH` clips alongside the originals so you can choose the preferred grade.【F:enhance.ps1†L1-L12】【F:tools/auto_enhance/auto_enhance.ps1†L1-L134】

## 6. Quality Assurance

* **Regression test.** Generate the synthetic sample and run `pytest -k pipeline` to exercise the full stack on a miniature match; it writes output under `examples/out/reels/`.【F:README.md†L106-L118】
* **Manual spot checks.** Review the generated `report.md`, skim the highest-scoring clips, and confirm the reel crossfades around the expected timeline.

## 7. Troubleshooting Checklist

| Symptom | Likely cause & fix |
| --- | --- |
| `librosa` warnings and silent audio scores | Install the optional dependency or ignore if your footage lacks audio channels—the system continues with motion-only scoring.【F:soccer_highlights/detect.py†L28-L45】 |
| FFmpeg command failures | Verify FFmpeg/FFprobe availability and paths; the `run_command` helper surfaces stderr output for quick diagnosis.【F:soccer_highlights/io.py†L36-L62】 |
| Smart shrink produces little motion context | Enable jersey bias with `--bias-blue` (adjust `colors.team_primary` first) or fall back to `--mode simple` for deterministic padding.【F:soccer_highlights/shrink.py†L205-L267】【F:soccer_highlights/config.py†L197-L208】 |
| Ranked clips start late | Reduce `rank.min_tail` or raise `rank.sustain` to change how aggressively the ranking step trims the lead-in.【F:soccer_highlights/config.py†L173-L177】【F:soccer_highlights/rank.py†L114-L193】 |
| Reel numbering overlaps text | Adjust the chosen profile’s `label_position` or create a custom profile entry in `config.yaml` for unique overlays.【F:soccer_highlights/config.py†L180-L246】【F:soccer_highlights/reels.py†L79-L95】 |

Armed with these procedures and references, an operator can ingest raw match footage, iterate on highlight selections, and ship branded reels with confidence.
