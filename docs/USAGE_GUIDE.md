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

## 5. Extended Workflows & Social Reels

### 5.1 Motion filtering before ranking

The optional `05_filter_by_motion.py` script culls low-action windows via motion and ball-speed heuristics; tune jersey HSV bounds and sensitivity flags to match your footage.【F:README.md†L85-L103】

### 5.2 Cheer-anchored Top-10 lists

When you have a CSV of crowd-cheer detections, run `08b_build_top10_cheers.py` to force those moments into the final list before filling remaining slots with the highest `action_score` clips.【F:README.md†L76-L83】

### 5.3 Portrait auto-framing and reel finishing

1. **Track per-frame centers and zoom.** Invoke `autoframe.py` on each source clip to generate a `*_autoframe.csv` and optional preview overlay. The CLI exposes granular goal anchoring, ball fusion, and smoothing controls (`--lead_frames`, `--deadband_xy`, `--goal_side`, `--ball-detector`, `--preview`, `--poly_out`, etc.) so you can tune behaviour per footage. Add `--poly_out clip.ps1vars` when you want a quick polynomial vars file straight from the tracker.【F:autoframe.py†L1204-L1344】【F:autoframe.py†L1970-L1984】【F:README_AUTOFAME.md†L13-L139】

   ```powershell
   python .\autoframe.py --in "C:\clips\DJI_20251001_203314_740_video.mov" `
       --csv .\out\DJI_20251001_203314_740_video_autoframe.csv --profile portrait `
       --roi goal --goal_side auto --preview .\out\DJI_20251001_203314_740_preview.mp4
   ```

2. **Fit smooth FFmpeg expressions.** Run `fit_expr.py` when you prefer tighter control than the inline `--poly_out`. The script reads tracker metadata, honours profile/ROI zoom limits from `configs/zoom.yaml`, and writes `$cxExpr`, `$cyExpr`, `$zExpr` assignments into a `.ps1vars` file alongside comments capturing the run flags (`--degree`, `--lead-ms`, `--deadzone`, goal bias, celebration locks, etc.).【F:fit_expr.py†L1-L640】

   ```powershell
   python .\fit_expr.py --csv .\out\DJI_20251001_203314_740_video_autoframe.csv `
       --out .\out\DJI_20251001_203314_740_video.ps1vars --profile portrait --roi goal --degree 5
   ```

3. **Render a 9:16 master.** Feed the expressions to `make_reel.ps1`, which sources the vars file, clamps crop math, renders portrait or landscape outputs, and can emit debug/compare overlays on demand. Pair it with the social preset in `config/reels.json` (`tiktok_9x16`: 1080×1920 @24 fps) whenever you need matching frame geometry for downstream packaging. Portrait runs automatically land in `out/portrait_reels/clean/<clip_basename>_WIDE_portrait_FINAL.mp4`, with optional crop/compare renders ready for `out/portrait_reels/debug/`.【F:make_reel.ps1†L1-L144】【F:config/reels.json†L1-L5】

   ```powershell
   pwsh -File .\make_reel.ps1 -Input "C:\clips\DJI_20251001_203314_740_video.mov" `
         -Vars .\out\DJI_20251001_203314_740_video.ps1vars -Profile portrait `
         -Debug .\out\portrait_reels\debug\DJI_20251001_203314_740_portrait_DEBUG.mp4
   ```

4. **Enhance the grade.** `.\enhance.ps1` wraps `tools/auto_enhance/auto_enhance.ps1`, scanning files or folders, appending `_ENH.mp4`, and exposing `rec709_smart`, `rec709_basic`, and `punchy` looks plus CRF/preset knobs. The inner script retries without `normalize` if your FFmpeg build lacks the filter, making it safe for short social clips.【F:enhance.ps1†L1-L10】【F:tools/auto_enhance/auto_enhance.ps1†L1-L134】

   ```powershell
   pwsh -File .\enhance.ps1 -In .\out\DJI_20251001_203314_740_portrait.mp4 -Profile rec709_smart -Crf 18 -Preset fast
   ```

5. **Apply branding.** `tools/tsc_brand.ps1` overlays Tulsa SC assets, titles, watermarks, optional lower thirds, and end cards. Passing `-Aspect 9x16` switches to the portrait ribbon/watermark set automatically, while font checks warn when Montserrat weights are missing. When the input lives under `out/portrait_reels/clean/`, the script defaults the branded export to `out/portrait_reels/branded/<clip_basename>_WIDE_portrait_FINAL.mp4` so the postable deliverable always resolves to the same filename.【F:tools/tsc_brand.ps1†L1-L214】

   ```powershell
   pwsh -File .\tools\tsc_brand.ps1 -In .\out\portrait_reels\clean\DJI_20251001_203314_740_WIDE_portrait_FINAL.mp4 `
         -Aspect 9x16 -Title "Claire Practice" -Subtitle "2025-10-01" -Watermark -EndCard
   ```

   The `out/portrait_reels/` tree is now the single source of truth for social-ready exports:

   ```
   out/
     portrait_reels/
       clean/    # postable masters without branding
       branded/  # matching FINAL renders with brand overlays
       debug/    # optional overlays, guides, and comparison passes
   ```

   **Data hand-off recap:** `autoframe.py` → CSV/preview (and optional `.ps1vars`) → `fit_expr.py` `.ps1vars` → `make_reel.ps1` portrait master → `.\enhance.ps1` `_ENH` grade → `tools/tsc_brand.ps1` branded final. Ensure FFmpeg/FFprobe remain on `PATH` for every PowerShell stage, and install the Montserrat family under `fonts/` to keep text overlays intact.【F:make_reel.ps1†L54-L117】【F:tools/auto_enhance/auto_enhance.ps1†L83-L133】【F:tools/tsc_brand.ps1†L37-L168】

6. **Batch processing.** `scripts/batch_autoframe.py` orchestrates `track_ball_cli.py` and `plan_render_cli.py` across entire clip folders with identical defaults to the documented PowerShell chain. Flags expose tracker ROI padding, YOLO confidence, zoom rails, and overwrite behaviour, keeping repeated runs consistent across platforms.【F:scripts/batch_autoframe.py†L1-L195】

### 5.4 Social reel shell helpers

The `05_make_social_reel.sh` helper sorts the detection CSV by score, grabs the top `N` clips, and concatenates them into a vertical or square deliverable with optional music bed and configurable audio offset. Override `TARGET_AR`, `MAX_LEN`, `BITRATE`, `MUSIC`, or `audioOffset` in the environment to match each platform before the script pads/crops and titles the mix.【F:05_make_social_reel.sh†L1-L47】

### 5.5 Image polish without portrait work

Outside the portrait pipeline you can still run `.\enhance.ps1` directly on landscape reels or raw match exports to normalize exposure and colour before publishing.【F:enhance.ps1†L1-L10】【F:tools/auto_enhance/auto_enhance.ps1†L43-L134】

### 5.6 Offline ball-path planning recipe

When you only need a stabilised ball trace and zoom plan (for example, to feed `render_follow_unified.py` without the rest of the portrait toolchain) run the lightweight `tools/ball_path_offline.py` helper. The script accepts the video path plus two tuning knobs—candidate budget and search radius—and optional manual initialisation flags. There are no `--dx/--dy/--sh` parameters; those belong to older planners and will be rejected by this CLI.【F:tools/ball_path_offline.py†L360-L409】

```powershell
$Proj = "C:\Users\scott\soccer-video"
$Stem = "042__2025-09-20__TSC_vs_RVFC__CORNER__t3992400.00-t4057200.00"
$Clip = Join-Path $Proj "out\atomic_clips\$Stem.mp4"
$Plan = Join-Path $Proj "out\follow_diag\$Stem\auto_hill\plan.jsonl"

New-Item -Force -ItemType Directory -Path (Split-Path $Plan) | Out-Null

python tools\ball_path_offline.py `
  --in "$Clip" `
  --out "$Plan" `
  --search-r 120 `
  --max-cands 5 `
  --init-manual
```

Use `--init-manual` the first time you build a plan so you can click the starting ball location; afterwards you can rerun without it if the cached template remains valid. Adjust `--search-r` when the ball moves faster than the default 280 px window, and lower `--max-cands` if you want to keep the beam search tight in cluttered frames.【F:tools/ball_path_offline.py†L360-L423】

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

## 8. Script Inventory & Legacy Utilities

| Script | Purpose |
| --- | --- |
| `02_detect_events.py` | Standalone motion/audio detector that identifies passes, shots, and intensity spikes, writing `out/highlights.csv` with start/end/score for ad-hoc runs outside the packaged CLI.【F:02_detect_events.py†L1-L188】 |
| `03_motion_zoom.py` | Early auto-zoom prototype that estimates motion regions, writes `out/crops.jsonl`, and pipes crops to FFmpeg or OpenCV when you need a quick follow-cam without the full autoframe stack.【F:03_motion_zoom.py†L1-L120】 |
| `track_ball_cli.py` | YOLO-accelerated ball tracker with colour gating, Kalman filtering, and JSON/CSV output used by batch planners and the autoframe fitter to keep the ball centred.【F:track_ball_cli.py†L1-L144】 |
| `plan_render_cli.py` | Camera planner that smooths ball tracks, enforces pan/zoom slew limits, and renders directed crops for follow-cam deliveries from the tracked CSV.【F:plan_render_cli.py†L1-L160】 |
| `render_follow_cli.py` | Legacy follow-camera renderer that turns a `track_csv` into 608×1080 crops using slew/acceleration, zoom hysteresis, and context-aware padding around nearby players.【F:render_follow_cli.py†L21-L200】 |
| `render_follow_autoz*.py` | Notebook-style variants tuned for different looks (e.g., cinematic, realzoom) that read YOLO tracks, apply Savitzky–Golay smoothing, and constrain pan/zoom behaviour via explicit parameter blocks.【F:render_follow_autoz.py†L1-L160】 |

Armed with these procedures and references, an operator can ingest raw match footage, iterate on highlight selections, and ship branded reels with confidence.
