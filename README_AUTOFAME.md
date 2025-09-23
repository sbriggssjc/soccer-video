# Autoframe Polynomial Zooms

These helpers keep the crop glued to the play instead of the center of the
frame. The flow is unchanged:

1. Track dense motion to produce smoothed per-frame crop centers and zoom
   factors.
2. Fit cubic FFmpeg expressions so we can feed the crop math straight into
   `ffmpeg` without thousands of literal numbers.
3. Render the reel with the existing even-dimension crop + scale chain, or
   produce a debug overlay to sanity check the box.

## 1. Track motion & emit CSV

```bash
python autoframe.py --in clip.mp4 --csv clip_zoom.csv --preview DEBUG__.mp4 \
    --roi goal --goal_side auto --profile portrait
```

The tracker now detects goal mouths, predicts the motion center using dense
optical flow, and keeps contextual zoom under control. Every knob is still
exposed through the CLI so you can tune how aggressively the crop reacts. All
parameters have sensible defaults, but every flag can be overridden:

| Flag | Default | Description |
| --- | --- | --- |
| `--lead N` | `6` | Extra EMA-based lead (frames) on top of the optical-flow prediction. |
| `--lead_ms MS` | `180` | Predictive lead from dense flow (milliseconds). |
| `--deadband PX` | `10` | Legacy scalar deadband retained for compatibility. |
| `--deadband_xy PX,PY` | `12,12` | Axis-specific deadband to kill micro jitter. |
| `--slew_xy PX,PY` | `40,40` | Max commanded center change per frame (pixels). |
| `--slew_z DZ` | `0.06` | Max zoom delta per frame (unitless). |
| `--padx R` | `0.22` | Extra horizontal padding around the action (fraction). |
| `--pady R` | `0.18` | Extra vertical padding around the action (fraction). |
| `--zoom_min Z` | `1.08` | Minimum zoom (crop scale denominator). |
| `--zoom_max Z` | `2.40` | Maximum zoom (larger ⇒ tighter crop). |
| `--zoom_k K` | `0.85` | Context-to-zoom responsiveness gain. |
| `--zoom_asym OUT,IN` | `0.75,0.35` | Separate easing for zoom-out vs zoom-in. |
| `--anchor_weight W` | `0.35` | Default blend weight toward the detected goal. |
| `--anchor_iou_min T` | `0.15` | Increase goal pull when IoU falls below this. |
| `--goal_side SIDE` | `auto` | Force `left`/`right` goal or let the tracker decide. |
| `--smooth_ema α` | `0.35` | EMA smoothing weight for the tracked center. |
| `--smooth_win N` | `0` | Optional odd-length boxcar on raw centers (0 disables). |
| `--chaos_thresh T` | `0.18` | Motion magnitude that triggers a defensive zoom-out. |
| `--hold_frames N` | `8` | Freeze the crop for N frames after regaining lock. |
| `--conf_floor C` | `0.15` | Confidence threshold used when evaluating locks. |
| `--flow_thresh T` | `0.18` | Threshold after normalised optical-flow magnitude. |

Existing flags still work:

* `--roi` toggles tuned presets (`generic` or `goal`).
* `--profile` controls the aspect ratio (`portrait` → 9:16, `landscape` → 16:9).
* `--preview` writes the debug overlay (crop box, crosshair, goal anchor, IoU).
* `--compare` writes an optional side-by-side stack of raw vs. overlayed frames.

The CSV now contains one row per frame with columns:

```
frame,cx,cy,z,w,h,x,y,conf,crowding,flow_mag,goal_x,goal_y,goal_w,goal_h,anchor_iou
```

The first four columns (`frame,cx,cy,z`) are unchanged for downstream tooling.
Additional fields expose the smoothed motion magnitude, the tracked goal box,
and per-frame goal IoU so you can spot mis-detections. Header comments also
include the CLI flags (`# cli=...`) for reproducibility alongside the usual
`# fps=` and zoom bounds.

## 2. Fit FFmpeg expressions

```bash
python fit_expr.py --csv clip_zoom.csv --out clip_zoom.ps1vars --profile portrait
```

`fit_expr.py` ignores extra CSV columns automatically. If the header provided
`zoom_min` / `zoom_max`, those bounds override the YAML defaults so the cubic
polynomial stays within the measured range.

The generated `.ps1vars` file now starts with a tiny comment block that echoes
the run flags (both the fitter and the tracked CSV). Drop it straight into the
PowerShell reel scripts to keep a trace of how the expressions were produced.

The `.ps1vars` file is just PowerShell assignments for `$cxExpr`, `$cyExpr`, and
`$zExpr`. Coefficients use `n` as the frame index and expand powers as
`n*n`, `n*n*n` so you can drop them directly into FFmpeg expressions.

## 3. Render with ffmpeg crop + debug overlay

```powershell
pwsh .\make_reel.ps1 -Input clip.mp4 -Vars clip_zoom.ps1vars `
    -Output clip_autoframe.mp4 -Debug clip_autoframe_debug.mp4
```

* The script keeps the even-dimension crop + scale (`scale=1080:1920` for
  portrait, `scale=1920:1080` for landscape).
* `$cxExpr/$cyExpr` are wrapped in `clip()` for safety; `$zExpr` is already
  range-limited in the PowerShell vars file.
* Pass `-Compare side_by_side.mp4` to get a horizontal stack of the final crop
  and the debug overlay.

## Tuning tips

* Increase `--deadband` if the crop jitters during static play.
* Increase `--slew_xy` / `--slew_z` for snappier camera moves.
* Bump `--lead` when the crop lags fast counter-attacks.
* Zoom is unitless: larger values tighten the crop. Clamp with
  `--zoom_min`/`--zoom_max` to keep the box from hitting the frame edges.
* If the camera holds too long when players leave frame, reduce
  `--hold_frames` or lower `--conf_floor`.
