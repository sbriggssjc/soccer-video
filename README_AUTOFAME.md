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
python autoframe.py --in clip.mp4 --csv clip_zoom.csv --preview clip_debug.mp4 \
    --roi generic --profile portrait
```

The tracker now exposes the full camera-op model via CLI flags so you can tune
how the crop follows the play. All parameters have sensible defaults, but every
flag can be overridden on the command line:

| Flag | Default | Description |
| --- | --- | --- |
| `--lead N` | `6` | Predict the center this many frames ahead using EMA velocity. |
| `--deadband PX` | `10` | Ignore smaller per-axis moves to avoid micro jitter. |
| `--slew_xy PX,PY` | `40,40` | Max commanded center change per frame (pixels). |
| `--slew_z DZ` | `0.06` | Max zoom delta per frame (unitless). |
| `--padx R` | `0.20` | Extra horizontal padding around the action (fraction). |
| `--pady R` | `0.16` | Extra vertical padding around the action (fraction). |
| `--zoom_min Z` | `1.08` | Minimum zoom (crop scale denominator). |
| `--zoom_max Z` | `2.40` | Maximum zoom (larger ⇒ tighter crop). |
| `--zoom_k K` | `0.85` | Crowd-to-zoom responsiveness gain. |
| `--zoom_asym IN,OUT` | `0.75,0.35` | Let zoom-in react faster than zoom-out. |
| `--smooth_ema α` | `0.35` | EMA smoothing weight for the tracked center. |
| `--smooth_win N` | `0` | Optional odd-length boxcar on raw centers (0 disables). |
| `--hold_frames N` | `8` | Freeze the crop for N frames when confidence dips. |
| `--conf_floor C` | `0.15` | Confidence threshold for holds (0–1). |
| `--flow_thresh T` | `0.18` | Threshold after normalised optical-flow magnitude. |

Existing flags still work:

* `--roi` toggles tuned presets (`generic` or `goal`).
* `--profile` controls the aspect ratio (`portrait` → 9:16, `landscape` → 16:9).
* `--preview` writes a debug MP4 with a moving red crop rectangle.

The CSV now contains one row per frame with columns:

```
frame,cx,cy,z,w,h,x,y,conf,crowding
```

The first four columns (`frame,cx,cy,z`) are unchanged for downstream tooling.
We also write handy metadata in the header (e.g. `# fps=29.97`,
`# zoom_min=1.08,zoom_max=2.40`).

## 2. Fit FFmpeg expressions

```bash
python fit_expr.py --csv clip_zoom.csv --out clip_zoom.ps1vars --profile portrait
```

`fit_expr.py` ignores extra CSV columns automatically. If the header provided
`zoom_min` / `zoom_max`, those bounds override the YAML defaults so the cubic
polynomial stays within the measured range.

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
