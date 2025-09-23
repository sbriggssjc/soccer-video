# Autoframe Polynomial Zooms

These helpers keep the crop glued to the play instead of the center of the
frame. The flow is:

1. Track dense optical flow to produce smoothed per-frame motion centers and
   zoom factors.
2. Fit cubic FFmpeg expressions so we can feed the crop math straight into
   `ffmpeg` without thousands of literal numbers.
3. Render the reel with the existing even-dimension crop + scale chain, or
   produce a debug overlay to sanity check the box.

## 1. Track motion & emit CSV

```bash
python autoframe.py --in clip.mp4 --csv clip_zoom.csv --preview clip_debug.mp4 \
    --roi generic --profile portrait
```

* `--roi` toggles tuned presets (`generic` or `goal`).
* `--profile` controls the aspect ratio (`portrait` → 9:16, `landscape` → 16:9).
* `--preview` is optional; it writes a debug MP4 with the animated crop box.
* Tunables live in `configs/zoom.yaml` (EMA alpha, padding, dz clamp, etc.).

The CSV contains one row per frame with columns `frame,cx,cy,z` after smoothing
and clamping.

## 2. Fit FFmpeg expressions

```bash
python fit_expr.py --csv clip_zoom.csv --out clip_zoom.ps1vars --profile portrait
```

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

This keeps the crop centered on motion, adds a deadband to prevent micro-jitter,
and adaptively zooms so the action fills frame without walking off the edges.
