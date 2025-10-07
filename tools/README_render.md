# Unified Render Follow Pipeline

The legacy `render_follow_*` scripts have been consolidated under a single,
argument-driven workflow.  Use the PowerShell wrapper for day-to-day tasks on
Windows, or invoke the Python CLI directly for scripting / automation.

## PowerShell wrapper

```powershell
pwsh -File tools\render_follow.ps1 -In ".\out\atomic_clips\2025-09-13__TSC_vs_NEOFC\022__SHOT__t3028.10-t3059.70.mp4" -Preset Cinematic -Portrait -CleanTemp
```

* `-Preset` selects the render flavour (`Cinematic`, `Gentle`, `RealZoom`).
* `-Portrait` enables a 1080x1920 canvas by default.  To specify a custom size
  use `-PortraitSize "720x1280"`.
* `-Fps`, `-Flip180`, and `-ExtraArgs` are passed through to the Python
  implementation.
* Add `-Legacy` to force the fallback cinematic renderer.

## Python CLI

```bash
python tools/render_follow_unified.py \
  --in "./out/atomic_clips/2025-09-13__TSC_vs_NEOFC/022__SHOT__t3028.10-t3059.70.mp4" \
  --preset realzoom \
  --portrait 1080x1920 \
  --fps 30 \
  --clean-temp \
  --labels-root "out/yolo"
```

Key options:

* `--in` / `--src`: input clip (MP4).  Both flags are accepted for compatibility.
* `--out`: override the default `.<stem>.__<PRESET>.mp4` output name.
* `--preset`: cinematic, gentle, or realzoom.  Presets tune camera smoothing,
  look-ahead, and padding.
* `--portrait`: target canvas size (`WIDTHxHEIGHT`).  Portrait mode keeps the
  ball roughly 60% up from the bottom while clamping zoom to avoid artifacts.
* `--labels-root`: search path for YOLO ball labels.  Multiple shards under
  `out/yolo/*/labels/<clip_stem>_*.txt` are merged automatically.
* `--brand-overlay` / `--endcard`: apply branding or append a 1s endcard.
* `--clean-temp`: wipe the working folder (`out/autoframe_work/<preset>/<clip>`) before rendering.
* `--log`: override the log location (defaults to `out/render_logs/`).

## Work folders & cleanup

* Frame caches live under `out/autoframe_work/<preset>/<clip_stem>/frames/`.
  Re-runs reuse cached frames unless `--clean-temp` (CLI) or `-CleanTemp`
  (PowerShell) is supplied.
* Log files are written to `out/render_logs/` alongside a JSON summary.
* Temporary concat manifests (`*.ffconcat`) are disposable and ignored by git.

## Troubleshooting

* **Missing labels** – The renderer falls back to a centre-biased eased camera
  path if no YOLO detections are found.  Verify the input stem matches the label
  file naming convention.
* **Audio missing** – Ensure ffmpeg can read the source audio stream.  The
  stitch step maps `-map 1:a:0?` so silent clips are handled gracefully.
* **Cleaning caches** – Delete the folder under `out/autoframe_work/<preset>/` or
  re-run with the clean switches mentioned above.
