# Follow-Crop & Sync Diagnostics

This walkthrough documents how to validate the portrait follow renderer end to end using only the officially supported tooling. Follow the sections in order whenever you see the camera locked to centre crop or the output MP4 drifts out of sync with its audio.

## 1. Capture the Wrapper Command

The PowerShell wrapper echoes the Python command it is about to execute. Capture that line so you can invoke it directly if you need to debug outside the wrapper.

```powershell
$Clip = "C:\Users\scott\soccer-video\out\atomic_clips\2025-10-12__TSC_SLSG_FallFestival\004__PRESSURE__t640.00-t663.00.mp4"
$Stem = [IO.Path]::GetFileNameWithoutExtension($Clip)

# capture the wrapper's echoed python command & stdout
& .\tools\render_follow.ps1 `
  -In $Clip `
  -Out "C:\Users\scott\soccer-video\out\portrait_reels\clean\$($Stem)_portrait_FINAL.mp4" `
  -Preset cinematic `
  -Portrait "1080x1920" `
  -Verbose 2>&1 | Tee-Object -Variable WrapperLog

# Show the exact Python command the wrapper echoed (it should appear in the log)
$WrapperLog | Out-String
```

If the echoed command is missing follow inputs or looks incorrect, copy that single Python line and run it manually in step 3.

## 2. Compare Input vs Output FPS

An FPS mismatch explains why the rendered video might play faster than the audio. Use `ffprobe` to check both the source clip and the most recent render.

```powershell
# Input clip fps & durations
ffprobe -v error -show_streams -select_streams v:0 -of default=nw=1:nk=1 "$Clip"
ffprobe -v error -show_streams -select_streams a:0 -of default=nw=1:nk=1 "$Clip"

# Output (last render) fps & durations
$Out = "C:\Users\scott\soccer-video\out\portrait_reels\clean\${Stem}_portrait_FINAL.mp4"
ffprobe -v error -show_streams -select_streams v:0 -of default=nw=1:nk=1 "$Out"
ffprobe -v error -show_streams -select_streams a:0 -of default=nw=1:nk=1 "$Out"
```

Note any large deviation in frame rate or duration between input and output before proceeding.

## 3. Run the Python Renderer Directly with Telemetry Enabled

Bypass the wrapper to ensure the renderer applies a follow plan and records telemetry.

```powershell
$Telem = "C:\Users\scott\soccer-video\out\render_logs\${Stem}.final.jsonl"
python tools\render_follow_unified.py `
  --in "$Clip" `
  --preset cinematic `
  --telemetry "$Telem"
```

Watch the console output for messages about plan loading, camera centres, and whether OpenCV or FFmpeg composition is active. If follow controls are missing here, the issue is within the renderer itself.

## 4. Overlay Telemetry for Visual QC

Overlay the telemetry to verify that a moving camera plan is in play.

```powershell
python tools\overlay_debug.py `
  --in "$Clip" `
  --telemetry "C:\Users\scott\soccer-video\out\render_logs\${Stem}.final.jsonl" `
  --out "C:\Users\scott\soccer-video\out\atomic_clips\2025-10-12__TSC_SLSG_FallFestival\${Stem}__x2.mp4"
```

The debug MP4 should show per-frame camera centres and zoom boxes. A static overlay means the follow plan was not applied.

## 5. Run Guardrail Checks Before a Batch

Use the guardrail utilities from the README to catch path issues, spot-check visuals, and confirm telemetry coverage before launching a wider render.

```powershell
# A) Path sanity
python tools\path_qc.py

# B) Visual sample
python tools\overlay_path_samples.py --in "$Clip" --path "out\render_logs\tester.ball.jsonl"

# C) Telemetry coverage (post-render)
python tools\inspect_telemetry.py "out\render_logs\${Stem}.final.jsonl"
```

These checks help surface missing assets, truncated telemetry, or incomplete tracking data that could lead to static crops or timing problems.
```
