# Unified Follow Renderer

## Quick start

```powershell
python tools\render_follow_unified.py --in "out/atomic_clips/2025-09-13__TSC_vs_NEOFC/022__SHOT__t3028.10-t3059.70.mp4" --preset cinematic --clean-temp
```

## Using the PowerShell wrapper

```powershell
.\tools\render_follow.ps1 -In ".\out\atomic_clips\2025-09-13__TSC_vs_NEOFC\022__SHOT__t3028.10-t3059.70.mp4" -Preset Cinematic -Portrait
```

The wrapper prints the resolved command before execution and exits with the same status code as the Python script.

## Calibration from tester clip

```powershell
python tools\calibrate_from_clip.py --in ".\out\atomic_clips\2025-09-13__TSC_vs_NEOFC\022__SHOT__t3028.10-t3059.70.mp4" --preset cinematic
```

This performs a lightweight random search around the preset defaults and updates `tools/render_presets.yaml` when a better fit is found. A JSON summary is written to `out/render_logs/`.

## Telemetry and debug overlay QC

```powershell
python tools\render_follow_unified.py --in "<clip>" --preset cinematic --telemetry "out/render_logs/<stem>.jsonl"
python tools\overlay_debug.py --in "<clip>" --telemetry "out/render_logs/<stem>.jsonl" --out "<clip>.__DEBUG.mp4"
```

Telemetry JSONL files include one record per rendered frame (with camera centres, zoom, and clamp flags). The debug overlay helper burns that information into a copy of the clip for quick QC sweeps.

## Pre-render QC checklist

Run these guardrails once the planner has produced a ball path but before the FFmpeg encode. They flag the most common tracking failures early so you do not waste time on full renders that will be rejected later.

```powershell
# A) Path sanity – confirm the solved ball trace stays inside the frame and moves at plausible speeds.
python tools\path_qc.py

# Expected ranges: bx/by min/max within the frame, median speed roughly 1–10 px/frame, and no huge spikes (> ~400 px/frame).

# B) Visual inspection – spot check a handful of frames with the recovered ball path overlayed.
python tools\overlay_path_samples.py --in ".\out\atomic_clips\2025-09-13__TSC_vs_NEOFC\022__SHOT__t3028.10-t3059.70.mp4" --path "out\render_logs\tester_022__SHOT.ball.jsonl"

# C) Telemetry coverage – after the render completes, make sure the telemetry has crops for the full clip.
python tools\inspect_telemetry.py "out\render_logs\tester_022__SHOT.jsonl"
```

Adjust the clip and log paths to match the shot you are validating. The first two checks should pass cleanly before you proceed to FFmpeg. After rendering, the telemetry inspector ensures every frame carries crop metadata and catches truncated exports immediately.

## Safer portrait composition

Portrait masters lean on the computed crop window for each frame. When we render with OpenCV the crop is applied directly to the decoded frame, so the portrait safe area always reflects the planner output. Recreating the same behaviour with an FFmpeg-only filtergraph is trickier because the crop expression has to change per-frame; it is easy to miss clamps and reintroduce ringing or jitter.

Until there is a dedicated FFmpeg composition path that can consume the telemetry safely, keep the default OpenCV composition. It has been reliable in production and still feeds FFmpeg for the final encode. Teams that must stay FFmpeg-only should plan to revisit a dynamic filtergraph once the expressions can be injected per frame without losing the safety checks provided by the current pipeline.
