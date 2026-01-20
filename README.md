# Soccer Video Portrait Reel Pipeline

This repository contains the **soccer-video** toolkit that we use to turn
manually curated match moments into polished 1080×1920 portrait reels. The
pipeline is purpose-built for Windows 10/11 with PowerShell and Python 3.10+,
and it relies on FFmpeg plus Real-ESRGAN for AI upscaling. The flow below starts
from hand-picked "atomic" clips and ends with fully branded reels ready for
social media.

1. Collect atomic clips under `out\atomic_clips\`.
2. Build/update the clip catalog (`atomic_index.csv`) and per-clip sidecars.
3. Run batch upscaling with Real-ESRGAN.
4. Run the portrait follow/crop/polish/branding renderer.
5. Export final reels to `out\portrait_reels\clean\`.
6. Track success, retry failures, and audit progress through the catalog files.

The sections below document every moving part, from one-time setup through
resume logic and troubleshooting.

---

## One-Time Setup

### Requirements

| Component | Notes |
|-----------|-------|
| Windows 10/11 | Run everything from Windows PowerShell or Windows Terminal. |
| Python 3.10+ | Install from the Microsoft Store or python.org. |
| FFmpeg & FFprobe | Place `ffmpeg.exe` and `ffprobe.exe` on `%PATH%`. |
| Real-ESRGAN NCNN Vulkan | Install the binary at `C:\Users\scott\soccer-video\tools\realesrgan\realesrgan-ncnn-vulkan.exe`. |

### Python Environment (optional but recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements-autoframe.txt
```

You can also install additional packages as needed via `pip install -e .` if
you plan to modify Python modules within this repository.

### FFmpeg Health Check

```powershell
ffmpeg -version
ffprobe -version
```

Both commands should succeed from any folder. If they do not, adjust your PATH
so the binaries are visible.

### Real-ESRGAN Placement

Download the NCNN Vulkan build and place it exactly at:

```
C:\Users\scott\soccer-video\tools\realesrgan\realesrgan-ncnn-vulkan.exe
```

The helper scripts call this binary via `tools\upscale.py`. The renderer falls
back to FFmpeg Lanczos scaling when Real-ESRGAN is missing, but AI upscaling is
strongly recommended for close-up sports footage.

### PowerShell Session Reset Snippet

Use this block at the top of a new PowerShell session to ensure fonts, assets,
and paths are ready:

```powershell
# --- RESET BLOCK (run inside repo root) ---
Set-Location C:\Users\scott\soccer-video
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:Path = "C:\Users\scott\soccer-video\tools\realesrgan;${env:Path}"

$global:FFmpegFonts = 'C\:\\Windows\\Fonts\\arial.ttf'
$global:FFmpegFontsBold = 'C\:\\Windows\\Fonts\\arialbd.ttf'
$global:BrandAssets = 'C:\\Users\\scott\\soccer-video\\brand\\tsc'
# -----------------------------------------
```

Set these global variables once per session if you plan to run FFmpeg
commands manually (some filters require escaped font paths).

---

## Directory Layout

```
out\atomic_clips\           # Hand-selected "atomic" highlights (input MP4s)
out\upscaled\               # AI-upscaled intermediates (e.g., __x2.mp4)
out\portrait_reels\clean\   # Final branded portrait reels (1080×1920)
out\_scratch\               # Scratch space for probes, frames, and caches (safe to delete)
out\catalog\                # Catalog + tracking artifacts (CSV + sidecars)
  atomic_index.csv          # Clip inventory and descriptive metadata
  pipeline_status.csv       # Step completion log, outputs, timestamps, errors
  sidecar\*.json            # Per-clip provenance + step history
brand\tsc\                  # Title ribbon, watermark, end card assets
tools\render_follow_unified.py
tools\upscale.py
tools\catalog.py            # Catalog + tracking helper (new)
Build-AtomicIndex.ps1       # Wrapper: scan atomic clips + update catalog
Run-UpscaleAndBrand.ps1     # Wrapper: upscale + follow/crop/polish/brand
Resume-Pipeline.ps1         # Wrapper: retry missing/failed steps
```

All generated media lives under `out\`. You can safely delete the contents of
`out\_scratch\` when you need to free disk space—the pipeline recreates scratch
files automatically. Output filenames are deterministic: cosmetic dot-tags like
`.__CINEMATIC` never repeat, preset suffixes appear once, and `_portrait_FINAL`
is only appended for final renders. Keep `out\catalog\` under version control to
audit history but avoid checking in large MP4 outputs.

### Two-Stage Portrait Follow

The follow stack is now explicitly split into two stages:

1) **Telemetry builder** – consumes a raw 1920×1080 clip and emits
   `out/telemetry/<stem>.ball.jsonl` rows shaped like::

       {"t": 3.125, "f": 75, "ball_x": 1570.3, "ball_y": 360.2, "ball_conf": 0.91,
        "carrier_x": 1505.2, "carrier_y": 520.7, "carrier_conf": 0.88,
        "source": "ball", "is_valid": true}

   Run `python tools/telemetry_builder.py --video <clip.mp4>` to generate the
   telemetry file before rendering. The builder prioritises direct ball hits,
   short carrier holds, and motion-based fallbacks; downstream consumers treat
   low-confidence or invalid rows as a signal to fall back to reactive follow.

2) **Camera follower** – the existing portrait renderer consumes telemetry (when
   available) to drive the crop center/zoom. It now rejects low-confidence
   telemetry and gracefully reverts to reactive follow if coverage or confidence
   drops below guardrails.

---

## Step 0 – Curate Atomic Clips

1. Review the match footage and manually export 16:9 moments into
   `out\atomic_clips\` (MP4 containers with H.264 video and stereo audio).
2. Use descriptive file names so later stages stay organized, for example:
   `001__SHOT__t155.50-t166.40.mp4` or `015__GOAL__t3028.10-t3038.20.mp4`.
3. Optional suffixes such as `_PK`, `_Corner`, or `_Breakaway` help with
   filtering and tagging in the catalog.

Atomic clips are the only required manual input. Everything else is scripted.

---

## Step 1 – Build or Refresh the Atomic Catalog

The catalog captures clip metadata (duration, frame size, SHA1 hash) and keeps a
per-clip JSON sidecar with provenance.

```powershell
pwsh -File .\Build-AtomicIndex.ps1
```

### Outputs

* `out\catalog\atomic_index.csv` – columns:
  `clip_path, clip_name, created_at, duration_s, width, height, fps, sha1_64, tags`
* `out\catalog\sidecar\{clip_stem}.json` – baseline record containing clip
  metadata plus empty step slots.

The PowerShell wrapper prints the number of clips scanned and how many records
changed. Rerun it any time you add, rename, or update atomic clips.

---

## Step 2 – Upscale (AI) + Step 3 – Follow/Crop/Polish/Brand

Execute both steps in one batch run. This wrapper handles caching, idempotency,
and catalog updates.

```powershell
pwsh -File .\Run-UpscaleAndBrand.ps1 -Scale 2 -Brand 'tsc' -OutDir 'C:\Users\scott\soccer-video\out\portrait_reels\clean'
```

### Under the Hood

1. Iterates over every `*.mp4` in `out\atomic_clips\`.
2. Runs `tools\upscale.py` with `--scale 2`.
   * Calls Real-ESRGAN at `tools\realesrgan\realesrgan-ncnn-vulkan.exe`.
   * Output saved to `out\upscaled\{clip_stem}__x2.mp4`.
   * Skips the step if the upscaled file is newer than the source clip.
3. Runs `tools\render_follow_unified.py` with:

   ```powershell
   python .\tools\render_follow_unified.py `
     --in "C:\Users\scott\soccer-video\out\atomic_clips\<clip>.mp4" `
     --portrait 1080x1920 --brand tsc --apply-polish `
     --upscale --upscale-scale 2 `
     --outdir "C:\Users\scott\soccer-video\out\portrait_reels\clean"
   ```

   * Applies motion-follow crops, camera polish, overlays, watermark, and end
     card using assets under `brand\tsc\`.
   * Final output saved as `out\portrait_reels\clean\{clip_stem}_portrait_FINAL.mp4`.
   * Skips rendering if the final MP4 is newer than both the atomic and
     upscaled sources.
4. Updates the catalog via `tools\catalog.py` with step timestamps, outputs,
   arguments, and errors (if any).

The wrapper continues when a clip fails (error logged in the catalog) and prints
`Batch complete.` once the queue is finished.

---

## Step 4 – Review Status & Resume Partial Runs

### Inspect Catalogs

* `out\catalog\pipeline_status.csv` – quick glance summary per clip:

  | Column | Description |
  |--------|-------------|
  | `clip_path` | Absolute path to the atomic clip. |
  | `upscale_done_at` | ISO timestamp of the last successful upscale. |
  | `upscaled_path` | Path to the upscaled intermediate. |
  | `follow_brand_done_at` | ISO timestamp of the last successful branding run. |
  | `branded_path` | Final reel location. |
  | `last_error` | Most recent error message (empty when clean). |
  | `last_run_at` | Timestamp of the last attempt (success or failure). |

* `out\catalog\sidecar\{clip}.json` – authoritative machine-readable state
  (metadata, step args, provenance, error history).

### Resume Examples

```powershell
pwsh -File .\Resume-Pipeline.ps1 -OnlyUpscale
pwsh -File .\Resume-Pipeline.ps1 -OnlyBrand
pwsh -File .\Resume-Pipeline.ps1 -Since '2025-10-09'
pwsh -File .\Resume-Pipeline.ps1 -Clip 't3028.10'
```

* `-OnlyUpscale` reruns Real-ESRGAN for clips that are missing upscaled files or
  whose source clips are newer.
* `-OnlyBrand` reruns the portrait render for clips missing branded outputs or
  with recorded errors.
* `-Since` filters entries whose `last_run_at` timestamp is on/after the
  provided date.
* `-Clip` filters rows where `clip_path` contains the provided substring.

The resume wrapper reads `pipeline_status.csv` (plus sidecars) to determine
which steps are stale or errored, then reruns just those steps while updating
catalog entries.

---

## Per-Clip Advanced Run

When you need to experiment with parameters or debug a single clip, run the
renderer directly:

```powershell
python .\tools\render_follow_unified.py `
  --in "C:\Users\scott\soccer-video\out\atomic_clips\001__SHOT__t155.50-t166.40.mp4" `
  --portrait 1080x1920 --brand tsc --apply-polish `
  --upscale --upscale-scale 2 `
  --outdir "C:\Users\scott\soccer-video\out\portrait_reels\clean"
```

Optionally add `--force` to re-render even if outputs already exist. After a
manual run, call `tools\catalog.py` to sync status:

```powershell
python .\tools\catalog.py --mark-upscaled --clip <clip> --out <upscaled> --scale 2 --model realesrgan-x4plus
python .\tools\catalog.py --mark-branded --clip <clip> --out <final> --brand tsc --args --portrait 1080x1920 --brand tsc
```

---

## Catalog & Tracking System

`tools\catalog.py` maintains two CSV summaries plus JSON sidecars.

### CSV 1 – `atomic_index.csv`

| Column | Description |
|--------|-------------|
| `clip_path` | Absolute path to the atomic MP4. |
| `clip_name` | File name only (useful for quick filtering). |
| `created_at` | File creation time (UTC ISO 8601). |
| `duration_s` | Duration from `ffprobe`, rounded to milliseconds. |
| `width` / `height` | Source resolution (pixels). |
| `fps` | Raw frame rate string from `ffprobe` (e.g., `60/1`). |
| `sha1_64` | First 64 bits of the file SHA1 hash (short audit ID). |
| `tags` | Free-form string for manual annotations. |

Run `Build-AtomicIndex.ps1` whenever atomic clips change to update this table.

### CSV 2 – `pipeline_status.csv`

Each processing attempt updates (or creates) a row keyed by `clip_path`. The
wrapper scripts call `tools\catalog.py --mark-upscaled` and
`--mark-branded` to upsert timestamps, output locations, and error notes. The
file can be opened in Excel for quick audits or filtered with PowerShell.

### Sidecar JSON – `out\catalog\sidecar\{clip_stem}.json`

Example:

```json
{
  "clip_path": "C:\\Users\\scott\\soccer-video\\out\\atomic_clips\\001__SHOT__t155.50-t166.40.mp4",
  "source_sha1_64": "abc1234567890def",
  "meta": {
    "duration_s": 10.9,
    "fps": "60/1",
    "size": [1920, 1080]
  },
  "steps": {
    "upscale": {
      "done": true,
      "at": "2025-10-10T12:34:56Z",
      "out": "C:\\Users\\scott\\soccer-video\\out\\upscaled\\001__SHOT__t155.50-t166.40__x2.mp4",
      "model": "realesrgan-x4plus",
      "scale": 2
    },
    "follow_crop_brand": {
      "done": true,
      "at": "2025-10-10T12:36:20Z",
      "out": "C:\\Users\\scott\\soccer-video\\out\\portrait_reels\\clean\\001__SHOT__t155.50-t166.40_portrait_FINAL.mp4",
      "brand": "tsc",
      "args": ["--portrait", "1080x1920", "--brand", "tsc", "--apply-polish", "--upscale", "--upscale-scale", "2", "--outdir", "C:\\Users\\scott\\soccer-video\\out\\portrait_reels\\clean"]
    }
  },
  "errors": []
}
```

When an error occurs, the script appends an entry to the `errors` array with the
step name, timestamp, and message for forensic tracking. The sidecar is the
authoritative machine-readable record; the CSVs are concise summaries for quick
review.

---

## Caching & Skip Logic

* **Upscale step** – skipped when an existing `__x{scale}.mp4` is newer than the
  atomic source. Delete the upscaled file to force regeneration.
* **Branding step** – skipped when the final `_portrait_FINAL.mp4` is newer than
  both the atomic clip and the upscaled intermediate. Delete the final file to
  force re-rendering.
* **CLI overrides** – add `--force` (renderer) or `--overwrite` (FFmpeg tooling)
  to bypass caching when experimenting.
* **Resume scripts** – automatically detect stale timestamps, missing outputs,
  and recorded errors to rerun only what is necessary.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ffprobe` not found | Confirm FFmpeg binaries are on PATH. Reopen PowerShell after edits. |
| Real-ESRGAN error | Ensure the binary path matches the documented location and that your GPU supports Vulkan. The pipeline falls back to FFmpeg Lanczos scaling, but quality is lower. |
| Font errors (`Could not load font file`) | Double-escape font paths (`C\:\\Windows\\Fonts\\arialbd.ttf`) or use the RESET BLOCK variables. |
| Missing brand assets | Confirm `brand\tsc\title_ribbon_1080x1920.png`, `watermark_corner_256.png`, and `end_card_1080x1920.png` exist. |
| `pipeline_status.csv` shows stale errors | Run `Resume-Pipeline.ps1` or manually call `tools\catalog.py` with `--mark-branded`/`--mark-upscaled` after verifying outputs. |
| Renderer slow | Adjust FFmpeg preset via command line (e.g., `--encode-preset medium`). Larger upscale factors (4×) produce better detail but increase render time. |
| Portrait follow locked or output video runs faster than audio | Follow the step-by-step checks in [`docs/DIAGNOSE_FOLLOW_SYNC.md`](docs/DIAGNOSE_FOLLOW_SYNC.md) to capture the wrapper command, compare FPS, run the renderer with telemetry, and inspect overlays. |

---

## Performance & Quality Tips for Sports Footage

* **Upscale before crop** – The pipeline already upscales before portrait crops;
  keep this order for crisp player details.
* **Scale choice** – Use 2× for most reels. Reserve 4× for hero clips where the
  source is extremely wide or soft.
* **Encoding** – Stick to CRF ~18 and preset `slow` for best quality/size
  balance. If throughput is more important, drop to `medium`.
* **Batch size** – Process clips in batches of 10–15 to keep scratch storage and
  GPU load manageable.
* **Tags** – Add keywords in `atomic_index.csv` (last column) to group plays by
  theme, athlete, or scenario.

---

## FAQ

**Where are logs stored?**
`tools\render_follow_unified.py` prints progress to stdout/stderr. Capture output
with `Tee-Object` if you need persistent logs. Errors also land in
`pipeline_status.csv` and the sidecar `errors` array.

**How do I add custom tags?**
Open `out\catalog\atomic_index.csv` in Excel or VS Code and populate the `tags`
column. Re-running `Build-AtomicIndex.ps1` preserves manual tags.

**Can I add a new brand theme?**
Drop assets under `brand\<name>\`, update `render_follow_unified.py` to
recognize the theme, then call `Run-UpscaleAndBrand.ps1 -Brand '<name>'`. The
catalog will record the brand name in the sidecars.

**How do I clear errors after fixing an issue?**
Rerun the failing step (either via `Resume-Pipeline.ps1` or manually) and the
catalog helper resets `last_error` once the step succeeds.

---

## Git Hygiene

* Ignore large media outputs in `.gitignore` (patterns such as `out/*.mp4`,
  `out/_tmp/`, `runs/`).
* Commit catalog updates (`atomic_index.csv`, `pipeline_status.csv`, sidecars)
  when they capture meaningful production state.
* Suggested commit message template for pipeline tweaks:
  `feat(pipeline): describe the automation change`

---

## Acceptance Checklist

After a successful batch run you should see:

* `out\upscaled\{clip_stem}__x2.mp4`
* `out\portrait_reels\clean\{clip_stem}_portrait_FINAL.mp4`
* `out\catalog\atomic_index.csv`
* `out\catalog\pipeline_status.csv`
* `out\catalog\sidecar\{clip_stem}.json`

Use the resume script whenever the process is interrupted. The catalog +
sidecars provide both human-readable and machine-friendly tracking so nothing
falls through the cracks.
