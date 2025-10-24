[CmdletBinding()]
param(
  [string]$Index = "out\events_selected_for_build.csv",
  [string]$Portrait = "1080x1920",
  [string]$Preset = "cinematic",
  [string]$LogsDir = "out\render_logs",
  [string]$BrandOverlay = "C:\Users\scott\soccer-video\brand\tsc\title_ribbon_1080x1920.png",
  [string]$EndCard = "C:\Users\scott\soccer-video\brand\tsc\end_card_1080x1920.png",
  [switch]$ListOnly  # just list what would run
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Index)) { throw "Index not found: $Index" }

Write-Host "Using index: $Index"

# --- Real-ESRGAN config ---
$RealESRGANExe   = "C:\\Users\\scott\\soccer-video\\tools\\realesrgan\\realesrgan-ncnn-vulkan.exe"
$UpscaleEnabled  = $true    # master switch; set $false to bypass upscaling
$UpscaleScale    = 2        # 2x is usually enough before portrait crops; 4x = heavier
$UpscaleModel    = "realesrgan-x4plus"  # solid general model
$UpscaleTmpRoot  = "C:\\Users\\scott\\soccer-video\\out\\_tmp\\upscale_frames"
$UpscaleOutRoot  = "C:\\Users\\scott\\soccer-video\\out\\upscaled"

# Make sure output dirs exist
New-Item -ItemType Directory -Force $UpscaleTmpRoot  | Out-Null
New-Item -ItemType Directory -Force $UpscaleOutRoot  | Out-Null

function Get-UpscaledPath([string]$inPath, [int]$scale) {
    $name = [IO.Path]::GetFileNameWithoutExtension($inPath)
    $ext  = ".mp4"
    return Join-Path $UpscaleOutRoot ("{0}__x{1}.mp4" -f $name, $scale)
}

function Invoke-Upscale {
    param(
        [Parameter(Mandatory=$true)][string]$In,
        [int]$Scale = $UpscaleScale,
        [string]$Model = $UpscaleModel
    )

    if (-not $UpscaleEnabled) { return $In } # passthrough

    if (-not (Test-Path $RealESRGANExe)) {
        Write-Warning "Real-ESRGAN not found; falling back to ffmpeg lanczos."
        return (Invoke-UpscaleFallback -In $In -Scale $Scale)
    }

    $Out = Get-UpscaledPath -inPath $In -scale $Scale

    # Skip if already upscaled and newer than source
    if ((Test-Path $Out) -and ((Get-Item $Out).LastWriteTime -gt (Get-Item $In).LastWriteTime)) {
        Write-Host "[UPSCALE] Skipping (already up-to-date): $Out"
        return $Out
    }

    # Try the direct-video path first (some builds support video i/o). If it fails, we fall back to frame path.
    Write-Host "[UPSCALE] Trying direct Real-ESRGAN video path..."
    $direct = & $RealESRGANExe -i $In -o $Out -n $Model -s $Scale 2>&1
    if ($LASTEXITCODE -eq 0 -and (Test-Path $Out)) {
        Write-Host "[UPSCALE] Success (direct): $Out"
        return $Out
    } else {
        Write-Warning "[UPSCALE] Direct path failed; falling back to frame extraction pipeline."
    }

    # ---- Frame pipeline (always works) ----
    $stamp    = [Guid]::NewGuid().ToString().Substring(0,8)
    $tmpInDir = Join-Path $UpscaleTmpRoot ("in_$stamp")
    $tmpOutDir= Join-Path $UpscaleTmpRoot ("out_$stamp")
    New-Item -ItemType Directory -Force $tmpInDir  | Out-Null
    New-Item -ItemType Directory -Force $tmpOutDir | Out-Null

    # 1) Probe source fps
    $fps = (& ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate `
        -of default=noprint_wrappers=1:nokey=1 -- "$In").Trim()
    if (-not $fps) { $fps = "30/1" }  # default fallback

    # 2) Extract frames losslessly
    & ffmpeg -hide_banner -y -i "$In" -map 0:v:0 -vsync 0 "$tmpInDir\%06d.png"
    if ($LASTEXITCODE -ne 0) { throw "[UPSCALE] ffmpeg extract failed." }

    # 3) ESRGAN upscale frames
    & $RealESRGANExe -i "$tmpInDir" -o "$tmpOutDir" -n $Model -s $Scale
    if ($LASTEXITCODE -ne 0) { throw "[UPSCALE] Real-ESRGAN on frames failed." }

    # 4) Re-encode video from frames; copy/encode audio from original
    #    Use same FPS; safe CRF/preset for quality
    & ffmpeg -hide_banner -y -framerate $fps -i "$tmpOutDir\%06d.png" `
        -i "$In" -map 0:v:0 -map 1:a? -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p "$Out"
    if ($LASTEXITCODE -ne 0) { throw "[UPSCALE] ffmpeg re-encode failed." }

    # 5) Clean temp
    Remove-Item -Recurse -Force $tmpInDir, $tmpOutDir

    Write-Host "[UPSCALE] Done (frames): $Out"
    return $Out
}

function Invoke-UpscaleFallback {
    param([Parameter(Mandatory=$true)][string]$In, [int]$Scale = 2)

    $Out = Get-UpscaledPath -inPath $In -scale $Scale

    if ((Test-Path $Out) -and ((Get-Item $Out).LastWriteTime -gt (Get-Item $In).LastWriteTime)) {
        Write-Host "[UPSCALE] Skipping (already up-to-date fallback): $Out"
        return $Out
    }

    # Lanczos upscale + light pre-clean; tune CRF/preset as desired
    & ffmpeg -hide_banner -y -i "$In" `
      -vf "scale=iw*${Scale}:ih*${Scale}:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0" `
      -map 0:v:0 -map 0:a? `
      -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -c:a aac -b:a 160k "$Out"

    if ($LASTEXITCODE -ne 0) { throw "[UPSCALE] ffmpeg fallback failed." }

    Write-Host "[UPSCALE] Done (fallback lanczos): $Out"
    return $Out
}

# --- Collect base IDs & explicit mp4 paths from BOTH raw text and CSV rows ---
$rxBase = [regex]'\d{3}__[^\\/",\r\n]+__t\d+(?:\.\d+)?-t\d+(?:\.\d+)?'
$rxMp4  = [regex]'(?:[A-Za-z]:)?(?:[\\/][^"''\r\n]+)+\.mp4'

$ids  = New-Object System.Collections.Generic.HashSet[string]
$mp4s = New-Object System.Collections.Generic.HashSet[string]

# a) scan raw text (handles arbitrary CSV)
$raw = Get-Content -LiteralPath $Index -Raw -Encoding UTF8
foreach ($m in $rxBase.Matches($raw)) { [void]$ids.Add($m.Value) }
foreach ($m in $rxMp4.Matches($raw))  { [void]$mp4s.Add($m.Value.Replace('/','\')) }

# b) scan parsed CSV (covers nicely-named columns like "clip","path", etc.)
try {
  $rows = Import-Csv -LiteralPath $Index
  foreach ($r in $rows) {
    foreach ($prop in $r.PSObject.Properties) {
      $val = [string]$prop.Value
      if ([string]::IsNullOrWhiteSpace($val)) { continue }
      foreach ($m in $rxBase.Matches($val)) { [void]$ids.Add($m.Value) }
      foreach ($m in $rxMp4.Matches($val))  { [void]$mp4s.Add($m.Value.Replace('/','\')) }
    }
  }
} catch {
  Write-Verbose "Import-Csv failed (non-fatal): $($_.Exception.Message)"
}

# --- Index all atomic clips on disk (skip quarantine & copies) ---
$allFiles = Get-ChildItem -Path "out\atomic_clips" -Filter *.mp4 -Recurse -File |
  Where-Object {
    $_.FullName -notmatch '\\quarantine\\' -and
    $_.Name -notmatch '(?i)(__copy|_copy)\.mp4$'
  }

$byBase = @{}
foreach ($f in $allFiles) { $byBase[$f.BaseName] = $f.FullName }

# --- Resolve targets from IDs + explicit mp4 paths ---
$resolved = New-Object System.Collections.Generic.HashSet[string]

foreach ($id in $ids) {
  if ($byBase.ContainsKey($id)) { [void]$resolved.Add($byBase[$id]) }
}

foreach ($p in $mp4s) {
  $cand = $p
  if (-not (Test-Path -LiteralPath $cand)) {
    $cand2 = Join-Path $PWD $p
    if (Test-Path -LiteralPath $cand2) { $cand = (Resolve-Path -LiteralPath $cand2).Path } else { continue }
  } else {
    $cand = (Resolve-Path -LiteralPath $cand).Path
  }
  if ($cand -notmatch '\\quarantine\\' -and $cand -notmatch '(?i)(__copy|_copy)\.mp4$') {
    [void]$resolved.Add($cand)
  }
}

$clips = @($resolved) | Sort-Object

# --- Diagnostics so you can see what's happening ---
Write-Host "Found on disk under out\atomic_clips : $($allFiles.Count) mp4s (after quarantine/copy filter)."
Write-Host "IDs parsed from index                 : $($ids.Count)"
Write-Host "Explicit mp4 paths parsed from index  : $($mp4s.Count)"
Write-Host "Resolved runnable clips               : $($clips.Count)"
if ($ids.Count -gt 0)   { Write-Host "Example ID(s): $((@($ids)[0..([Math]::Min(4,$ids.Count-1))]) -join ', ')" }
if ($clips.Count -gt 0) { Write-Host "First clip: $($clips[0])" }

if ($clips.Count -eq 0) {
  Write-Warning "Nothing matched. Quick checklist:"
  Write-Host "  • Does your CSV contain either base IDs like 022__SHOT__t3028.10-t3059.70 OR full mp4 paths?"
  Write-Host "  • Do the files exist under out\atomic_clips (and not in \quarantine\ or named *_copy.mp4)?"
  Write-Host "  • If your CSV uses a different ID pattern, send me one example row."
  throw "No eligible clips after filtering."
}

if ($ListOnly) {
  Write-Host "`n[ListOnly] Would render these $($clips.Count) clips:"
  $clips | ForEach-Object { Write-Host "  - $_" }
  return
}

# --- Make sure output dirs exist ---
New-Item -ItemType Directory -Path $LogsDir -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "out\portrait_reels\clean" -ErrorAction SilentlyContinue | Out-Null

foreach ($clip in $clips) {
  $base   = [IO.Path]::GetFileNameWithoutExtension($clip)
  $parent = Split-Path $clip -Parent

  $hi = Invoke-Upscale -In $clip -Scale $UpscaleScale

  # prefer locked path if present
  $ballPath = Join-Path $LogsDir ("{0}.ball.lock.jsonl" -f $base)
  if (-not (Test-Path $ballPath)) { $ballPath = Join-Path $LogsDir ("{0}.ball.jsonl" -f $base) }

  $tel   = Join-Path $LogsDir ("{0}.final.jsonl" -f $base)
  $log   = Join-Path $LogsDir ("{0}.final.log"  -f $base)
  $final = Join-Path "out\portrait_reels\clean" ("{0}_portrait_FINAL.mp4" -f $base)

  Write-Host "`n=== RENDER ==="
  Write-Host "IN : $clip"
  if ($hi -ne $clip) {
    Write-Host "UPS: $hi"
  }
  Write-Host "OUT: $final"

  [string[]]$extraArgs = @(
    "--telemetry", $tel,
    "--log", $log,
    "--lookahead", "24",
    "--smoothing", "0.65",
    "--zoom-min", "1.08",
    "--zoom-max", "1.45",
    "--speed-limit", "280"
  )

  if (Test-Path $ballPath) {
    $extraArgs += @("--ball-path", $ballPath, "--ball-key-x", "bx_stab", "--ball-key-y", "by_stab")
  }
  if (Test-Path $BrandOverlay) { $extraArgs += @("--brand-overlay", $BrandOverlay) }
  if (Test-Path $EndCard)      { $extraArgs += @("--endcard", $EndCard) }

  $renderParams = @{
    In        = $hi
    Out       = $final
    Preset    = $Preset
    ExtraArgs = $extraArgs
    CleanTemp = $true
  }
  if ($Portrait) {
    $renderParams["Portrait"] = $true
    $renderParams["PortraitSize"] = $Portrait
  }

  & (Join-Path $PSScriptRoot 'render_follow.ps1') @renderParams
  if ($LASTEXITCODE -ne 0) { throw "Unified renderer failed." }

  # DEBUG overlay
  $dbg = Join-Path $parent ("{0}__x2.mp4" -f $base)
  python tools\sanitize_telemetry.py --in $tel
  python -m tools.overlay_debug --in $clip --telemetry $tel --out $dbg
}

Write-Host "`nAll done."

