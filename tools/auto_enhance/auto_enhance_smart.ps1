<#
Auto-enhance (SMART) → analyzes a short sample and adjusts EQ dynamically.
Usage:
  .\tools\auto_enhance\auto_enhance_smart.ps1 -In "path\to\input.mp4"
  .\tools\auto_enhance\auto_enhance_smart.ps1 -In "path\to\folder" -Recurse

Options:
  -SampleStart  float seconds to skip before sampling (default 1.0)
  -SampleDur    float seconds to analyze (default 8.0)
  -Crf          10..40 (default 20)
  -Preset       x264 preset (default veryfast)
  -TryWB        try white balance (default: on)
  -DryRun       show commands only

Notes:
  • Requires FFmpeg build with 'signalstats' (common). 
  • If 'whitebalance' is missing, we auto-fallback to no WB.
  • Outputs next to source as *_ENH_SMART.mp4
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [string]$In,

  [double]$SampleStart = 1.0,
  [double]$SampleDur   = 8.0,

  [ValidateRange(10,40)]
  [int]$Crf = 20,

  [string]$Preset = "veryfast",
  [switch]$Recurse,
  [switch]$TryWB = $true,
  [switch]$DryRun
)

function Get-VideoFiles($root, $recurse) {
  if (Test-Path $root -PathType Leaf) {
    if ($root -match '\.(mp4|mov|mkv|m4v|avi)$') { ,(Resolve-Path $root).Path } else { @() }
  } elseif (Test-Path $root -PathType Container) {
    $opt = @{}
    if ($recurse) { $opt.Recurse = $true }
    Get-ChildItem $root -File @opt -Include *.mp4,*.mov,*.mkv,*.m4v,*.avi | ForEach-Object { $_.FullName }
  } else {
    Write-Error "Path not found: $root"; @()
  }
}

function Clamp([double]$v, [double]$lo, [double]$hi) {
  if ($v -lt $lo) { return $lo }
  if ($v -gt $hi) { return $hi }
  return $v
}

function Get-DurationSec([string]$inPath){
  $durStr = & ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$inPath"
  if (-not $durStr) { throw "Get-DurationSec: ffprobe failed for $inPath" }
  [double]::Parse($durStr, [System.Globalization.CultureInfo]::InvariantCulture)
}

# Very portable: ask ffmpeg to detect scenes and print showinfo; parse pts_time from stderr.
function Get-Scenes-Portable([string]$inPath, [double]$thresh){
  $tmp = [IO.Path]::GetTempFileName()
  $args = @(
    '-hide_banner','-nostats','-i', $inPath,
    '-filter_complex', ("select='gt(scene,{0})',showinfo" -f ('{0:0.###}' -f $thresh)),
    '-f','null','NUL'
  )
  $stderr = & ffmpeg @args 2>&1 | Tee-Object -FilePath $tmp
  $cuts = New-Object System.Collections.Generic.List[double]
  # showinfo lines look like: "pts_time:12.345 ... n: ..."
  Select-String -Path $tmp -Pattern 'pts_time:([0-9]+\.[0-9]+|[0-9]+)' | ForEach-Object {
    $t = [double]::Parse($_.Matches[0].Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    if ($t -gt 0) { $cuts.Add($t) }
  }
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  $cuts | Sort-Object -Unique
}

# Simple fixed-chunk fallback: split every $chunkSec seconds.
function Build-FixedChunks([double]$total, [double]$chunkSec){
  if ($chunkSec -le 0) { $chunkSec = 12.0 }
  $ranges = New-Object System.Collections.Generic.List[pscustomobject]
  $t = 0.0
  while ($t -lt $total) {
    $end = [math]::Min($total, $t + $chunkSec)
    $ranges.Add([pscustomobject]@{ Start=$t; End=$end; Dur=($end-$t) })
    $t = $end
  }
  $ranges
}

# Parse a stats file produced by signalstats/metadata=print
function Parse-Stats([string]$statsPath) {
  # We’ll collect YAVG, YMIN, YMAX, SATAVG per frame, then average.
  $yavgSum = 0.0; $satSum = 0.0; $frames = 0
  $yminMin =  1.0; $ymaxMax = 0.0

  $rx = [regex]'YAVG:(?<yavg>[\d\.]+).*?YMIN:(?<ymin>[\d\.]+).*?YMAX:(?<ymax>[\d\.]+).*?SATAVG:(?<satavg>[\d\.]+)'
  Get-Content -LiteralPath $statsPath -ErrorAction SilentlyContinue | ForEach-Object {
    $m = $rx.Match($_)
    if ($m.Success) {
      $frames++
      $y = [double]$m.Groups['yavg'].Value
      $ymin = [double]$m.Groups['ymin'].Value
      $ymax = [double]$m.Groups['ymax'].Value
      $sat = [double]$m.Groups['satavg'].Value

      $yavgSum += $y
      $satSum  += $sat
      if ($ymin -lt $yminMin) { $yminMin = $ymin }
      if ($ymax -gt $ymaxMax) { $ymaxMax = $ymax }
    }
  }

  if ($frames -eq 0) {
    throw "No signalstats frames parsed. Is 'signalstats' available in your FFmpeg build?"
  }

  $yavg = $yavgSum / $frames
  $sat  = $satSum  / $frames
  $yrng = $ymaxMax - $yminMin

  [pscustomobject]@{
    YAVG = $yavg     # Average luma (0..1)
    SAT  = $sat      # Average saturation proxy (0..1)
    YMIN = $yminMin  # Lowest luma seen
    YMAX = $ymaxMax  # Highest luma seen
    YRNG = $yrng     # Luma dynamic range seen
    FRAMES = $frames
  }
}

# Compute eq parameters from stats (gentle, broadcast-safe oriented)
function Compute-EQ([double]$yavg, [double]$yrng, [double]$sat) {
  # Targets (empirical, nice-looking defaults):
  $targetY  = 0.50    # aim median-ish luminance
  $targetRg = 0.72    # aim for healthy dynamic range
  $targetSa = 0.45    # reasonable chroma for field sports

  # Contrast: push if range is low, pull if range is high.
  $contrast = 1.0 + Clamp(($targetRg - $yrng) * 0.65, -0.20, 0.20)
  $contrast = Clamp($contrast, 0.85, 1.20)

  # Gamma: lift dark or tame bright.
  # If underexposed (yavg < 0.45) → gamma up; if too bright → gamma down slightly
  $gamma = 1.0
  if ($yavg -lt 0.45)     { $gamma = 1.0 + Clamp((0.45 - $yavg) * 0.35, 0.00, 0.12) }
  elseif ($yavg -gt 0.60) { $gamma = 1.0 - Clamp(($yavg - 0.60) * 0.20, 0.00, 0.05) }
  $gamma = Clamp($gamma, 0.95, 1.12)

  # Brightness: tiny nudge toward target
  $brightness = Clamp(($targetY - $yavg) * 0.08, -0.03, 0.03)

  # Saturation: push toward target; keep gentle
  $satGain = 1.0 + Clamp(($targetSa - $sat) * 0.9, -0.15, 0.20)
  $satGain = Clamp($satGain, 0.85, 1.20)

  [pscustomobject]@{
    contrast   = [math]::Round($contrast,   3)
    gamma      = [math]::Round($gamma,      3)
    brightness = [math]::Round($brightness, 3)
    saturation = [math]::Round($satGain,    3)
  }
}

# Try to build a filter chain. Probe WB by attempting the render; if it fails, fallback.
function Build-FilterChain([pscustomobject]$eq, [bool]$wannaWB) {
  # Always clamp to broadcast-safe tv range and output a friendly pixel format.
  $parts = New-Object System.Collections.Generic.List[string]
  $parts.Add('scale=in_range=auto:out_range=tv')

  # Attempt gentle white balance first (if requested)
  if ($wannaWB) {
    # 'whitebalance' params are empirical; wb_strength ~0.05 is subtle
    $parts.Add('whitebalance=rw=0.95:gw=1.00:bw=1.05:wp=0.9')
  }

  # Dynamic eq from stats
  $parts.Add(("eq=contrast={0}:saturation={1}:gamma={2}:brightness={3}" -f `
    $eq.contrast, $eq.saturation, $eq.gamma, $eq.brightness))

  # Final format for max compatibility
  $parts.Add('format=yuv420p')

  return ($parts -join ',')
}

function Analyze-And-ComputeEQ([string]$inPath, [double]$ss, [double]$dur) {
  $tmp = [IO.Path]::GetTempFileName()
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  $tmp = [IO.Path]::ChangeExtension($tmp, '.txt')

  # Use metadata=print to write stats; discard actual output to NUL on Windows.
  $args = @(
    '-hide_banner', '-y',
    '-ss', ('{0:0.###}' -f $ss), '-t', ('{0:0.###}' -f $dur),
    '-i', $inPath,
    '-vf', 'scale=in_range=auto:out_range=tv,signalstats,metadata=print:file=' + $tmp,
    '-f', 'null', 'NUL'
  )

  Write-Host "`n[SMART] Sampling $inPath  (ss=$ss, t=$dur)"
  Write-Host "ffmpeg $($args -join ' ')`n"
  $p = Start-Process -FilePath 'ffmpeg' -ArgumentList $args -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) { throw "FFmpeg signalstats pass failed." }

  $stats = Parse-Stats $tmp
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue

  Write-Host "[SMART] Stats → YAVG=$($stats.YAVG.ToString('0.000'))  YRNG=$($stats.YRNG.ToString('0.000'))  SAT=$($stats.SAT.ToString('0.000'))  Frames=$($stats.FRAMES)"
  $eq = Compute-EQ -yavg $stats.YAVG -yrng $stats.YRNG -sat $stats.SAT
  Write-Host ("[SMART] Suggested eq → contrast={0}  saturation={1}  gamma={2}  brightness={3}" -f `
    $eq.contrast, $eq.saturation, $eq.gamma, $eq.brightness)
  return $eq
}

function Invoke-FFMPEG([string]$inPath, [string]$filter, [int]$crf, [string]$preset, [switch]$dry) {
  $dir  = Split-Path $inPath -Parent
  $base = [IO.Path]::GetFileNameWithoutExtension($inPath)
  $out  = Join-Path $dir ($base + "_ENH_SMART.mp4")

  $args = @(
    '-hide_banner', '-y',
    '-i', $inPath,
    '-map', '0:v:0', '-map', '0:a:0?',
    '-vf', $filter,
    '-c:v', 'libx264', '-preset', $preset, '-crf', $crf,
    '-c:a', 'aac', '-b:a', '192k',
    $out
  )

  Write-Host "`n=== Enhancing (SMART): $inPath"
  Write-Host "Filter: $filter"
  Write-Host "ffmpeg $($args -join ' ')`n"

  if ($dry) { return $out }

  $p = Start-Process -FilePath 'ffmpeg' -ArgumentList $args -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) {
    # If this failed and filter includes whitebalance, fallback without it.
    if ($filter -match 'whitebalance=') {
      Write-Warning "White balance likely unsupported. Retrying without WB…"
      $fallback = ($filter -replace '(^|,)whitebalance=[^,]+,?', '$1').Trim(',')
      $args[$args.IndexOf('-vf')+1] = $fallback
      $p2 = Start-Process -FilePath 'ffmpeg' -ArgumentList $args -NoNewWindow -PassThru -Wait
      if ($p2.ExitCode -ne 0) { throw "FFmpeg failed on fallback (no WB) for $inPath" }
    } else {
      throw "FFmpeg failed for $inPath"
    }
  }

  return $out
}

$files = Get-VideoFiles -root $In -recurse:$Recurse
if (-not $files.Count) { Write-Error "No input videos found."; exit 1 }

$outs = @()
foreach ($f in $files) {
  try {
    $eq = Analyze-And-ComputeEQ -inPath $f -ss $SampleStart -dur $SampleDur
    $filter = Build-FilterChain -eq $eq -wannaWB:$TryWB
    $o = Invoke-FFMPEG -inPath $f -filter $filter -crf $Crf -preset $Preset -dry:$DryRun
    if ($o) { $outs += $o }
  } catch {
    Write-Warning $_
  }
}

Write-Host "`nDone (SMART). Outputs:"
$outs | ForEach-Object { Write-Host "  $_" }
