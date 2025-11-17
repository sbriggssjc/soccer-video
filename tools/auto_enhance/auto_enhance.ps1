<# 
Auto-enhance video to broadcast-safe Rec.709 with gentle smart defaults.
Usage:
  .\tools\auto_enhance\auto_enhance.ps1 -In "path\to\input.mp4"
  .\tools\auto_enhance\auto_enhance.ps1 -In "path\to\folder" -Recurse
Options:
  -Profile rec709_smart (default), rec709_basic, punchy
  -Crf 18..28 (default 20)
  -Preset veryfast|faster|medium... (default veryfast)
  -DryRun   (shows the ffmpeg commands without running them)
Outputs go next to source as *_ENH.mp4
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [string]$In,

  [ValidateSet("rec709_smart","rec709_basic","punchy")]
  [string]$Profile = "rec709_smart",

  [ValidateRange(10, 40)]
  [int]$Crf = 20,

  [string]$Preset = "veryfast",

  [switch]$Recurse,
  [switch]$DryRun
)

function Get-VideoFiles($root, $recurse) {
  if (Test-Path $root -PathType Leaf) {
    if ($root -match '\\.(mp4|mov|mkv|m4v|avi)$') { ,(Resolve-Path $root).Path } else { @() }
  } elseif (Test-Path $root -PathType Container) {
    $opt = @{}
    if ($recurse) { $opt.Recurse = $true }
    Get-ChildItem $root -File @opt -Include *.mp4,*.mov,*.mkv,*.m4v,*.avi | ForEach-Object { $_.FullName }
  } else {
    Write-Error "Path not found: $root"; @()
  }
}

# Build filter graph by profile.
function Get-Filter($profile) {
  switch ($profile) {
    # Best effort "smart" defaults:
    # 1) Convert to broadcast-safe range (tv) from whatever the source is.
    # 2) Light normalization (if available), mild eq for contrast/sat/gamma.
    # 3) Format to a common pixel format for maximum compatibility.
    "rec709_smart" {
      @(
        'scale=in_range=auto:out_range=tv'  # clamp to broadcast-safe levels
        # Try normalize if available; if filter missing, the call will fall back to basic profile below.
        'normalize=blackpt=0.0:whitept=1.0:smoothing=0.0'
        'eq=contrast=1.08:saturation=1.10:gamma=1.03:brightness=0.00'
        'format=yuv420p'
      ) -join ','
    }
    # Basic and very safe: no normalize, just range + eq + format
    "rec709_basic" {
      @(
        'scale=in_range=auto:out_range=tv'
        'eq=contrast=1.06:saturation=1.08:gamma=1.02:brightness=0.00'
        'format=yuv420p'
      ) -join ','
    }
    # Slightly stronger look for outdoor daylight footage
    "punchy" {
      @(
        'scale=in_range=auto:out_range=tv'
        'eq=contrast=1.14:saturation=1.15:gamma=1.02:brightness=0.00'
        'format=yuv420p'
      ) -join ','
    }
  }
}

function Invoke-FFMPEG($inPath, $filter, $crf, $preset, $dry) {
  $dir  = Split-Path $inPath -Parent
  $base = [IO.Path]::GetFileNameWithoutExtension($inPath)
  `$OutPath  = Join-Path $dir ($base + "_ENH.mp4")

  $args = @(
    '-hide_banner', '-y',
    '-i', $inPath,
    '-map', '0:v:0', '-map', '0:a:0?',   # include first audio if present
    '-vf', $filter,
    '-c:v', 'libx264', '-preset', $preset, '-crf', $crf,
    '-c:a', 'aac', '-b:a', '192k',
    `$OutPath
  )

  Write-Host "`n=== Enhancing: $inPath"
  Write-Host "ffmpeg $($args -join ' ')`n"

  if (-not $dry) {
    $p = Start-Process -FilePath 'ffmpeg' -ArgumentList $args -NoNewWindow -PassThru -Wait
    if ($p.ExitCode -ne 0) {
      # Fallback: if normalize caused failure, retry without it (basic profile).
      if ($filter -match 'normalize=') {
        Write-Warning "Normalize likely unsupported in this FFmpeg build. Retrying with rec709_basicâ€¦"
        $fallback = (Get-Filter 'rec709_basic')
        $args[$args.IndexOf('-vf')+1] = $fallback
        $p2 = Start-Process -FilePath 'ffmpeg' -ArgumentList $args -NoNewWindow -PassThru -Wait
        if ($p2.ExitCode -ne 0) { throw "FFmpeg failed on fallback for $inPath" }
      } else {
        throw "FFmpeg failed for $inPath"
      }
    }
  }

  return `$OutPath
}

$files = Get-VideoFiles -root $In -recurse:$Recurse
if (-not $files.Count) { Write-Error "No input videos found."; exit 1 }

$filter = Get-Filter $Profile
Write-Host "Profile: $Profile"
Write-Host "Filter : $filter"
Write-Host "Count  : $($files.Count)"

$outs = @()
foreach ($f in $files) {
  try {
    $o = Invoke-FFMPEG -inPath $f -filter $filter -crf $Crf -preset $Preset -dry:$DryRun
    if ($o) { $outs += $o }
  } catch {
    Write-Warning $_
  }
}

Write-Host "`nDone. Outputs:"
$outs | ForEach-Object { Write-Host "  $_" }

