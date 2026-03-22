# ═══════════════════════════════════════════════════════════
# TSC Season Burst Montage - Build Script
# Generated: 2026-03-03T22:35:34
# ═══════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

$outW = 1080
$outH = 1920
$crf = 18

# Portrait reel root (preferred source - polished 1080x1920)
$portraitRoot = "D:\Projects\soccer-video\out\portrait_reels"

# Brand assets
$watermarkPNG = "D:\Projects\soccer-video\brand\tsc\watermark_corner_256_transparent.png"
$endCardPNG = "D:\Projects\soccer-video\brand\tsc\end_card_1080x1920.png"
$fontPath = "D:\Projects\soccer-video\fonts\Montserrat-ExtraBold.ttf"
$fontPathSemi = "D:\Projects\soccer-video\fonts\Montserrat-SemiBold.ttf"
$hasBrandAssets = (Test-Path $watermarkPNG) -and (Test-Path $fontPath)
if (-not $hasBrandAssets) {
  Write-Warning "Brand assets missing - watermark and labels will be skipped"
  Write-Warning "  Watermark: $watermarkPNG"
  Write-Warning "  Font: $fontPath"
}

function Brand-BurstClip {
  param(
    [string]$SrcClip,
    [string]$OutFile,
    [double]$Start,
    [double]$Duration,
    [double]$Fps,
    [string]$Label,
    [bool]$IsPortrait
  )
  # Convert paths to FFmpeg format (forward slashes, escaped drive letter)
  $ffFont = ($fontPath -replace '\\', '/') -replace '^([A-Za-z]):','$1\\:'
  $ffWM = ($watermarkPNG -replace '\\', '/') -replace '^([A-Za-z]):','$1\\:'

  # Escape label for FFmpeg drawtext
  $esc = $Label -replace ':','\\:' -replace "'","\\'" -replace '%','%%'

  if ($IsPortrait) {
    $scaleF = '[0:v]copy[base]'
  } else {
    $scaleF = '[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1[base]'
  }

  # Filter graph: scale -> watermark -> action label (single-quoted to protect [brackets])
  $filterG = '{0};[base][1:v]overlay=W-w-32:32:format=auto:alpha=0.65[wm];[wm]drawtext=fontfile=''{1}'':text=''{2}'':fontsize=56:fontcolor=0xB7A37C:borderw=3:bordercolor=0x1F2B3D@0.8:x=(w-text_w)/2:y=h-220:enable=''between(t\,0\,1.5)''[out]' -f $scaleF, $ffFont, $esc

  ffmpeg -hide_banner -loglevel warning -y `
    -ss $Start -t $Duration -i $SrcClip `
    -i $watermarkPNG `
    -filter_complex $filterG `
    -map '[out]' `
    -r $Fps -c:v libx264 -crf $crf -preset fast -an `
    $OutFile
}

function Brand-BurstClipSimple {
  # Fallback when brand assets are missing - just trim, no overlays
  param(
    [string]$SrcClip,
    [string]$OutFile,
    [double]$Start,
    [double]$Duration,
    [double]$Fps,
    [bool]$IsPortrait
  )
  if ($IsPortrait) {
    ffmpeg -hide_banner -loglevel warning -y `
      -ss $Start -t $Duration -i $SrcClip `
      -r $Fps -c:v libx264 -crf $crf -preset fast -an `
      $OutFile
  } else {
    ffmpeg -hide_banner -loglevel warning -y `
      -ss $Start -t $Duration -i $SrcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r $Fps -c:v libx264 -crf $crf -preset fast -an `
      $OutFile
  }
}

# Working directory for extracted bursts
$burstDir = Join-Path $PSScriptRoot "burst_clips"
if (!(Test-Path $burstDir)) { New-Item -ItemType Directory -Force $burstDir | Out-Null }

$concatEntries = @()
$extractCount = 0
$skipCount = 0

# ─── September 20, 2025: TSC vs Greenwood (39 bursts) ───

# ─── Generate branded intro card ───
$introCard = Join-Path $burstDir "intro_2025-09-20__TSC_vs_Greenwood.mp4"
$introFont = ($fontPath -replace '\\\\', '/') -replace '^([A-Za-z]):','$1\\:'
$introFilters = "drawbox=x=0:y=0:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=0:y=1914:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=80:y=780:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawbox=x=80:y=1140:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawtext=text='TULSA SOCCER CLUB':fontsize=32:fontcolor=0xB7A37C@0.85:x=(w-text_w)/2:y=200,drawtext=text='TSC vs Greenwood':fontsize=72:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=(h-text_h)/2-80,drawtext=text='September 20, 2025':fontsize=40:fontcolor=0xB7A37C@0.9:x=(w-text_w)/2:y=(h)/2+10"
$introFilters = $introFilters -replace 'drawtext=','drawtext=fontfile=''$introFont'':'
if (Test-Path $fontPath) {
  ffmpeg -hide_banner -loglevel warning -y -f lavfi -i "color=c=0x1F2B3D:s=1080x1920:r=30:d=3.0" `
    -vf $introFilters `
    -c:v libx264 -crf $crf -preset fast -pix_fmt yuv420p -an `
    $introCard
} else {
  # No font - plain navy card
  ffmpeg -hide_banner -loglevel warning -y -f lavfi -i "color=c=0x1F2B3D:s=1080x1920:r=30:d=3.0" `
    -c:v libx264 -crf $crf -preset fast -pix_fmt yuv420p -an `
    $introCard
}
if (Test-Path $introCard) {
  $concatEntries += "file '$introCard'"
  Write-Host "  Intro card generated"
}

$slateFile = Join-Path $PSScriptRoot "slates\2025-09-20__TSC_vs_Greenwood__slate.mp4"
if (Test-Path $slateFile) { $concatEntries += "file '$slateFile'" }

# Clip 002: GOAL (score=9.0, burst=5.0s @ 7.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__002__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '002__2025-09-20__TSC_vs_Greenwood__GOAL__t20400.00-t21180.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [002] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\002__2025-09-20__TSC_vs_Greenwood__GOAL__t20400.00-t21180.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.52 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.52 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 011: GOAL (score=9.0, burst=5.0s @ 6.0s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__011__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '011__2025-09-20__TSC_vs_Greenwood__GOAL__t54120.00-t54780.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [011] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\011__2025-09-20__TSC_vs_Greenwood__GOAL__t54120.00-t54780.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.99 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.99 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 022: GOAL (score=9.0, burst=5.0s @ 3.1s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__022__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '022__2025-09-20__TSC_vs_Greenwood__GOAL__t1578.00-t1586.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [022] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\022__2025-09-20__TSC_vs_Greenwood__GOAL__t1578.00-t1586.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.13 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.13 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 025: GOAL (score=9.0, burst=5.0s @ 4.8s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__025__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '025__2025-09-20__TSC_vs_Greenwood__GOAL__t1885.00-t1895.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [025] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\025__2025-09-20__TSC_vs_Greenwood__GOAL__t1885.00-t1895.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.76 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.76 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 033: GOAL (score=7.5, burst=5.0s @ 12.6s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__033__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '033__2025-09-20__TSC_vs_Greenwood__GOAL__t2562.00-t2582.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [033] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\033__2025-09-20__TSC_vs_Greenwood__GOAL__t2562.00-t2582.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.59 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.59 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 037: PRESSURE AND GOAL (score=7.5, burst=5.0s @ 12.6s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__037__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '037__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_GOAL__t2729.00-t2749.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [037] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\037__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_GOAL__t2729.00-t2749.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.59 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.59 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 004: PRESSURE TWO SHOTS (score=5.8, burst=5.0s @ 6.8s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__004__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '004__2025-09-20__TSC_vs_Greenwood__PRESSURE_TWO_SHOTS__t26640.00-t27360.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [004] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\004__2025-09-20__TSC_vs_Greenwood__PRESSURE_TWO_SHOTS__t26640.00-t27360.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.77 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.77 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 007: DRIBBLING SHOT (score=5.8, burst=5.0s @ 7.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__007__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '007__2025-09-20__TSC_vs_Greenwood__DRIBBLING_SHOT__t44340.00-t45120.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [007] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\007__2025-09-20__TSC_vs_Greenwood__DRIBBLING_SHOT__t44340.00-t45120.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.46 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.46 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 009: PRESSURE SHOT (score=5.8, burst=5.0s @ 9.6s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__009__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '009__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t48120.00-t49080.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [009] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\009__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t48120.00-t49080.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.62 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.62 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 012: BUILD SHOT (score=5.8, burst=5.0s @ 8.2s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__012__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '012__2025-09-20__TSC_vs_Greenwood__BUILD_SHOT__t58380.00-t59220.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [012] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\012__2025-09-20__TSC_vs_Greenwood__BUILD_SHOT__t58380.00-t59220.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.15 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.15 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 013: PRESSURE SHOT (score=5.8, burst=5.0s @ 7.4s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__013__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '013__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t59280.00-t60060.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [013] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\013__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t59280.00-t60060.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.43 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.43 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 023: BUILD UP PLAY AND SHOT (score=5.8, burst=5.0s @ 5.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__023__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '023__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t1711.00-t1722.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [023] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\023__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t1711.00-t1722.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.51 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.51 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 024: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 4.8s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__024__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '024__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t1871.00-t1881.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [024] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\024__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t1871.00-t1881.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.76 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.76 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 026: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 5.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__026__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '026__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t1951.00-t1962.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [026] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\026__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t1951.00-t1962.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.48 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.48 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 027: BUILD UP PLAY AND SHOT (score=5.8, burst=5.0s @ 10.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__027__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '027__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2040.00-t2058.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [027] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\027__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2040.00-t2058.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.49 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.49 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 028: BUILD UP PLAY AND SHOT (score=5.8, burst=5.0s @ 6.9s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__028__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '028__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2092.00-t2105.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [028] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\028__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2092.00-t2105.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.89 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.89 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 029: BUILD UP PLAY AND SHOT (score=5.8, burst=5.0s @ 10.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__029__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '029__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2134.00-t2152.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [029] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\029__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2134.00-t2152.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.49 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.49 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 032: BUILD UP PLAY AND SHOT (score=5.8, burst=5.0s @ 10.4s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__032__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '032__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2544.00-t2561.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [032] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\032__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2544.00-t2561.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.43 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.43 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 034: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 11.9s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__034__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '034__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t2646.00-t2665.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [034] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\034__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t2646.00-t2665.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.87 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.87 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 036: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 5.4s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__036__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '036__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t2705.00-t2715.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [036] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\036__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t2705.00-t2715.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.39 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.39 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 030: FREE KICK WON AND GOAL (score=5.2, burst=5.0s @ 29.2s, fps=23.976) ⚠ END-BIASED (clip=36s, review recommended)
Write-Warning "Clip 030 `(FREE KICK WON AND GOAL`) is 36s -- end-biased burst at 29.2-34.2s may need manual review"
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__030__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '030__2025-09-20__TSC_vs_Greenwood__FREE_KICK_WON_AND_GOAL__t2175.00-t2211.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [030] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\030__2025-09-20__TSC_vs_Greenwood__FREE_KICK_WON_AND_GOAL__t2175.00-t2211.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 29.22 -Duration 5.00 `
      -Fps 23.976 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 29.22 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 001: BUILD UP PLAY AND SHOT (score=4.8, burst=5.0s @ 15.4s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__001__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '001__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t315.00-t339.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [001] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\001__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t315.00-t339.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 15.44 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 15.44 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 003: BUILD UP PLAY SHOT (score=4.8, burst=5.0s @ 12.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__003__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '003__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_SHOT__t25260.00-t26460.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [003] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\003__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_SHOT__t25260.00-t26460.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.53 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.53 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 006: BUILD UP PLAY SHOT (score=4.8, burst=5.0s @ 14.7s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__006__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '006__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_SHOT__t41100.00-t42480.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [006] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\006__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_SHOT__t41100.00-t42480.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.66 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.66 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 008: PRESSURE SHOT (score=4.8, burst=5.0s @ 19.0s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__008__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '008__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t45240.00-t46980.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [008] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\008__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t45240.00-t46980.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 18.98 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 18.98 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 031: BUILD UP PLAY AND SHOT (score=4.8, burst=5.0s @ 12.6s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__031__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '031__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2245.00-t2266.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [031] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\031__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2245.00-t2266.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.62 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.62 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 039: BUILD UP PLAY AND SHOT (score=4.8, burst=5.0s @ 12.6s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__039__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '039__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2894.00-t2914.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [039] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\039__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t2894.00-t2914.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.56 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.56 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 005: WIN FREE KICK (score=3.6, burst=3.5s @ 9.7s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__005__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '005__2025-09-20__TSC_vs_Greenwood__WIN_FREE_KICK__t32280.00-t33180.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [005] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\005__2025-09-20__TSC_vs_Greenwood__WIN_FREE_KICK__t32280.00-t33180.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.68 -Duration 3.50 `
      -Fps 24.000 -Label "FREE KICK" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.68 -Duration 3.50 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 035: CORNER (score=3.6, burst=4.0s @ 8.8s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__035__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '035__2025-09-20__TSC_vs_Greenwood__CORNER__t2687.00-t2701.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [035] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\035__2025-09-20__TSC_vs_Greenwood__CORNER__t2687.00-t2701.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.77 -Duration 4.00 `
      -Fps 23.976 -Label "CORNER" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.77 -Duration 4.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 021: BUILD UP PLAY AND SHOT (score=3.4, burst=5.0s @ 28.9s, fps=23.976) ⚠ END-BIASED (clip=36s, review recommended)
Write-Warning "Clip 021 `(BUILD UP PLAY AND SHOT`) is 36s -- end-biased burst at 28.9-33.9s may need manual review"
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__021__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '021__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t1437.00-t1472.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [021] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\021__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_AND_SHOT__t1437.00-t1472.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 28.93 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 28.93 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 038: PRESSURE AND SHOT (score=3.4, burst=5.0s @ 48.5s, fps=24.000) ⚠ END-BIASED (clip=58s, review recommended)
Write-Warning "Clip 038 `(PRESSURE AND SHOT`) is 58s -- end-biased burst at 48.5-53.5s may need manual review"
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__038__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '038__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t2770.00-t2827.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [038] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\038__2025-09-20__TSC_vs_Greenwood__PRESSURE_AND_SHOT__t2770.00-t2827.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 48.47 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 48.47 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 010: DEFENSE BUILD UP (score=2.0, burst=3.5s @ 13.6s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__010__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '010__2025-09-20__TSC_vs_Greenwood__DEFENSE_BUILD_UP__t49260.00-t50880.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [010] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\010__2025-09-20__TSC_vs_Greenwood__DEFENSE_BUILD_UP__t49260.00-t50880.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 13.56 -Duration 3.50 `
      -Fps 24.000 -Label "BUILD" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 13.56 -Duration 3.50 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 014: t1063.00-t1073.00 portrait POST (score=0.0, burst=4.0s @ 4.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__014__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '014__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t1063.00-t1073.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [014] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\014__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t1063.00-t1073.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.48 -Duration 4.00 `
      -Fps 24.000 -Label "T1063.00-T1073.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.48 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 015: t1081.00-t1088.00 portrait POST (score=0.0, burst=4.0s @ 2.7s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__015__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '015__2025-09-20__TSC_vs_Greenwood__SHOT__t1081.00-t1088.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [015] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\015__2025-09-20__TSC_vs_Greenwood__SHOT__t1081.00-t1088.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.68 -Duration 4.00 `
      -Fps 24.000 -Label "T1081.00-T1088.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.68 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 016: t1195.00-t1209.00 portrait POST (score=0.0, burst=4.0s @ 6.8s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__016__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '016__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_SHOT__t1195.00-t1209.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [016] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\016__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_SHOT__t1195.00-t1209.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.85 -Duration 4.00 `
      -Fps 24.000 -Label "T1195.00-T1209.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.85 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 017: t1212.00-t1224.00 portrait POST (score=0.0, burst=4.0s @ 5.7s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__017__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '017__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t1212.00-t1224.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [017] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\017__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t1212.00-t1224.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.65 -Duration 4.00 `
      -Fps 24.000 -Label "T1212.00-T1224.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.65 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 018: t1292.00-t1329.00 portrait POST (score=0.0, burst=4.0s @ 31.2s, fps=24.000) ⚠ END-BIASED (clip=38s, review recommended)
Write-Warning "Clip 018 `(t1292.00-t1329.00 portrait POST`) is 38s -- end-biased burst at 31.2-35.2s may need manual review"
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__018__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '018__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_GOAL__t1292.00-t1329.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [018] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\018__2025-09-20__TSC_vs_Greenwood__BUILD_UP_PLAY_GOAL__t1292.00-t1329.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 31.23 -Duration 4.00 `
      -Fps 24.000 -Label "T1292.00-T1329.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 31.23 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 019: t1369.00-t1383.00 portrait POST (score=0.0, burst=4.0s @ 6.8s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__019__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '019__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t1369.00-t1383.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [019] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\019__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t1369.00-t1383.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.83 -Duration 4.00 `
      -Fps 24.000 -Label "T1369.00-T1383.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.83 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 020: t1393.00-t1409.00 portrait POST (score=0.0, burst=4.0s @ 8.0s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_Greenwood__020__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '020__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t1393.00-t1409.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [020] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_Greenwood\020__2025-09-20__TSC_vs_Greenwood__PRESSURE_SHOT__t1393.00-t1409.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.03 -Duration 4.00 `
      -Fps 23.976 -Label "T1393.00-T1409.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.03 -Duration 4.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# ─── September 20, 2025: TSC vs RVFC (44 bursts) ───

# ─── Generate branded intro card ───
$introCard = Join-Path $burstDir "intro_2025-09-20__TSC_vs_RVFC.mp4"
$introFont = ($fontPath -replace '\\\\', '/') -replace '^([A-Za-z]):','$1\\:'
$introFilters = "drawbox=x=0:y=0:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=0:y=1914:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=80:y=780:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawbox=x=80:y=1140:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawtext=text='TULSA SOCCER CLUB':fontsize=32:fontcolor=0xB7A37C@0.85:x=(w-text_w)/2:y=200,drawtext=text='TSC vs RVFC':fontsize=72:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=(h-text_h)/2-80,drawtext=text='September 20, 2025':fontsize=40:fontcolor=0xB7A37C@0.9:x=(w-text_w)/2:y=(h)/2+10"
$introFilters = $introFilters -replace 'drawtext=','drawtext=fontfile=''$introFont'':'
if (Test-Path $fontPath) {
  ffmpeg -hide_banner -loglevel warning -y -f lavfi -i "color=c=0x1F2B3D:s=1080x1920:r=30:d=3.0" `
    -vf $introFilters `
    -c:v libx264 -crf $crf -preset fast -pix_fmt yuv420p -an `
    $introCard
} else {
  # No font - plain navy card
  ffmpeg -hide_banner -loglevel warning -y -f lavfi -i "color=c=0x1F2B3D:s=1080x1920:r=30:d=3.0" `
    -c:v libx264 -crf $crf -preset fast -pix_fmt yuv420p -an `
    $introCard
}
if (Test-Path $introCard) {
  $concatEntries += "file '$introCard'"
  Write-Host "  Intro card generated"
}

$slateFile = Join-Path $PSScriptRoot "slates\2025-09-20__TSC_vs_RVFC__slate.mp4"
if (Test-Path $slateFile) { $concatEntries += "file '$slateFile'" }

# Clip 004: PRESSURE GOAL (score=9.0, burst=5.0s @ 8.3s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__004__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '004__2025-09-20__TSC_vs_RVFC__PRESSURE_GOAL__t17880.00-t18720.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [004] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\004__2025-09-20__TSC_vs_RVFC__PRESSURE_GOAL__t17880.00-t18720.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.27 -Duration 5.00 `
      -Fps 23.976 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.27 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 007: PRESSURE GOAL (score=9.0, burst=5.0s @ 11.1s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__007__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '007__2025-09-20__TSC_vs_RVFC__PRESSURE_GOAL__t31740.00-t32820.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [007] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\007__2025-09-20__TSC_vs_RVFC__PRESSURE_GOAL__t31740.00-t32820.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.12 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.12 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 025: PRESSURE AND GOAL (score=9.0, burst=5.0s @ 3.8s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__025__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '025__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t1512.00-t1520.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [025] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\025__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t1512.00-t1520.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.75 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.75 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 034: PRESSURE AND GOAL (score=9.0, burst=5.0s @ 5.7s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__034__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '034__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t2158.00-t2169.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [034] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\034__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t2158.00-t2169.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.66 -Duration 5.00 `
      -Fps 23.976 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.66 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 036: PRESSURE AND GOAL (score=9.0, burst=5.0s @ 11.4s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__036__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '036__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t2348.00-t2367.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [036] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\036__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t2348.00-t2367.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.39 -Duration 5.00 `
      -Fps 23.976 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.39 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 043: GOAL (score=9.0, burst=5.0s @ 10.6s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__043__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '043__2025-09-20__TSC_vs_RVFC__GOAL__t3010.00-t3028.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [043] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\043__2025-09-20__TSC_vs_RVFC__GOAL__t3010.00-t3028.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.61 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.61 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 044: GOAL (score=9.0, burst=5.0s @ 8.4s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__044__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '044__2025-09-20__TSC_vs_RVFC__GOAL__t3051.00-t3066.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [044] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\044__2025-09-20__TSC_vs_RVFC__GOAL__t3051.00-t3066.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.45 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.45 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 038: PRESSURE AND GOAL (score=7.5, burst=5.0s @ 12.1s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__038__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '038__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t2411.00-t2431.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [038] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\038__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t2411.00-t2431.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.11 -Duration 5.00 `
      -Fps 24.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.11 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 039: PRESSURE AND GOAL (score=7.5, burst=5.0s @ 12.8s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__039__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '039__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t2485.00-t2506.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [039] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\039__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_GOAL__t2485.00-t2506.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.83 -Duration 5.00 `
      -Fps 23.976 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.83 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 001: SHOT (score=5.8, burst=5.0s @ 4.7s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__001__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '001__2025-09-20__TSC_vs_RVFC__SHOT__t122.00-t131.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [001] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\001__2025-09-20__TSC_vs_RVFC__SHOT__t122.00-t131.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.67 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.67 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 002: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 9.7s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__002__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '002__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t132.00-t148.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [002] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\002__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t132.00-t148.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.71 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.71 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 003: PRESSURE SHOT (score=5.8, burst=5.0s @ 7.5s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__003__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '003__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t15480.00-t16260.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [003] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\003__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t15480.00-t16260.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.55 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.55 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 005: BUILD UP PLAY SHOT (score=5.8, burst=5.0s @ 10.4s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__005__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '005__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t21000.00-t22020.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [005] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\005__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t21000.00-t22020.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.40 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.40 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 006: SHOT (score=5.8, burst=5.0s @ 6.1s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__006__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '006__2025-09-20__TSC_vs_RVFC__SHOT__t31020.00-t31680.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [006] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\006__2025-09-20__TSC_vs_RVFC__SHOT__t31020.00-t31680.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.08 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.08 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 009: PRESSURE SHOT (score=5.8, burst=5.0s @ 9.7s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__009__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '009__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t36240.00-t37200.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [009] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\009__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t36240.00-t37200.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.65 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.65 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 010: PRESSURE SHOT (score=5.8, burst=5.0s @ 8.2s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__010__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '010__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t40500.00-t41340.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [010] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\010__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t40500.00-t41340.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.21 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.21 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 011: PRESSURE SHOT (score=5.8, burst=5.0s @ 7.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__011__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '011__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t41340.00-t42120.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [011] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\011__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t41340.00-t42120.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.49 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.49 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 012: BUILD UP PLAY SHOT (score=5.8, burst=5.0s @ 6.0s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__012__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '012__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t44760.00-t45420.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [012] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\012__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t44760.00-t45420.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.05 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.05 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 026: DRIBBLING AND SHOT (score=5.8, burst=5.0s @ 5.0s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__026__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '026__2025-09-20__TSC_vs_RVFC__DRIBBLING_AND_SHOT__t1800.00-t1810.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [026] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\026__2025-09-20__TSC_vs_RVFC__DRIBBLING_AND_SHOT__t1800.00-t1810.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.00 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.00 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 029: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 5.0s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__029__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '029__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t1883.00-t1893.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [029] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\029__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t1883.00-t1893.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.97 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.97 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 030: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 7.1s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__030__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '030__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t1898.00-t1911.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [030] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\030__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t1898.00-t1911.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.13 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.13 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 031: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 5.7s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__031__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '031__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t1964.00-t1975.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [031] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\031__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t1964.00-t1975.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.69 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.69 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 033: BUILD UP PLAY AND SHOT (score=5.8, burst=5.0s @ 11.4s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__033__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '033__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_AND_SHOT__t2131.00-t2150.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [033] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\033__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_AND_SHOT__t2131.00-t2150.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.42 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.42 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 035: BUILD UP PLAY AND SHOT (score=5.8, burst=5.0s @ 8.5s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__035__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '035__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_AND_SHOT__t2297.00-t2312.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [035] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\035__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_AND_SHOT__t2297.00-t2312.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.54 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.54 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 037: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 4.2s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__037__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '037__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t2388.00-t2397.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [037] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\037__2025-09-20__TSC_vs_RVFC__PRESSURE_AND_SHOT__t2388.00-t2397.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.19 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.19 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 015: SHOT (score=4.8, burst=5.0s @ 2.8s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__015__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '015__2025-09-20__TSC_vs_RVFC__SHOT__t55800.00-t56220.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [015] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\015__2025-09-20__TSC_vs_RVFC__SHOT__t55800.00-t56220.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.83 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.83 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 016: BUILD UP PLAY SHOT (score=4.8, burst=5.0s @ 14.7s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__016__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '016__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t59940.00-t61320.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [016] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\016__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t59940.00-t61320.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.66 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.66 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 027: SHOT (score=4.8, burst=5.0s @ 2.4s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__027__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '027__2025-09-20__TSC_vs_RVFC__SHOT__t1827.00-t1834.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [027] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\027__2025-09-20__TSC_vs_RVFC__SHOT__t1827.00-t1834.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.42 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.42 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 032: BUILD UP PLAY AND SHOT (score=4.8, burst=5.0s @ 14.3s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__032__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '032__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_AND_SHOT__t2034.00-t2057.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [032] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\032__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_AND_SHOT__t2034.00-t2057.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.33 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.33 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 040: BUILD UP PLAY AND SHOT (score=4.8, burst=5.0s @ 17.8s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__040__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '040__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_AND_SHOT__t2571.00-t2599.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [040] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\040__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_AND_SHOT__t2571.00-t2599.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 17.84 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 17.84 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 041: CORNER (score=3.6, burst=4.0s @ 5.4s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__041__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '041__2025-09-20__TSC_vs_RVFC__CORNER__t2712.00-t2722.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [041] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\041__2025-09-20__TSC_vs_RVFC__CORNER__t2712.00-t2722.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.38 -Duration 4.00 `
      -Fps 24.000 -Label "CORNER" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.38 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 042: CORNER (score=3.6, burst=4.0s @ 11.1s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__042__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '042__2025-09-20__TSC_vs_RVFC__CORNER__t2765.00-t2783.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [042] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\042__2025-09-20__TSC_vs_RVFC__CORNER__t2765.00-t2783.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.14 -Duration 4.00 `
      -Fps 24.000 -Label "CORNER" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.14 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 008: BUILD UP PLAY SHOT (score=3.4, burst=5.0s @ 29.1s, fps=24.000) ⚠ END-BIASED (clip=36s, review recommended)
Write-Warning "Clip 008 `(BUILD UP PLAY SHOT`) is 36s -- end-biased burst at 29.1-34.1s may need manual review"
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__008__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '008__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t33900.00-t36000.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [008] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\008__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t33900.00-t36000.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 29.08 -Duration 5.00 `
      -Fps 24.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 29.08 -Duration 5.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 013: BUILD UP PLAY SHOT (score=3.4, burst=5.0s @ 37.0s, fps=23.976) ⚠ END-BIASED (clip=45s, review recommended)
Write-Warning "Clip 013 `(BUILD UP PLAY SHOT`) is 45s -- end-biased burst at 37.0-42.0s may need manual review"
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__013__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '013__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t49260.00-t51900.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [013] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\013__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t49260.00-t51900.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 36.96 -Duration 5.00 `
      -Fps 23.976 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 36.96 -Duration 5.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 028: CORNER (score=3.0, burst=4.0s @ 14.8s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__028__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '028__2025-09-20__TSC_vs_RVFC__CORNER__t1852.00-t1875.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [028] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\028__2025-09-20__TSC_vs_RVFC__CORNER__t1852.00-t1875.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.83 -Duration 4.00 `
      -Fps 24.000 -Label "CORNER" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.83 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 014: DEFENSE BUILD UP (score=2.4, burst=3.5s @ 8.1s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__014__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '014__2025-09-20__TSC_vs_RVFC__DEFENSE_BUILD_UP__t52800.00-t53820.00__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [014] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\014__2025-09-20__TSC_vs_RVFC__DEFENSE_BUILD_UP__t52800.00-t53820.00.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.06 -Duration 3.50 `
      -Fps 24.000 -Label "BUILD" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.06 -Duration 3.50 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 017: t1088.00-t1124.00 portrait POST (score=0.0, burst=4.0s @ 30.4s, fps=23.976) ⚠ END-BIASED (clip=37s, review recommended)
Write-Warning "Clip 017 `(t1088.00-t1124.00 portrait POST`) is 37s -- end-biased burst at 30.4-34.4s may need manual review"
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__017__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '017__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t1088.00-t1124.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [017] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\017__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t1088.00-t1124.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 30.38 -Duration 4.00 `
      -Fps 23.976 -Label "T1088.00-T1124.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 30.38 -Duration 4.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 018: t1147.00-t1153.00 portrait POST (score=0.0, burst=4.0s @ 2.1s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__018__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '018__2025-09-20__TSC_vs_RVFC__DRIBBLING_SHOT__t1147.00-t1153.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [018] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\018__2025-09-20__TSC_vs_RVFC__DRIBBLING_SHOT__t1147.00-t1153.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.08 -Duration 4.00 `
      -Fps 23.976 -Label "T1147.00-T1153.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.08 -Duration 4.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 019: t1158.00-t1182.00 portrait POST (score=0.0, burst=4.0s @ 12.9s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__019__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '019__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t1158.00-t1182.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [019] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\019__2025-09-20__TSC_vs_RVFC__BUILD_UP_PLAY_SHOT__t1158.00-t1182.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.88 -Duration 4.00 `
      -Fps 23.976 -Label "T1158.00-T1182.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.88 -Duration 4.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 020: t1236.00-t1243.00 portrait POST (score=0.0, burst=4.0s @ 2.7s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__020__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '020__2025-09-20__TSC_vs_RVFC__SHOT__t1236.00-t1243.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [020] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\020__2025-09-20__TSC_vs_RVFC__SHOT__t1236.00-t1243.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.68 -Duration 4.00 `
      -Fps 24.000 -Label "T1236.00-T1243.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.68 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 021: t1275.00-t1293.00 portrait POST (score=0.0, burst=4.0s @ 9.3s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__021__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '021__2025-09-20__TSC_vs_RVFC__CROSS__t1275.00-t1293.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [021] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\021__2025-09-20__TSC_vs_RVFC__CROSS__t1275.00-t1293.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.28 -Duration 4.00 `
      -Fps 24.000 -Label "T1275.00-T1293.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.28 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 022: t1300.00-t1310.00 portrait POST (score=0.0, burst=4.0s @ 4.5s, fps=23.976)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__022__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '022__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t1300.00-t1310.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [022] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\022__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t1300.00-t1310.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.48 -Duration 4.00 `
      -Fps 23.976 -Label "T1300.00-T1310.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.48 -Duration 4.00 `
      -Fps 23.976 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 023: t1316.00-t1329.00 portrait POST (score=0.0, burst=4.0s @ 6.2s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__023__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '023__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t1316.00-t1329.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [023] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\023__2025-09-20__TSC_vs_RVFC__PRESSURE_SHOT__t1316.00-t1329.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.25 -Duration 4.00 `
      -Fps 24.000 -Label "T1316.00-T1329.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.25 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 024: t1402.00-t1421.00 portrait POST (score=0.0, burst=4.0s @ 9.8s, fps=24.000)
$burstOut = "$burstDir\2025-09-20__TSC_vs_RVFC__024__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2025-09-20__TSC_vs_RVFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '024__2025-09-20__TSC_vs_RVFC__FREE_KICK__t1402.00-t1421.00_portrait_POST__portrait__FINAL*.mp4'
$portraitHits = @()
if (Test-Path $gameDir) {
  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {
  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -eq 0) {
  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)
}
if ($portraitHits.Count -gt 0) {
  # Pick newest render to avoid stale files
  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $srcClip = $newest.FullName
  $isPortrait = $true
  if ($portraitHits.Count -gt 1) {
    Write-Host "  [024] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"
  }
} else {
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2025-09-20__TSC_vs_RVFC\024__2025-09-20__TSC_vs_RVFC__FREE_KICK__t1402.00-t1421.00_portrait_POST.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.85 -Duration 4.00 `
      -Fps 24.000 -Label "T1402.00-T1421.00 PORTRAIT POST" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.85 -Duration 4.00 `
      -Fps 24.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# ─── Generate end card ───
$endCardVid = Join-Path $burstDir "end_card.mp4"
if (Test-Path $endCardPNG) {
  ffmpeg -hide_banner -loglevel warning -y -loop 1 -t 3.0 -i $endCardPNG `
    -c:v libx264 -crf $crf -preset fast -pix_fmt yuv420p -an `
    $endCardVid
  $concatEntries += "file '$endCardVid'"
  Write-Host "  End card generated"
} else {
  Write-Warning "End card PNG not found: $endCardPNG"
}

# ═══════════════════════════════════════════════════════════
# PASS 2: Assemble all bursts + slates into final montage
# ═══════════════════════════════════════════════════════════

$concatFile = Join-Path $PSScriptRoot "burst_concat_list.txt"
$concatEntries | Out-File -FilePath $concatFile -Encoding ascii

$output = Join-Path $PSScriptRoot "BurstHighlights__multi_2games.mp4"
$outputDir = Split-Path -Parent $output
if (!(Test-Path $outputDir)) { New-Item -ItemType Directory -Force $outputDir | Out-Null }

Write-Host ""
Write-Host "Assembling $extractCount bursts `($skipCount skipped`)..."

ffmpeg -y -f concat -safe 0 -i $concatFile `
  -c:v libx264 -crf $crf -preset medium `
  -c:a aac -b:a 128k -ar 48000 `
  -movflags +faststart `
  $output

Write-Host ""
Write-Host "Done! Montage: $output"
Write-Host "  Bursts extracted: $extractCount"
Write-Host "  Skipped `(missing`): $skipCount"

# Cleanup burst clips (uncomment to keep them)
# Remove-Item $burstDir -Recurse -Force
# Remove-Item $concatFile -Force