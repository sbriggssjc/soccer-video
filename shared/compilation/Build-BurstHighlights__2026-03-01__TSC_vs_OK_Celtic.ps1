# ═══════════════════════════════════════════════════════════
# TSC Season Burst Montage - Build Script
# Generated: 2026-03-04T16:19:03
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

# ─── March 01, 2026: TSC vs OK Celtic (30 bursts) ───

# ─── Generate branded intro card ───
$introCard = Join-Path $burstDir "intro_2026-03-01__TSC_vs_OK_Celtic.mp4"
$introFont = ($fontPath -replace '\\\\', '/') -replace '^([A-Za-z]):','$1\\:'
$introFilters = "drawbox=x=0:y=0:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=0:y=1914:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=80:y=780:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawbox=x=80:y=1140:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawtext=text='TULSA SOCCER CLUB':fontsize=32:fontcolor=0xB7A37C@0.85:x=(w-text_w)/2:y=200,drawtext=text='TSC vs OK Celtic':fontsize=72:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=(h-text_h)/2-80,drawtext=text='March 01, 2026':fontsize=40:fontcolor=0xB7A37C@0.9:x=(w-text_w)/2:y=(h)/2+10,drawtext=text='7-0 Win':fontsize=56:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=(h)/2+70,drawtext=text='Celtic Cup 2026':fontsize=28:fontcolor=0xFFFFFF@0.6:x=(w-text_w)/2:y=1100"
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

$slateFile = Join-Path $PSScriptRoot "slates\2026-03-01__TSC_vs_OK_Celtic__slate.mp4"
if (Test-Path $slateFile) { $concatEntries += "file '$slateFile'" }

# Clip 009: Pressure & Goal (score=9.0, burst=5.5s @ 3.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__009__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '009__Pressure & Goal__412-425__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\009__Pressure & Goal__412-425.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.00 -Duration 5.50 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.00 -Duration 5.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 013: Corner & Goal (score=9.0, burst=5.0s @ 9.0s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__013__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '013__Corner & Goal__962-978__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\013__Corner & Goal__962-978.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.02 -Duration 5.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.02 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 025: Corner & Goal (score=9.0, burst=5.0s @ 7.6s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__025__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '025__Corner & Goal__1952-1966__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\025__Corner & Goal__1952-1966.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.58 -Duration 5.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.58 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 026: Free Kick & Goal (score=9.0, burst=3.5s @ 8.5s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__026__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '026__Free Kick & Goal__2309-2328__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\026__Free Kick & Goal__2309-2328.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.50 -Duration 3.50 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.50 -Duration 3.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 027: Goal (score=9.0, burst=5.0s @ 5.4s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__027__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '027__Goal__2482-2493__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\027__Goal__2482-2493.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.42 -Duration 5.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.42 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 004: Build & Goal (score=7.5, burst=5.0s @ 13.3s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__004__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '004__Build & Goal__133-155__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\004__Build & Goal__133-155.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 13.34 -Duration 5.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 13.34 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 020: Pressure & Goal (score=7.5, burst=5.0s @ 14.1s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__020__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '020__Pressure & Goal__1644-1667__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\020__Pressure & Goal__1644-1667.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.06 -Duration 5.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.06 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 001: Defense, Build & Shot (score=5.8, burst=5.0s @ 6.1s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__001__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '001__Defense, Build & Shot__16-28__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\001__Defense, Build & Shot__16-28.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.14 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.14 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 002: Pressure & Shot (score=5.8, burst=5.0s @ 4.0s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__002__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '002__Pressure & Shot__28-37__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\002__Pressure & Shot__28-37.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.98 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.98 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 003: Combination & Shot (score=5.8, burst=5.0s @ 4.7s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__003__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '003__Combination & Shot__47-57__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\003__Combination & Shot__47-57.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.70 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.70 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 005: Shot (score=5.8, burst=5.0s @ 4.7s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__005__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '005__Shot__238-248__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\005__Shot__238-248.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.70 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.70 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 007: Cross & Shot (score=5.8, burst=5.0s @ 5.4s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__007__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '007__Cross & Shot__390-401__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\007__Cross & Shot__390-401.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.42 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.42 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 008: Pressure & Shot (score=5.8, burst=5.0s @ 4.7s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__008__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '008__Pressure & Shot__402-412__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\008__Pressure & Shot__402-412.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.70 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.70 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 010: Build & Shot (score=5.8, burst=5.0s @ 6.1s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__010__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '010__Build & Shot__592-604__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\010__Build & Shot__592-604.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.14 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.14 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 012: Pressure & Shot (score=5.8, burst=5.0s @ 4.7s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__012__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '012__Pressure & Shot__940-950__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\012__Pressure & Shot__940-950.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.70 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.70 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 014: Build & Shot (score=5.8, burst=5.0s @ 11.2s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__014__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '014__Build & Shot__1040-1059__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\014__Build & Shot__1040-1059.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.18 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.18 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 015: Pressure & Shot (score=5.8, burst=5.0s @ 10.5s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__015__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '015__Pressure & Shot__1106-1124__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\015__Pressure & Shot__1106-1124.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.46 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.46 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 018: Pressure & Shot (score=5.8, burst=5.0s @ 4.0s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__018__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '018__Pressure & Shot__1287-1296__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\018__Pressure & Shot__1287-1296.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.98 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.98 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 021: Pressure & Shot (score=5.8, burst=5.0s @ 8.3s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__021__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '021__Pressure & Shot__1707-1722__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\021__Pressure & Shot__1707-1722.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.30 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.30 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 022: Pressure & Shot (score=5.8, burst=5.0s @ 8.3s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__022__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '022__Pressure & Shot__1724-1739__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\022__Pressure & Shot__1724-1739.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.30 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.30 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 028: Pressure & Shot (score=5.8, burst=5.0s @ 5.4s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__028__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '028__Pressure & Shot__2729-2740__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\028__Pressure & Shot__2729-2740.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.42 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.42 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 029: Defense, Build & Shot (score=5.8, burst=5.0s @ 10.5s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__029__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '029__Defense, Build & Shot__2827-2845__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\029__Defense, Build & Shot__2827-2845.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.46 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.46 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 006: Build & Shot (score=4.8, burst=5.0s @ 14.1s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__006__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '006__Build & Shot__271-294__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\006__Build & Shot__271-294.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.06 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 14.06 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 011: Build & Cross (score=4.8, burst=3.5s @ 5.7s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__011__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '011__Build & Cross__860-871__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\011__Build & Cross__860-871.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.73 -Duration 3.50 `
      -Fps 30.000 -Label "CROSS" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.73 -Duration 3.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 017: Pressure & Shot (score=4.8, burst=5.0s @ 12.6s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__017__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '017__Pressure & Shot__1235-1256__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\017__Pressure & Shot__1235-1256.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.62 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 12.62 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 030: Dribbling & Shot (score=4.8, burst=5.0s @ 2.0s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__030__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '030__Dribbling & Shot__2986-2993__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\030__Dribbling & Shot__2986-2993.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.00 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.00 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 016: Corner (score=3.6, burst=4.0s @ 5.9s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__016__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '016__Corner__1206-1217__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\016__Corner__1206-1217.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.92 -Duration 4.00 `
      -Fps 30.000 -Label "CORNER" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.92 -Duration 4.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 024: Corner (score=3.6, burst=4.0s @ 11.0s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__024__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '024__Corner__1912-1930__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\024__Corner__1912-1930.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.96 -Duration 4.00 `
      -Fps 30.000 -Label "CORNER" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.96 -Duration 4.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 023: Free Kick (score=3.0, burst=3.5s @ 2.5s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__023__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '023__Free Kick__1808-1814__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\023__Free Kick__1808-1814.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.50 -Duration 3.50 `
      -Fps 30.000 -Label "FREE KICK" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.50 -Duration 3.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 019: Build & Skill (score=2.4, burst=4.0s @ 7.0s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_OK_Celtic__019__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_OK_Celtic"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '019__Build & Skill__1396-1414__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_OK_Celtic\019__Build & Skill__1396-1414.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.00 -Duration 4.00 `
      -Fps 30.000 -Label "BUILD" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.00 -Duration 4.00 `
      -Fps 30.000 -IsPortrait $isPortrait
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

$output = Join-Path $PSScriptRoot "BurstHighlights__2026-03-01__TSC_vs_OK_Celtic.mp4"
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