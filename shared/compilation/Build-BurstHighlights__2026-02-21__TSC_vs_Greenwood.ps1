# ═══════════════════════════════════════════════════════════
# TSC Season Burst Montage - Build Script
# Generated: 2026-03-04T16:13:51
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

# ─── February 21, 2026: TSC vs Greenwood (20 bursts) ───

# ─── Generate branded intro card ───
$introCard = Join-Path $burstDir "intro_2026-02-21__TSC_vs_Greenwood.mp4"
$introFont = ($fontPath -replace '\\\\', '/') -replace '^([A-Za-z]):','$1\\:'
$introFilters = "drawbox=x=0:y=0:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=0:y=1914:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=80:y=780:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawbox=x=80:y=1140:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawtext=text='TULSA SOCCER CLUB':fontsize=32:fontcolor=0xB7A37C@0.85:x=(w-text_w)/2:y=200,drawtext=text='TSC vs Greenwood':fontsize=72:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=(h-text_h)/2-80,drawtext=text='February 21, 2026':fontsize=40:fontcolor=0xB7A37C@0.9:x=(w-text_w)/2:y=(h)/2+10,drawtext=text='7-2 Win':fontsize=56:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=(h)/2+70,drawtext=text='WSA Futures Cup - Group Stage Match':fontsize=28:fontcolor=0xFFFFFF@0.6:x=(w-text_w)/2:y=1100"
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

$slateFile = Join-Path $PSScriptRoot "slates\2026-02-21__TSC_vs_Greenwood__slate.mp4"
if (Test-Path $slateFile) { $concatEntries += "file '$slateFile'" }

# Clip 001: Free Kick & Goal (score=9.0, burst=10.0s @ 3.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__001__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '001__Free Kick & Goal__551.43-567.97__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\001__Free Kick & Goal__551.43-567.97.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.00 -Duration 10.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.00 -Duration 10.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 002: Goal (score=9.0, burst=8.0s @ 0.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__002__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '002__Goal__751.47-763.77__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\002__Goal__751.47-763.77.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 8.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 8.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 010: Pressure & Goal (score=9.0, burst=8.0s @ 4.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__010__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '010__Pressure & Goal__2007.87-2023.9__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\010__Pressure & Goal__2007.87-2023.9.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.00 -Duration 8.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.00 -Duration 8.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 014: Free Kick & Goal (score=9.0, burst=7.5s @ 0.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__014__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '014__Free Kick & Goal__2410.93-2422.9__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\014__Free Kick & Goal__2410.93-2422.9.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 7.50 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 7.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 017: Corner & Own Goal (score=9.0, burst=6.5s @ 0.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__017__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '017__Corner & Own Goal__2735.87-2747.27__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\017__Corner & Own Goal__2735.87-2747.27.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 6.50 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 6.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 020: Defense & Goal (score=9.0, burst=8.5s @ 4.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__020__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '020__Defense & Goal__3020.83-3038.03__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\020__Defense & Goal__3020.83-3038.03.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.00 -Duration 8.50 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 4.00 -Duration 8.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 005: Corner, Shot & Goal (score=7.5, burst=21.5s @ 0.5s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__005__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '005__Corner, Shot & Goal__1624.73-1650.67__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\005__Corner, Shot & Goal__1624.73-1650.67.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.50 -Duration 21.50 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.50 -Duration 21.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 003: Build Up & Shot (score=5.8, burst=6.5s @ 10.5s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__003__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '003__Build Up & Shot__971.5-989.5__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\003__Build Up & Shot__971.5-989.5.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.50 -Duration 6.50 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 10.50 -Duration 6.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 012: Pressure & Shot (score=5.8, burst=5.0s @ 3.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__012__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '012__Pressure & Shot__2139.2-2155.9__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\012__Pressure & Shot__2139.2-2155.9.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.00 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.00 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 016: Skill & Shot (score=5.8, burst=6.0s @ 7.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__016__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '016__Skill & Shot__2573.5-2592.4__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\016__Skill & Shot__2573.5-2592.4.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.00 -Duration 6.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 7.00 -Duration 6.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 018: Defense, Cross & Shot (score=5.8, burst=13.5s @ 2.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__018__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '018__Defense, Cross & Shot__2807.57-2824.8__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\018__Defense, Cross & Shot__2807.57-2824.8.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.00 -Duration 13.50 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.00 -Duration 13.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 019: Defense & Shot (score=5.8, burst=5.0s @ 3.1s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__019__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '019__Defense & Shot__2845.33-2853.4__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\019__Defense & Shot__2845.33-2853.4.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.08 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.08 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 006: Cross & Shot (score=4.8, burst=6.0s @ 0.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__006__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '006__Cross & Shot__1740.5-1748.37__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\006__Cross & Shot__1740.5-1748.37.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 6.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 6.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 007: Defense, Build, Cross & Shot (score=4.8, burst=16.0s @ 3.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__007__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '007__Defense, Build, Cross & Shot__1780.87-1802.2__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\007__Defense, Build, Cross & Shot__1780.87-1802.2.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.00 -Duration 16.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.00 -Duration 16.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 008: Free Kick (score=3.6, burst=4.0s @ 2.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__008__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '008__Free Kick__1839.63-1848.5__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\008__Free Kick__1839.63-1848.5.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.00 -Duration 4.00 `
      -Fps 30.000 -Label "FREE KICK" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.00 -Duration 4.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 004: Dribbling & Skill (score=2.6, burst=8.2s @ 3.2s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__004__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '004__Dribbling & Skill__1573.5-1586.53__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\004__Dribbling & Skill__1573.5-1586.53.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.25 -Duration 8.25 `
      -Fps 30.000 -Label "DRIBBLING" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.25 -Duration 8.25 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 013: Build, Dribble (score=2.6, burst=11.0s @ 6.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__013__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '013__Build, Dribble__2334.13-2352.27__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\013__Build, Dribble__2334.13-2352.27.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.00 -Duration 11.00 `
      -Fps 30.000 -Label "DRIBBLE" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.00 -Duration 11.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 011: Defense & Build (score=2.4, burst=8.0s @ 6.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__011__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '011__Defense & Build__2050.8-2068.47__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\011__Defense & Build__2050.8-2068.47.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.00 -Duration 8.00 `
      -Fps 30.000 -Label "BUILD" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.00 -Duration 8.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 009: Defense & Build (score=2.0, burst=5.0s @ 0.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__009__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '009__Defense & Build__1929.17-1951.37__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\009__Defense & Build__1929.17-1951.37.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 5.00 `
      -Fps 30.000 -Label "BUILD" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 0.00 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 015: Defense, Pressure & Build (score=1.4, burst=15.0s @ 2.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__015__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '015__Defense, Pressure & Build__2448.5-2501.37__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\015__Defense, Pressure & Build__2448.5-2501.37.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.00 -Duration 15.00 `
      -Fps 30.000 -Label "BUILD" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 2.00 -Duration 15.00 `
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

$output = Join-Path $PSScriptRoot "BurstHighlights__2026-02-21__TSC_vs_Greenwood.mp4"
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