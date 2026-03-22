# ═══════════════════════════════════════════════════════════
# TSC Season Burst Montage - Build Script
# Generated: 2026-03-03T22:12:54
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

# ─── March 01, 2026: TSC vs North OKC (19 bursts) ───

# ─── Generate branded intro card ───
$introCard = Join-Path $burstDir "intro_2026-03-01__TSC_vs_North_OKC.mp4"
$introFont = ($fontPath -replace '\\\\', '/') -replace '^([A-Za-z]):','$1\\:'
$introFilters = "drawbox=x=0:y=0:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=0:y=1914:w=1080:h=6:c=0x9B1B33:t=fill,drawbox=x=80:y=780:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawbox=x=80:y=1140:w=920:h=3:c=0xB7A37C@0.6:t=fill,drawtext=text='TULSA SOCCER CLUB':fontsize=32:fontcolor=0xB7A37C@0.85:x=(w-text_w)/2:y=200,drawtext=text='TSC vs North OKC':fontsize=72:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=(h-text_h)/2-80,drawtext=text='March 01, 2026':fontsize=40:fontcolor=0xB7A37C@0.9:x=(w-text_w)/2:y=(h)/2+10,drawtext=text='5-0 W':fontsize=56:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=(h)/2+70,drawtext=text='OSSL Spring 2026':fontsize=28:fontcolor=0xFFFFFF@0.6:x=(w-text_w)/2:y=1100"
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

$slateFile = Join-Path $PSScriptRoot "slates\2026-03-01__TSC_vs_North_OKC__slate.mp4"
if (Test-Path $slateFile) { $concatEntries += "file '$slateFile'" }

# Clip 006: Build, Cross & Goal (score=9.0, burst=5.0s @ 11.9s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__006__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '006__Build, Cross & Goal__544-564__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\006__Build, Cross & Goal__544-564.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.90 -Duration 5.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 11.90 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 009: Cross & Goal (score=9.0, burst=5.0s @ 9.7s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__009__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '009__Cross & Goal__1169-1186__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\009__Cross & Goal__1169-1186.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.74 -Duration 5.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 9.74 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 010: Pressure & Goal (score=9.0, burst=5.0s @ 9.0s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__010__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '010__Pressure & Goal__1408-1424__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\010__Pressure & Goal__1408-1424.mp4'
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

# Clip 013: Goal (score=9.0, burst=5.0s @ 6.1s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__013__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '013__Goal__2193-2205__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\013__Goal__2193-2205.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.14 -Duration 5.00 `
      -Fps 30.000 -Label "GOAL" -IsPortrait $isPortrait
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

# Clip 002: Build & Shot (score=5.8, burst=5.0s @ 6.9s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__002__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '002__Build & Shot__155-168__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\002__Build & Shot__155-168.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.86 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 6.86 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 005: Pressure & Shot (score=5.8, burst=5.0s @ 8.3s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__005__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '005__Pressure & Shot__364-379__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\005__Pressure & Shot__364-379.mp4'
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

# Clip 007: Pressure, Build & Shot (score=5.8, burst=5.0s @ 6.1s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__007__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '007__Pressure, Build & Shot__622-634__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\007__Pressure, Build & Shot__622-634.mp4'
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

# Clip 014: Pressure & Shot (score=5.8, burst=5.0s @ 6.1s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__014__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '014__Pressure & Shot__2297-2309__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\014__Pressure & Shot__2297-2309.mp4'
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

# Clip 015: Pressure & Shot (score=5.8, burst=5.0s @ 5.4s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__015__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '015__Pressure & Shot__2409-2420__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\015__Pressure & Shot__2409-2420.mp4'
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

# Clip 016: Pressure & Shot (score=4.8, burst=5.0s @ 16.2s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__016__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '016__Pressure & Shot__2434-2460__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\016__Pressure & Shot__2434-2460.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 16.22 -Duration 5.00 `
      -Fps 30.000 -Label "SHOT" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 16.22 -Duration 5.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 001: Corner (score=3.6, burst=4.0s @ 5.2s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__001__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '001__Corner__69-79__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\001__Corner__69-79.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.20 -Duration 4.00 `
      -Fps 30.000 -Label "CORNER" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.20 -Duration 4.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 003: Corner (score=3.6, burst=4.0s @ 8.8s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__003__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '003__Corner__188-203__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\003__Corner__188-203.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.80 -Duration 4.00 `
      -Fps 30.000 -Label "CORNER" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 8.80 -Duration 4.00 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 017: Free Kick (score=3.6, burst=3.5s @ 5.5s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__017__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '017__Free Kick__2483-2493__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\017__Free Kick__2483-2493.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.45 -Duration 3.50 `
      -Fps 30.000 -Label "FREE KICK" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 5.45 -Duration 3.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 019: Corner (score=3.6, burst=4.0s @ 5.9s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__019__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '019__Corner__2688-2699__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\019__Corner__2688-2699.mp4'
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

# Clip 004: Free Kick (score=3.0, burst=3.5s @ 1.5s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__004__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '004__Free Kick__283-288__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\004__Free Kick__283-288.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 1.50 -Duration 3.50 `
      -Fps 30.000 -Label "FREE KICK" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 1.50 -Duration 3.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 008: Free Kick (score=3.0, burst=3.5s @ 1.5s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__008__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '008__Free Kick__943-948__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\008__Free Kick__943-948.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 1.50 -Duration 3.50 `
      -Fps 30.000 -Label "FREE KICK" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 1.50 -Duration 3.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 011: Free Kick (score=3.0, burst=3.5s @ 1.5s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__011__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '011__Free Kick__1473-1478__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\011__Free Kick__1473-1478.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 1.50 -Duration 3.50 `
      -Fps 30.000 -Label "FREE KICK" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 1.50 -Duration 3.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 012: Free Kick (score=3.0, burst=3.5s @ 3.3s, fps=30.000)
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__012__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '012__Free Kick__1638-1645__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\012__Free Kick__1638-1645.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.29 -Duration 3.50 `
      -Fps 30.000 -Label "FREE KICK" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 3.29 -Duration 3.50 `
      -Fps 30.000 -IsPortrait $isPortrait
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 018: Build & Dribble (score=1.5, burst=11.0s @ 42.0s, fps=30.000) [MANUAL OVERRIDE]
$burstOut = "$burstDir\2026-03-01__TSC_vs_North_OKC__018__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-03-01__TSC_vs_North_OKC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = '018__Build & Dribble__2580-2634__portrait__FINAL*.mp4'
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
  $srcClip = 'D:\Projects\soccer-video\out\atomic_clips\2026-03-01__TSC_vs_North_OKC\018__Build & Dribble__2580-2634.mp4'
}
if (Test-Path $srcClip) {
  if ($hasBrandAssets) {
    Brand-BurstClip -SrcClip $srcClip -OutFile $burstOut `
      -Start 42.00 -Duration 11.00 `
      -Fps 30.000 -Label "DRIBBLE" -IsPortrait $isPortrait
  } else {
    Brand-BurstClipSimple -SrcClip $srcClip -OutFile $burstOut `
      -Start 42.00 -Duration 11.00 `
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

$output = Join-Path $PSScriptRoot "BurstHighlights__2026-03-01__TSC_vs_North_OKC.mp4"
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