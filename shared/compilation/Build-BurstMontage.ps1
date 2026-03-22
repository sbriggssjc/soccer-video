# ═══════════════════════════════════════════════════════════
# TSC Season Burst Montage — Build Script
# Generated: 2026-03-03T17:03:47
# ═══════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

$outW = 1080
$outH = 1920
$crf = 18

# Portrait reel root (preferred source — polished 1080x1920)
$portraitRoot = "D:\Projects\soccer-video\out\portrait_reels"

# Working directory for extracted bursts
$burstDir = Join-Path $PSScriptRoot "burst_clips"
if (!(Test-Path $burstDir)) { New-Item -ItemType Directory -Force $burstDir | Out-Null }

$concatEntries = @()
$extractCount = 0
$skipCount = 0

# ─── February 21, 2026: TSC vs Greenwood (20 bursts) ───

$slateFile = Join-Path $PSScriptRoot "slates\2026-02-21__TSC_vs_Greenwood__slate.mp4"
if (Test-Path $slateFile) { $concatEntries += "file '$slateFile'" }

# Clip 001: Free Kick & Goal (score=9.0, burst=5.0s @ 9.4s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__001__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "001__Free Kick & Goal__551.43-567.97__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\001__Free Kick & Goal__551.43-567.97.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.41 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 001"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.41 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 002: Goal (score=9.0, burst=5.0s @ 6.4s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__002__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "002__Goal__751.47-763.77__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\002__Goal__751.47-763.77.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.36 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 002"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.36 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 010: Pressure & Goal (score=9.0, burst=5.0s @ 9.1s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__010__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "010__Pressure & Goal__2007.87-2023.9__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\010__Pressure & Goal__2007.87-2023.9.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.05 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 010"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.05 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 014: Free Kick & Goal (score=9.0, burst=5.0s @ 6.1s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__014__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "014__Free Kick & Goal__2410.93-2422.9__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\014__Free Kick & Goal__2410.93-2422.9.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.12 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 014"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.12 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 017: Corner & Own Goal (score=9.0, burst=5.0s @ 5.7s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__017__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "017__Corner & Own Goal__2735.87-2747.27__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\017__Corner & Own Goal__2735.87-2747.27.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.71 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 017"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.71 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 020: Defense & Goal (score=9.0, burst=5.0s @ 9.9s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__020__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "020__Defense & Goal__3020.83-3038.03__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\020__Defense & Goal__3020.83-3038.03.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.89 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 020"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.89 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 005: Corner, Shot & Goal (score=7.5, burst=5.0s @ 16.2s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__005__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "005__Corner, Shot & Goal__1624.73-1650.67__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\005__Corner, Shot & Goal__1624.73-1650.67.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 16.18 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 005"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 16.18 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 003: Build Up & Shot (score=5.8, burst=5.0s @ 10.5s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__003__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "003__Build Up & Shot__971.5-989.5__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\003__Build Up & Shot__971.5-989.5.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.47 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 003"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.47 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 012: Pressure & Shot (score=5.8, burst=5.0s @ 9.5s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__012__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "012__Pressure & Shot__2139.2-2155.9__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\012__Pressure & Shot__2139.2-2155.9.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.53 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 012"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.53 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 016: Skill & Shot (score=5.8, burst=5.0s @ 11.1s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__016__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "016__Skill & Shot__2573.5-2592.4__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\016__Skill & Shot__2573.5-2592.4.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.11 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 016"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.11 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 018: Defense, Cross & Shot (score=5.8, burst=5.0s @ 9.9s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__018__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "018__Defense, Cross & Shot__2807.57-2824.8__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\018__Defense, Cross & Shot__2807.57-2824.8.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.91 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 018"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.91 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
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
$pFilter = "019__Defense & Shot__2845.33-2853.4__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\019__Defense & Shot__2845.33-2853.4.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.08 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 019"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.08 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 006: Cross & Shot (score=4.8, burst=5.0s @ 2.9s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__006__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "006__Cross & Shot__1740.5-1748.37__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\006__Cross & Shot__1740.5-1748.37.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 2.88 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 006"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 2.88 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 007: Defense, Build, Cross & Shot (score=4.8, burst=5.0s @ 12.9s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__007__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "007__Defense, Build, Cross & Shot__1780.87-1802.2__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\007__Defense, Build, Cross & Shot__1780.87-1802.2.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.86 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 007"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.86 -t 5.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 008: Free Kick (score=3.6, burst=3.5s @ 4.6s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__008__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "008__Free Kick__1839.63-1848.5__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\008__Free Kick__1839.63-1848.5.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.64 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 008"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.64 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 004: Dribbling & Skill (score=2.6, burst=4.0s @ 4.5s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__004__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "004__Dribbling & Skill__1573.5-1586.53__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\004__Dribbling & Skill__1573.5-1586.53.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.52 -t 4.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 004"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.52 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 013: Build, Dribble (score=2.6, burst=4.0s @ 7.1s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__013__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "013__Build, Dribble__2334.13-2352.27__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\013__Build, Dribble__2334.13-2352.27.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.07 -t 4.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 013"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.07 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 011: Defense & Build (score=2.4, burst=3.5s @ 8.0s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__011__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "011__Defense & Build__2050.8-2068.47__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\011__Defense & Build__2050.8-2068.47.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.97 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 011"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.97 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 009: Defense & Build (score=2.0, burst=3.5s @ 10.5s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__009__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "009__Defense & Build__1929.17-1951.37__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\009__Defense & Build__1929.17-1951.37.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.46 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 009"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.46 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 015: Defense, Pressure & Build (score=1.4, burst=3.5s @ 27.3s, fps=30.000)
$burstOut = "$burstDir\2026-02-21__TSC_vs_Greenwood__015__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-21__TSC_vs_Greenwood"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "015__Defense, Pressure & Build__2448.5-2501.37__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-21__TSC_vs_Greenwood\015__Defense, Pressure & Build__2448.5-2501.37.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 27.33 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 015"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 27.33 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# ═══════════════════════════════════════════════════════════
# PASS 2: Assemble all bursts + slates into final montage
# ═══════════════════════════════════════════════════════════

$concatFile = Join-Path $PSScriptRoot "burst_concat_list.txt"
$concatEntries | Out-File -FilePath $concatFile -Encoding ascii

$output = Join-Path $PSScriptRoot "TSC_Season_BurstMontage_2025-26.mp4"
$outputDir = Split-Path -Parent $output
if (!(Test-Path $outputDir)) { New-Item -ItemType Directory -Force $outputDir | Out-Null }

Write-Host ""
Write-Host "Assembling $extractCount bursts ($skipCount skipped)..."

ffmpeg -y -f concat -safe 0 -i $concatFile `
  -c:v libx264 -crf $crf -preset medium `
  -c:a aac -b:a 128k -ar 48000 `
  -movflags +faststart `
  $output

Write-Host ""
Write-Host "Done! Montage: $output"
Write-Host "  Bursts extracted: $extractCount"
Write-Host "  Skipped (missing): $skipCount"

# Cleanup burst clips (uncomment to keep them)
# Remove-Item $burstDir -Recurse -Force
# Remove-Item $concatFile -Force