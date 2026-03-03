# ═══════════════════════════════════════════════════════════
# TSC Season Burst Montage — Build Script
# Generated: 2026-03-03T22:42:53
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

# ─── February 23, 2026: TSC vs NEO FC (34 bursts) ───

$slateFile = Join-Path $PSScriptRoot "slates\2026-02-23__TSC_vs_NEOFC__slate.mp4"
if (Test-Path $slateFile) { $concatEntries += "file '$slateFile'" }

# Clip 005: BUILD AND GOAL (score=9.0, burst=5.0s @ 8.9s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__005__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "005__2026-02-23__TSC_vs_NEOFC__BUILD_AND_GOAL__t416.00-t430.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\005__2026-02-23__TSC_vs_NEOFC__BUILD_AND_GOAL__t416.00-t430.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.88 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 005"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.88 -t 5.00 `
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

# Clip 027: PRESSURE, DRIBBLING AND GOAL (score=9.0, burst=5.0s @ 8.0s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__027__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "027__2026-02-23__TSC_vs_NEOFC__PRESSURE,_DRIBBLING_AND_GOAL__t2366.00-t2377.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\027__2026-02-23__TSC_vs_NEOFC__PRESSURE,_DRIBBLING_AND_GOAL__t2366.00-t2377.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.01 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 027"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.01 -t 5.00 `
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

# Clip 002: BUILD AND SHOTS (score=5.8, burst=5.0s @ 9.4s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__002__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.40 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 002"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.40 -t 5.00 `
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

# Clip 004: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 10.6s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__004__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.60 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 004"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.60 -t 5.00 `
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

# Clip 019: BUILD AND SHOT (score=5.8, burst=5.0s @ 10.7s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__019__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "019__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOT__t1585.00-t1602.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\019__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOT__t1585.00-t1602.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.70 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 019"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.70 -t 5.00 `
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

# Clip 020: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 3.5s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__020__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "020__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t1627.00-t1634.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\020__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t1627.00-t1634.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.47 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 020"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.47 -t 5.00 `
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

# Clip 021: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 9.7s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__021__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "021__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t1639.00-t1653.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\021__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t1639.00-t1653.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.74 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 021"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.74 -t 5.00 `
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

# Clip 022: BUILD AND SHOT (score=5.8, burst=5.0s @ 10.6s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__022__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "022__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOT__t1712.00-t1727.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\022__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOT__t1712.00-t1727.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.56 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 022"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.56 -t 5.00 `
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

# Clip 023: THROUGH BALL, SKILL AND SHOT (score=5.8, burst=5.0s @ 8.3s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__023__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "023__2026-02-23__TSC_vs_NEOFC__THROUGH_BALL,_SKILL_AND_SHOT__t2178.00-t2191.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\023__2026-02-23__TSC_vs_NEOFC__THROUGH_BALL,_SKILL_AND_SHOT__t2178.00-t2191.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.35 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 023"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.35 -t 5.00 `
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

# Clip 024: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 7.7s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__024__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "024__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t2247.00-t2257.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\024__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t2247.00-t2257.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.72 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 024"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.72 -t 5.00 `
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

# Clip 025: BUILD, CROSS AND SHOT (score=5.8, burst=5.0s @ 8.8s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__025__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "025__2026-02-23__TSC_vs_NEOFC__BUILD,_CROSS_AND_SHOT__t2280.00-t2295.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\025__2026-02-23__TSC_vs_NEOFC__BUILD,_CROSS_AND_SHOT__t2280.00-t2295.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.78 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 025"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.78 -t 5.00 `
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

# Clip 026: PRESSURE AND SHOT (score=5.8, burst=5.0s @ 4.4s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__026__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "026__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t2317.00-t2325.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\026__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t2317.00-t2325.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.41 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 026"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.41 -t 5.00 `
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

# Clip 028: SHOT (score=5.8, burst=5.0s @ 5.0s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__028__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "028__2026-02-23__TSC_vs_NEOFC__SHOT__t2434.00-t2440.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\028__2026-02-23__TSC_vs_NEOFC__SHOT__t2434.00-t2440.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.04 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 028"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.04 -t 5.00 `
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

# Clip 029: DEFENSE, COUNTER AND SHOT (score=5.8, burst=5.0s @ 8.7s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__029__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "029__2026-02-23__TSC_vs_NEOFC__DEFENSE,_COUNTER_AND_SHOT__t2478.00-t2492.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\029__2026-02-23__TSC_vs_NEOFC__DEFENSE,_COUNTER_AND_SHOT__t2478.00-t2492.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.73 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 029"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.73 -t 5.00 `
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

# Clip 030: SHOT (score=5.8, burst=5.0s @ 3.4s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__030__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "030__2026-02-23__TSC_vs_NEOFC__SHOT__t2793.00-t2801.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\030__2026-02-23__TSC_vs_NEOFC__SHOT__t2793.00-t2801.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.40 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 030"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.40 -t 5.00 `
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

# Clip 031: SHOT (score=5.8, burst=5.0s @ 3.8s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__031__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "031__2026-02-23__TSC_vs_NEOFC__SHOT__t2844.00-t2848.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\031__2026-02-23__TSC_vs_NEOFC__SHOT__t2844.00-t2848.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.80 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 031"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.80 -t 5.00 `
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

# Clip 032: DEFENSE, BUILD AND SHOT (score=5.8, burst=5.0s @ 9.7s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__032__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "032__2026-02-23__TSC_vs_NEOFC__DEFENSE,_BUILD_AND_SHOT__t2864.00-t2877.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\032__2026-02-23__TSC_vs_NEOFC__DEFENSE,_BUILD_AND_SHOT__t2864.00-t2877.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.74 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 032"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.74 -t 5.00 `
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

# Clip 033: DEFENSE, DRIBBLING AND SHOT (score=5.8, burst=5.0s @ 10.2s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__033__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "033__2026-02-23__TSC_vs_NEOFC__DEFENSE,_DRIBBLING_AND_SHOT__t2941.00-t2955.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\033__2026-02-23__TSC_vs_NEOFC__DEFENSE,_DRIBBLING_AND_SHOT__t2941.00-t2955.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.22 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 033"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.22 -t 5.00 `
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

# Clip 034: CROSS AND SHOT (score=5.8, burst=5.0s @ 4.0s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__034__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "034__2026-02-23__TSC_vs_NEOFC__CROSS_AND_SHOT__t2976.00-t2983.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\034__2026-02-23__TSC_vs_NEOFC__CROSS_AND_SHOT__t2976.00-t2983.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.98 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 034"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.98 -t 5.00 `
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

# Clip 015: BUILD AND GOAL (score=5.2, burst=5.0s @ 21.4s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__015__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "015__2026-02-23__TSC_vs_NEOFC__BUILD_AND_GOAL__t1091.00-t1120.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\015__2026-02-23__TSC_vs_NEOFC__BUILD_AND_GOAL__t1091.00-t1120.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 21.45 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 015"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 21.45 -t 5.00 `
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

# Clip 003: BUILD AND CROSS (score=4.8, burst=3.5s @ 10.8s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__003__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "003__2026-02-23__TSC_vs_NEOFC__BUILD_AND_CROSS__t106.00-t122.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\003__2026-02-23__TSC_vs_NEOFC__BUILD_AND_CROSS__t106.00-t122.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.81 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 003"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.81 -t 3.50 `
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

# Clip 006: BUILD AND SHOT (score=4.8, burst=5.0s @ 12.7s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__006__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "006__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOT__t456.00-t477.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\006__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOT__t456.00-t477.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.72 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 006"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.72 -t 5.00 `
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

# Clip 007: CROSS (score=4.8, burst=3.5s @ 4.6s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__007__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "007__2026-02-23__TSC_vs_NEOFC__CROSS__t649.00-t656.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\007__2026-02-23__TSC_vs_NEOFC__CROSS__t649.00-t656.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.60 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 007"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.60 -t 3.50 `
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

# Clip 010: DEFENSE AND CROSS (score=4.8, burst=3.5s @ 8.9s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__010__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "010__2026-02-23__TSC_vs_NEOFC__DEFENSE_AND_CROSS__t863.00-t875.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\010__2026-02-23__TSC_vs_NEOFC__DEFENSE_AND_CROSS__t863.00-t875.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.95 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 010"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.95 -t 3.50 `
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

# Clip 011: PRESSURE AND CROSS (score=4.8, burst=3.5s @ 9.1s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__011__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "011__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_CROSS__t883.00-t896.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\011__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_CROSS__t883.00-t896.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.08 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 011"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.08 -t 3.50 `
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

# Clip 013: PRESSURE AND CROSS (score=4.8, burst=3.5s @ 11.4s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__013__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "013__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_CROSS__t967.00-t982.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\013__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_CROSS__t967.00-t982.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.44 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 013"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.44 -t 3.50 `
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

# Clip 014: BUILD, CROSS AND SHOT (score=4.8, burst=5.0s @ 12.1s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__014__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "014__2026-02-23__TSC_vs_NEOFC__BUILD,_CROSS_AND_SHOT__t1031.00-t1050.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\014__2026-02-23__TSC_vs_NEOFC__BUILD,_CROSS_AND_SHOT__t1031.00-t1050.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.14 -t 5.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 014"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.14 -t 5.00 `
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

# Clip 017: DEFENSE, BUILD AND CROSS (score=4.8, burst=3.5s @ 10.9s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__017__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "017__2026-02-23__TSC_vs_NEOFC__DEFENSE,_BUILD_AND_CROSS__t1437.00-t1453.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\017__2026-02-23__TSC_vs_NEOFC__DEFENSE,_BUILD_AND_CROSS__t1437.00-t1453.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.94 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 017"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.94 -t 3.50 `
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

# Clip 001: SAVE (score=4.7, burst=4.0s @ 10.0s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__001__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "001__2026-02-23__TSC_vs_NEOFC__SAVE__t1.00-t16.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\001__2026-02-23__TSC_vs_NEOFC__SAVE__t1.00-t16.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.00 -t 4.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 001"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.00 -t 4.00 `
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

# Clip 012: FREE KICK (score=3.6, burst=3.5s @ 5.3s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__012__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "012__2026-02-23__TSC_vs_NEOFC__FREE_KICK__t950.00-t957.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\012__2026-02-23__TSC_vs_NEOFC__FREE_KICK__t950.00-t957.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.26 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 012"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.26 -t 3.50 `
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

# Clip 009: CORNER (score=3.0, burst=4.0s @ 3.2s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__009__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "009__2026-02-23__TSC_vs_NEOFC__CORNER__t850.00-t856.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\009__2026-02-23__TSC_vs_NEOFC__CORNER__t850.00-t856.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.18 -t 4.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 009"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.18 -t 4.00 `
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

# Clip 008: DRIBBLING (score=2.6, burst=4.0s @ 6.3s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__008__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "008__2026-02-23__TSC_vs_NEOFC__DRIBBLING__t796.00-t808.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\008__2026-02-23__TSC_vs_NEOFC__DRIBBLING__t796.00-t808.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.27 -t 4.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 008"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.27 -t 4.00 `
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

# Clip 016: DEFENSE AND BUILD (score=2.4, burst=3.5s @ 7.9s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__016__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "016__2026-02-23__TSC_vs_NEOFC__DEFENSE_AND_BUILD__t1269.00-t1286.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\016__2026-02-23__TSC_vs_NEOFC__DEFENSE_AND_BUILD__t1269.00-t1286.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.91 -t 3.50 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 016"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.91 -t 3.50 `
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

# Clip 018: BUILD (score=2.0, burst=4.0s @ 16.8s, fps=30.000)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__018__burst.mp4"
$srcClip = $null
$isPortrait = $false
# Tiered portrait search: game subfolder > clean/ > recursive
$gameDir = Join-Path $portraitRoot "2026-02-23__TSC_vs_NEOFC"
$cleanDir = Join-Path $portraitRoot "clean"
$pFilter = "018__2026-02-23__TSC_vs_NEOFC__BUILD__t1535.00-t1561.00__portrait__FINAL*.mp4"
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
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\018__2026-02-23__TSC_vs_NEOFC__BUILD__t1535.00-t1561.00.mp4"
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim at native fps
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 16.83 -t 4.00 `
      -i $srcClip `
      -r 30.000 `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 018"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 16.83 -t 4.00 `
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