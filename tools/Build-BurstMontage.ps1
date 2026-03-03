# ═══════════════════════════════════════════════════════════
# TSC Season Burst Montage — Build Script
# Generated: 2026-03-03T15:05:53
# ═══════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

$outW = 1080
$outH = 1920
$fps = 30.0
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

# Clip 005: BUILD AND GOAL (score=9.0, burst=4.5s @ 11.2s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__005__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\005__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\005__2026-02-23__TSC_vs_NEOFC__BUILD_AND_GOAL__t416.00-t430.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.18 -t 4.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 005"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.18 -t 4.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 027: PRESSURE, DRIBBLING AND GOAL (score=9.0, burst=4.0s @ 10.0s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__027__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\027__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\027__2026-02-23__TSC_vs_NEOFC__PRESSURE,_DRIBBLING_AND_GOAL__t2366.00-t2377.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.97 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 027"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.97 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 002: BUILD AND SHOTS (score=5.8, burst=4.0s @ 11.2s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__002__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\002__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.23 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 002"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.23 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 004: PRESSURE AND SHOT (score=5.8, burst=3.5s @ 12.4s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__004__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\004__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.45 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 004"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.45 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 019: BUILD AND SHOT (score=5.8, burst=4.0s @ 12.7s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__019__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\019__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\019__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOT__t1585.00-t1602.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.67 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 019"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.67 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 020: PRESSURE AND SHOT (score=5.8, burst=3.5s @ 4.8s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__020__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\020__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\020__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t1627.00-t1634.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.85 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 020"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.85 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 021: PRESSURE AND SHOT (score=5.8, burst=3.5s @ 11.5s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__021__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\021__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\021__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t1639.00-t1653.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.51 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 021"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.51 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 022: BUILD AND SHOT (score=5.8, burst=4.0s @ 12.5s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__022__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\022__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\022__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOT__t1712.00-t1727.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.51 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 022"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.51 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 023: THROUGH BALL, SKILL AND SHOT (score=5.8, burst=3.5s @ 10.0s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__023__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\023__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\023__2026-02-23__TSC_vs_NEOFC__THROUGH_BALL,_SKILL_AND_SHOT__t2178.00-t2191.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.00 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 023"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.00 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 024: PRESSURE AND SHOT (score=5.8, burst=3.5s @ 9.3s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__024__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\024__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\024__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t2247.00-t2257.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.33 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 024"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.33 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 025: BUILD, CROSS AND SHOT (score=5.8, burst=4.0s @ 10.5s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__025__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\025__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\025__2026-02-23__TSC_vs_NEOFC__BUILD,_CROSS_AND_SHOT__t2280.00-t2295.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.53 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 025"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.53 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 026: PRESSURE AND SHOT (score=5.8, burst=3.5s @ 5.7s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__026__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\026__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\026__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t2317.00-t2325.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.74 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 026"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.74 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 028: SHOT (score=5.8, burst=3.5s @ 6.4s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__028__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\028__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\028__2026-02-23__TSC_vs_NEOFC__SHOT__t2434.00-t2440.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.41 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 028"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.41 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 029: DEFENSE, COUNTER AND SHOT (score=5.8, burst=3.5s @ 10.4s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__029__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\029__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\029__2026-02-23__TSC_vs_NEOFC__DEFENSE,_COUNTER_AND_SHOT__t2478.00-t2492.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.42 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 029"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.42 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 030: SHOT (score=5.8, burst=3.5s @ 4.8s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__030__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\030__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\030__2026-02-23__TSC_vs_NEOFC__SHOT__t2793.00-t2801.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.80 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 030"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.80 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 031: SHOT (score=5.8, burst=3.5s @ 5.1s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__031__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\031__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\031__2026-02-23__TSC_vs_NEOFC__SHOT__t2844.00-t2848.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.11 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 031"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.11 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 032: DEFENSE, BUILD AND SHOT (score=5.8, burst=4.0s @ 11.6s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__032__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\032__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\032__2026-02-23__TSC_vs_NEOFC__DEFENSE,_BUILD_AND_SHOT__t2864.00-t2877.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.60 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 032"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.60 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 033: DEFENSE, DRIBBLING AND SHOT (score=5.8, burst=3.5s @ 12.0s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__033__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\033__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\033__2026-02-23__TSC_vs_NEOFC__DEFENSE,_DRIBBLING_AND_SHOT__t2941.00-t2955.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.03 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 033"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 12.03 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 034: CROSS AND SHOT (score=5.8, burst=3.5s @ 5.0s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__034__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\034__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\034__2026-02-23__TSC_vs_NEOFC__CROSS_AND_SHOT__t2976.00-t2983.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.00 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 034"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.00 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 015: BUILD AND GOAL (score=5.2, burst=4.5s @ 26.0s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__015__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\015__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\015__2026-02-23__TSC_vs_NEOFC__BUILD_AND_GOAL__t1091.00-t1120.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 26.03 -t 4.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 015"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 26.03 -t 4.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 003: BUILD AND CROSS (score=4.8, burst=3.5s @ 10.8s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__003__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\003__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\003__2026-02-23__TSC_vs_NEOFC__BUILD_AND_CROSS__t106.00-t122.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.81 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 003"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.81 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 006: BUILD AND SHOT (score=4.8, burst=4.0s @ 14.9s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__006__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\006__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\006__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOT__t456.00-t477.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 14.91 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 006"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 14.91 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 007: CROSS (score=4.8, burst=3.5s @ 4.6s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__007__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\007__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\007__2026-02-23__TSC_vs_NEOFC__CROSS__t649.00-t656.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.60 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 007"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 4.60 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 010: DEFENSE AND CROSS (score=4.8, burst=3.5s @ 8.9s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__010__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\010__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\010__2026-02-23__TSC_vs_NEOFC__DEFENSE_AND_CROSS__t863.00-t875.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.95 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 010"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 8.95 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 011: PRESSURE AND CROSS (score=4.8, burst=3.5s @ 9.1s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__011__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\011__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\011__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_CROSS__t883.00-t896.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.08 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 011"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 9.08 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 013: PRESSURE AND CROSS (score=4.8, burst=3.5s @ 11.4s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__013__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\013__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\013__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_CROSS__t967.00-t982.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.44 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 013"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 11.44 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 014: BUILD, CROSS AND SHOT (score=4.8, burst=4.0s @ 14.3s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__014__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\014__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\014__2026-02-23__TSC_vs_NEOFC__BUILD,_CROSS_AND_SHOT__t1031.00-t1050.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 14.27 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 014"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 14.27 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 017: DEFENSE, BUILD AND CROSS (score=4.8, burst=3.5s @ 10.9s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__017__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\017__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\017__2026-02-23__TSC_vs_NEOFC__DEFENSE,_BUILD_AND_CROSS__t1437.00-t1453.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.94 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 017"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.94 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 001: SAVE (score=4.7, burst=4.0s @ 10.0s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__001__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\001__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\001__2026-02-23__TSC_vs_NEOFC__SAVE__t1.00-t16.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.00 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 001"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 10.00 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 012: FREE KICK (score=3.6, burst=3.5s @ 5.3s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__012__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\012__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\012__2026-02-23__TSC_vs_NEOFC__FREE_KICK__t950.00-t957.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.26 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 012"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 5.26 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 009: CORNER (score=3.0, burst=4.0s @ 3.2s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__009__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\009__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\009__2026-02-23__TSC_vs_NEOFC__CORNER__t850.00-t856.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.18 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 009"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 3.18 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 008: DRIBBLING (score=2.6, burst=4.0s @ 6.3s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__008__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\008__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\008__2026-02-23__TSC_vs_NEOFC__DRIBBLING__t796.00-t808.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.27 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 008"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 6.27 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 016: DEFENSE AND BUILD (score=2.4, burst=3.5s @ 7.9s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__016__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\016__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\016__2026-02-23__TSC_vs_NEOFC__DEFENSE_AND_BUILD__t1269.00-t1286.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.91 -t 3.50 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 016"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 7.91 -t 3.50 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  }
  $concatEntries += "file '$burstOut'"
  $extractCount++
} else {
  Write-Warning "Missing: $srcClip"
  $skipCount++
}

# Clip 018: BUILD (score=2.0, burst=4.0s @ 16.8s)
$burstOut = "$burstDir\2026-02-23__TSC_vs_NEOFC__018__burst.mp4"
# Prefer portrait reel, fall back to atomic clip
$portraitHits = @(Get-ChildItem -Path "$portraitRoot\2026-02-23__TSC_vs_NEOFC\018__*.mp4" -ErrorAction SilentlyContinue)
if ($portraitHits.Count -gt 0) {
  $srcClip = $portraitHits[0].FullName
  $isPortrait = $true
} else {
  $srcClip = "D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\018__2026-02-23__TSC_vs_NEOFC__BUILD__t1535.00-t1561.00.mp4"
  $isPortrait = $false
}
if (Test-Path $srcClip) {
  if ($isPortrait) {
    # Portrait reel: already 1080x1920, just trim
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 16.83 -t 4.00 `
      -i $srcClip `
      -c:v libx264 -crf $crf -preset fast -an `
      $burstOut
  } else {
    # Landscape fallback: scale + letterbox to portrait
    Write-Warning "Using landscape fallback for clip 018"
    ffmpeg -hide_banner -loglevel warning -y `
      -ss 16.83 -t 4.00 `
      -i $srcClip `
      -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
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