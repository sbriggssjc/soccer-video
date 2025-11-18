param(
  [string]$Root = "C:\Users\scott\soccer-video",
  [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

$Trash = Join-Path $Root "out\_trash_intermediates"
New-Item -ItemType Directory -Force -Path $Trash | Out-Null

Write-Host "Trash folder: $Trash" -ForegroundColor Cyan

# 1) Intermediates under autoframe_work + upscaled
$paths = @(
  Join-Path $Root "out\autoframe_work",
  Join-Path $Root "out\upscaled"
)

foreach ($p in $paths) {
  if (-not (Test-Path $p)) { continue }
  Get-ChildItem -Path $p -Recurse -File -Include *.mp4 | ForEach-Object {
    $dest = Join-Path $Trash ($_.FullName.Substring($Root.Length).TrimStart('\'))
    if ($WhatIf) {
      Write-Host "[MOVE] $($_.FullName) -> $dest"
    } else {
      New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
      Move-Item -Force $_.FullName $dest
    }
  }
}

# 2) Non-final variants inside atomic_clips
$atomicRoot = Join-Path $Root "out\atomic_clips"
if (Test-Path $atomicRoot) {
  Get-ChildItem $atomicRoot -Recurse -File -Filter '*.mp4' | ForEach-Object {
    $name = $_.Name
    # Keep only plain atomic (no __CINEMATIC, no __DEBUG, no __x2, etc.)
    if ($name -like '*__CINEMATIC*' -or $name -like '*__DEBUG*' -or $name -like '*__x2*') {
      $dest = Join-Path $Trash ($_.FullName.Substring($Root.Length).TrimStart('\'))
      if ($WhatIf) {
        Write-Host "[MOVE] $($_.FullName) -> $dest"
      } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        Move-Item -Force $_.FullName $dest
      }
    }
  }
}

# 3) Non-final portrait reels (anything not *_portrait_FINAL.mp4)
$reelsRoot = Join-Path $Root "out\portrait_reels"
if (Test-Path $reelsRoot) {
  Get-ChildItem $reelsRoot -Recurse -File -Filter '*.mp4' | ForEach-Object {
    if ($_.Name -notlike '*_portrait_FINAL.mp4') {
      $dest = Join-Path $Trash ($_.FullName.Substring($Root.Length).TrimStart('\'))
      if ($WhatIf) {
        Write-Host "[MOVE] $($_.FullName) -> $dest"
      } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        Move-Item -Force $_.FullName $dest
      }
    }
  }
}

Write-Host "Cleanup complete." -ForegroundColor Green
