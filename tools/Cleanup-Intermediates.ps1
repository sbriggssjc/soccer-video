param(
  [string]$Root = "C:\Users\scott\soccer-video",
  [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$fullRoot = [System.IO.Path]::GetFullPath($Root)
$Trash = Join-Path $fullRoot "out\_trash_intermediates"
$ErrorActionPreference = "Stop"

$Trash = Join-Path $Root "out\_trash_intermediates"
New-Item -ItemType Directory -Force -Path $Trash | Out-Null

Write-Host "Trash folder: $Trash" -ForegroundColor Cyan

function Move-ToTrash([System.IO.FileSystemInfo]$Item) {
  $rel = $Item.FullName.Substring($fullRoot.Length).TrimStart('\','/')
  $dest = Join-Path $Trash $rel
  if ($WhatIf) {
    Write-Host "[MOVE] $($Item.FullName) -> $dest"
    return
  }
  $destDir = Split-Path $dest
  New-Item -ItemType Directory -Force -Path $destDir | Out-Null
  Move-Item -Force $Item.FullName $dest
}
# 1) Intermediates under autoframe_work + upscaled
$paths = @(
  (Join-Path $Root "out\autoframe_work"),
  (Join-Path $Root "out\upscaled")
)

# 1) Intermediates under autoframe_work + upscaled
$workRoots = @()
$workRoots += (Join-Path $fullRoot "out\autoframe_work")
$workRoots += (Join-Path $fullRoot "out\upscaled")
foreach ($p in $workRoots) {
  if (-not (Test-Path $p)) { continue }
  Get-ChildItem -Path $p -Recurse -File -Include *.mp4,*.mov,*.mkv | ForEach-Object {
    Move-ToTrash $_
  Get-ChildItem -Path $p -Recurse -File -Include *.mp4 | ForEach-Object {
    $rel  = $_.FullName.Substring($Root.Length).TrimStart('\')
    $dest = Join-Path $Trash $rel
    if ($WhatIf) {
      Write-Host "[MOVE] $($_.FullName) -> $dest"
    } else {
      New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
      Move-Item -Force $_.FullName $dest
    }
  }
}

# 2) Non-final variants inside atomic_clips
$atomicRoot = Join-Path $fullRoot "out\atomic_clips"
$atomicPatterns = @('*__CINEMATIC*','*__DEBUG*','*__TEST*','*__WORKING*','*__TEMP*','*__x2*')
if (Test-Path $atomicRoot) {
  Get-ChildItem $atomicRoot -Recurse -File -Include *.mp4 | ForEach-Object {
    $name = $_.Name
    $shouldRemove = $false
    foreach ($pattern in $atomicPatterns) {
      if ($name -like $pattern) { $shouldRemove = $true; break }
    }
    if ($shouldRemove) {
      Move-ToTrash $_
  Get-ChildItem $atomicRoot -Recurse -File -Filter "*.mp4" | ForEach-Object {
    $name = $_.Name
    # Keep only plain atomic (no __CINEMATIC, no __DEBUG, no __x2, etc.)
    if ($name -like "*__CINEMATIC*" -or $name -like "*__DEBUG*" -or $name -like "*__x2*") {
      $rel  = $_.FullName.Substring($Root.Length).TrimStart('\')
      $dest = Join-Path $Trash $rel
      if ($WhatIf) {
        Write-Host "[MOVE] $($_.FullName) -> $dest"
      } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        Move-Item -Force $_.FullName $dest
      }
    }
  }
}

# 3) Non-final portrait reels (anything not *_portrait_FINAL*.mp4)
$reelsRoot = Join-Path $fullRoot "out\portrait_reels"
if (Test-Path $reelsRoot) {
  Get-ChildItem $reelsRoot -Recurse -File -Include *.mp4 | ForEach-Object {
    if ($_.Name -notlike '*_portrait_FINAL*.mp4') {
      Move-ToTrash $_
  Get-ChildItem $reelsRoot -Recurse -File -Filter "*.mp4" | ForEach-Object {
    if ($_.Name -notlike "*_portrait_FINAL.mp4") {
      $rel  = $_.FullName.Substring($Root.Length).TrimStart('\')
      $dest = Join-Path $Trash $rel
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
