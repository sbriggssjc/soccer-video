param(
  [string]$Root = "C:\Users\scott\soccer-video",
  [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$fullRoot = [System.IO.Path]::GetFullPath($Root)
$trashRoot = Join-Path $fullRoot "out\_trash_intermediates"
if (-not $WhatIf) {
  New-Item -ItemType Directory -Force -Path $trashRoot | Out-Null
}

function Move-ToTrash {
  param([System.IO.FileSystemInfo]$Item)
  if (-not $Item) { return }
  if ($Item -is [System.IO.FileInfo]) {
    $name = $Item.Name.ToLowerInvariant()
    if ($name.EndsWith('.json') -or $name.EndsWith('.jsonl')) {
      return
    }
  }
  $rel = $Item.FullName.Substring($fullRoot.Length).TrimStart(@('\\','/'))
  $dest = Join-Path $trashRoot $rel
  if ($WhatIf) {
    Write-Host "[MOVE] $($Item.FullName) -> $dest" -ForegroundColor Yellow
    return
  }
  $destDir = Split-Path -Parent $dest
  New-Item -ItemType Directory -Force -Path $destDir | Out-Null
  Move-Item -Force -LiteralPath $Item.FullName -Destination $dest
}

Write-Host "Cleaning intermediates under $fullRoot" -ForegroundColor Cyan

$videoExtensions = @('*.mp4','*.mov','*.mkv')
foreach ($work in @("out\autoframe_work","out\upscaled")) {
  $path = Join-Path $fullRoot $work
  if (-not (Test-Path $path)) { continue }
  Get-ChildItem -Path $path -Recurse -File -Include $videoExtensions | ForEach-Object { Move-ToTrash $_ }
}

$planRoot = Join-Path $fullRoot "out\plans"
if (Test-Path $planRoot) {
  Write-Host "Preserving camera plans under $planRoot" -ForegroundColor DarkCyan
}

$atomicRoot = Join-Path $fullRoot "out\atomic_clips"
$atomicPatterns = @('*__CINEMATIC*','*__DEBUG*','*__TEST*','*__WORKING*','*__TEMP*','*__x2*','*.rerender*.mp4','.__*.mp4')
if (Test-Path $atomicRoot) {
  Get-ChildItem -Path $atomicRoot -Recurse -File -Filter "*.mp4" | ForEach-Object {
    $name = $_.Name
    $isRerender = $_.FullName -match "\\rerender(\\|$)"
    $matchPattern = $false
    foreach ($pattern in $atomicPatterns) {
      if ($name -like $pattern) { $matchPattern = $true; break }
    }
    if ($matchPattern -or $isRerender) {
      Move-ToTrash $_
    }
  }
}

$reelsRoot = Join-Path $fullRoot "out\portrait_reels"
if (Test-Path $reelsRoot) {
  Get-ChildItem -Path $reelsRoot -Recurse -File -Filter "*.mp4" | ForEach-Object {
    if ($_.Name -notmatch '_portrait_FINAL\.mp4$') {
      Move-ToTrash $_
    }
  }
}

Write-Host "Cleanup complete. Intermediates moved to $trashRoot" -ForegroundColor Green
