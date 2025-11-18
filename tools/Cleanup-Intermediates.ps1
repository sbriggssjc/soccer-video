param(
  [string]$Root = "C:\\Users\\scott\\soccer-video",
  [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$fullRoot = [System.IO.Path]::GetFullPath($Root)
$Trash = Join-Path $fullRoot "out\_trash_intermediates"
New-Item -ItemType Directory -Force -Path $Trash | Out-Null

Write-Host "Trash folder: $Trash" -ForegroundColor Cyan

$script:Stats = @{}
function Add-Stat([string]$Bucket) {
  if (-not $script:Stats.ContainsKey($Bucket)) {
    $script:Stats[$Bucket] = 0
  }
  $script:Stats[$Bucket] += 1
}

function Move-ToTrash([System.IO.FileSystemInfo]$Item, [string]$Reason) {
  if (-not $Item) { return }
  if (-not $Item.FullName.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Warning "Skipping $($Item.FullName) because it is outside the repo root."
    return
  }
  $rel = $Item.FullName.Substring($fullRoot.Length).TrimStart('\\','/')
  $dest = Join-Path $Trash $rel
  if ($WhatIf) {
    Write-Host "[MOVE:$Reason] $($Item.FullName) -> $dest"
    return
  }
  $destDir = Split-Path $dest
  New-Item -ItemType Directory -Force -Path $destDir | Out-Null
  if (Test-Path $dest) {
    Remove-Item -Force -Recurse $dest
  }
  Move-Item -Force -Path $Item.FullName -Destination $dest
  Add-Stat $Reason
}

# 1) Intermediates under autoframe_work + upscaled
$workRoots = @()
$workRoots += (Join-Path $fullRoot "out\autoframe_work")
$workRoots += (Join-Path $fullRoot "out\upscaled")
foreach ($p in $workRoots) {
  if (-not (Test-Path $p)) { continue }
  Get-ChildItem -Path $p -Recurse -File -Include *.mp4,*.mov,*.mkv | ForEach-Object {
    Move-ToTrash $_ "autoframe"
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
      Move-ToTrash $_ "atomic"
    }
  }
}

# 3) Non-final portrait reels (anything not *_portrait_FINAL*.mp4)
$reelsRoot = Join-Path $fullRoot "out\portrait_reels"
if (Test-Path $reelsRoot) {
  Get-ChildItem $reelsRoot -Recurse -File -Include *.mp4 | ForEach-Object {
    if ($_.Name -notlike '*_portrait_FINAL*.mp4') {
      Move-ToTrash $_ "portrait"
    }
  }
}

$removed = ($script:Stats.GetEnumerator() | Measure-Object -Property Value -Sum).Sum
if (-not $removed) { $removed = 0 }
Write-Host "Cleanup complete. Removed $removed item(s)." -ForegroundColor Green
if ($script:Stats.Count -gt 0) {
  $script:Stats.GetEnumerator() | Sort-Object Name | ForEach-Object {
    Write-Host ("  {0}: {1}" -f $_.Key, $_.Value) -ForegroundColor DarkGray
  }
}
