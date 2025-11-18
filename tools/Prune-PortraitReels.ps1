param(
  [string]$Root   = "C:\\Users\\scott\\soccer-video",
  [switch]$WhatIf
)

$PortraitRoot = Join-Path $Root "out\\portrait_reels\\clean"
$TrashRoot    = Join-Path $Root "out\\_trash_portrait_reels"

if (-not (Test-Path $PortraitRoot)) {
  Write-Error "Portrait root not found: $PortraitRoot"
  exit 1
}

New-Item -ItemType Directory -Force -Path $TrashRoot | Out-Null

Write-Host "Pruning portraits under $PortraitRoot" -ForegroundColor Cyan
Write-Host "Trash -> $TrashRoot" -ForegroundColor Yellow

# Helper: priority ranking for which file to KEEP per ClipID
function Get-KeepScore([IO.FileInfo]$file) {
  $name = $file.Name

  # 0 = best, higher = worse
  if ($name -like "*__WIDE_portrait_FINAL.__BRANDTMP.mp4") { return 0 }
  if ($name -like "*__WIDE_portrait_FINAL.mp4")             { return 1 }
  if ($name -like "*__CINEMATIC_portrait_FINAL*")           { return 2 }
  if ($name -like "*_portrait_FINAL*")                      { return 3 }
  return 9
}

Get-ChildItem -LiteralPath $PortraitRoot -Filter "*.mp4" |
  Where-Object { $_.Name -match '^\d{3}__' } |  # ClipID prefix
  Group-Object { $_.Name.Substring(0,3) } |     # group by ClipID "001", "002", ...
  ForEach-Object {
    $clipId    = $_.Name
    $files     = $_.Group

    if ($files.Count -le 1) {
      Write-Host "ClipID $clipId already has a single portrait" -ForegroundColor DarkGray
      return
    }

    # Choose the one to keep: lowest score, then newest
    $keep = $files |
      Sort-Object @{ Expression = { Get-KeepScore $_ } },
                   @{ Expression = "LastWriteTime"; Descending = $true } |
      Select-Object -First 1

    Write-Host ""
    Write-Host "ClipID $clipId -> keeping: $($keep.Name)" -ForegroundColor Green

    foreach ($f in $files) {
      if ($f.FullName -eq $keep.FullName) { continue }
      $dest = Join-Path $TrashRoot $f.Name
      if ($WhatIf) {
        Write-Host "  [WhatIf] Would move $($f.FullName) -> $dest" -ForegroundColor Yellow
      } else {
        Write-Host "  Moving $($f.FullName) -> $dest" -ForegroundColor Yellow
        Move-Item -LiteralPath $f.FullName -Destination $dest -Force
      }
    }
  }

Write-Host ""
Write-Host "Portrait pruning complete." -ForegroundColor Cyan
