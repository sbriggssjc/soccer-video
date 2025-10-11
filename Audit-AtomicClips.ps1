param(
  [string]$CinematicRoot = ".\out\autoframe_work\cinematic",
  [string]$BrandedRoot   = ".\out\portrait_reels\branded",
  [string]$OutCsv        = ".\AtomicClips.MissingReport.csv"
)

$ErrorActionPreference = 'Stop'

function Strip-Qualifiers([string]$name) {
  $n = $name
  $n = $n -replace '(__portrait_FINAL.*)$',''
  $n = $n -replace '(\.__DEBUG.*)$',''
  $n
}
function Extract-TimeToken([string]$name) {
  if ($name -match 't\d+\.\d+-t\d+\.\d+') { return $Matches[0] }
  return $null
}

if (!(Test-Path $CinematicRoot)) { Write-Host "Missing $CinematicRoot" -ForegroundColor Red; exit 1 }
$brandExists = Test-Path $BrandedRoot
if (-not $brandExists) { Write-Host "WARNING: Missing $BrandedRoot (branded check will be 0 for all)" -ForegroundColor Yellow }

$brandFiles = @()
if ($brandExists) {
  $brandFiles = Get-ChildItem -LiteralPath $BrandedRoot -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.Extension -match '\.mp4$' }
}

$clipDirs =
  Get-ChildItem -LiteralPath $CinematicRoot -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -ne 'follow' }

$rows = foreach ($d in $clipDirs) {
  $clip = $d.Name
  $proxy      = Join-Path $d.FullName "proxy.mp4"
  $follow     = Join-Path $d.FullName "follow"
  $trf        = Join-Path $follow     "transforms.trf"
  $stabilized = Join-Path $follow     "stabilized.mp4"

  $brandedPath = $null
  if ($brandFiles.Count -gt 0) {
    $exact = $brandFiles | Where-Object { $_.BaseName -eq $clip } | Select-Object -First 1
    if ($exact) { $brandedPath = $exact.FullName }
    if (-not $brandedPath) {
      $stripped = Strip-Qualifiers $clip
      $cand = $brandFiles | Where-Object { $_.BaseName -eq $stripped } | Select-Object -First 1
      if ($cand) { $brandedPath = $cand.FullName }
    }
    if (-not $brandedPath) {
      $tok = Extract-TimeToken $clip
      if ($tok) {
        $cand = $brandFiles | Where-Object { $_.BaseName -like "*$tok*" } | Select-Object -First 1
        if ($cand) { $brandedPath = $cand.FullName }
      }
    }
    if (-not $brandedPath) {
      $cand = $brandFiles | Where-Object { $_.BaseName -like "*$clip*" } | Select-Object -First 1
      if ($cand) { $brandedPath = $cand.FullName }
    }
  }

  [pscustomobject]@{
    clip_folder       = $clip
    proxy_exists      = Test-Path $proxy
    transforms_exists = Test-Path $trf
    stabilized_exists = Test-Path $stabilized
    branded_exists    = [bool]$brandedPath
    proxy_path        = if (Test-Path $proxy)      { (Resolve-Path $proxy).Path } else { $null }
    transforms_path   = if (Test-Path $trf)        { (Resolve-Path $trf).Path } else { $null }
    stabilized_path   = if (Test-Path $stabilized) { (Resolve-Path $stabilized).Path } else { $null }
    branded_path      = $brandedPath
  }
}

$missing = $rows | Where-Object {
  -not $_.proxy_exists -or -not $_.transforms_exists -or -not $_.stabilized_exists -or -not $_.branded_exists
}
$rows | Sort-Object clip_folder | Export-Csv -NoTypeInformation -Encoding UTF8 $OutCsv

Write-Host ""
Write-Host "=== Audit Summary ===" -ForegroundColor Cyan
Write-Host ("Total clips:             {0}" -f ($rows.Count))
Write-Host ("Proxy exists:            {0}" -f (($rows | Where-Object proxy_exists).Count))
Write-Host ("Transforms exists:       {0}" -f (($rows | Where-Object transforms_exists).Count))
Write-Host ("Stabilized exists:       {0}" -f (($rows | Where-Object stabilized_exists).Count))
Write-Host ("Branded exists:          {0}" -f (($rows | Where-Object branded_exists).Count))
Write-Host ("Clips with any missing:  {0}" -f ($missing.Count))
Write-Host ""
Write-Host ("Wrote: {0}" -f (Resolve-Path $OutCsv).Path) -ForegroundColor Green

$missingBrand = $rows | Where-Object { $_.stabilized_exists -and -not $_.branded_exists }
if ($missingBrand) {
  Write-Host "`nBranded missing for (have stabilized):" -ForegroundColor Yellow
  $missingBrand | Select-Object clip_folder, stabilized_path | Format-Table -AutoSize
}
