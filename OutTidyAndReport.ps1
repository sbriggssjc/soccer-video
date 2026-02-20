<#
OutTidyAndReport.ps1  —  tidy + report for the repo's out\ directory
- Top-level size rollup (skips junctions to avoid double counting)
- Largest files
- Portrait junction insights (largest + ext breakdown)
- Empty dir cleanup
- _tmp duplicate cleanup by SHA256 (only deletes with -Commit)
#>
param(
  [string]$OutRoot = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "out"),
  [int]$Top = 25,
  [switch]$Commit
)
function Show-GB([long]$bytes){ [math]::Round($bytes/1GB,2) }

Write-Host "=== Top-level size rollup (junctions skipped) ==="
Get-ChildItem -Depth 1 -Attributes !ReparsePoint $OutRoot |
  ForEach-Object {
    $bytes = if ($_.PSIsContainer) {
      (Get-ChildItem -Recurse -File $_.FullName -EA SilentlyContinue | Measure-Object Length -Sum).Sum
    } else { (Get-Item $_.FullName).Length }
    [pscustomobject]@{ Name=$_.Name; GB = Show-GB $bytes }
  } | Sort-Object GB -Desc | Format-Table -Auto

# Show any junctions so it’s obvious what’s being skipped
Write-Host "`n=== Junctions at out\ (skipped in rollup) ==="
Get-ChildItem $OutRoot | Where-Object { $_.Attributes -match 'ReparsePoint' } |
  Select-Object Name, FullName

Write-Host "`n=== Largest $Top files under out\ ==="
Get-ChildItem -Recurse -File $OutRoot |
  Sort-Object Length -Desc |
  Select-Object -First $Top FullName,@{n='GB';e={ Show-GB $_.Length }}

# Portrait junction insights
$Portrait = Join-Path $OutRoot 'portrait_1080x1920'
if (Test-Path $Portrait) {
  Write-Host "`n=== Largest 15 under portrait_1080x1920 ==="
  Get-ChildItem -Path "$Portrait\*" -Recurse -File -EA SilentlyContinue |
    Sort-Object Length -Desc |
    Select-Object -First 15 FullName,@{n='GB';e={ Show-GB $_.Length }}

  Write-Host "`n=== Size by extension under portrait_1080x1920 ==="
  Get-ChildItem -Path "$Portrait\*" -Recurse -File -EA SilentlyContinue |
    Group-Object Extension |
    ForEach-Object {
      $sum = ($_.Group | Measure-Object Length -Sum).Sum
      [pscustomobject]@{ Ext = $_.Name; GB = Show-GB $sum; Count = $_.Count }
    } | Sort-Object GB -Desc | Format-Table -Auto
} else {
  Write-Host "`nInfo: $Portrait not found."
}

# ---------- Empty dir cleanup ----------
Write-Host "`n=== Removing empty folders under out\ (commit=$($Commit.IsPresent)) ==="
$empties = Get-ChildItem -Recurse -Directory $OutRoot -EA SilentlyContinue |
  Where-Object { -not (Get-ChildItem -Force $_.FullName -EA SilentlyContinue | Select-Object -First 1) }
$empties | ForEach-Object {
  if ($Commit) { Remove-Item -LiteralPath $_.FullName -Recurse -Force -WhatIf:$false }
  else         { Remove-Item -LiteralPath $_.FullName -Recurse -Force -WhatIf }
}
Write-Host ("Empty dirs found: {0}" -f ($empties | Measure-Object | Select-Object -ExpandProperty Count))

# ---------- _tmp vs root de-dupe by SHA256 ----------
Write-Host "`n=== _tmp duplicate cleanup (commit=$($Commit.IsPresent)) ==="
$tmpRoot = Join-Path $OutRoot '_tmp'
if (Test-Path $tmpRoot) {
  $minBytes = 25MB
  $outside = Get-ChildItem -Recurse -File $OutRoot -EA SilentlyContinue |
             Where-Object { $_.FullName -notlike "$tmpRoot\*" -and $_.Length -ge $minBytes }
  $inside  = Get-ChildItem -Recurse -File $tmpRoot -EA SilentlyContinue |
             Where-Object { $_.Length -ge $minBytes }

  Write-Host ("Hashing {0} outside + {1} inside files..." -f $outside.Count, $inside.Count)
  $outsideHashes = @{}
  foreach ($f in $outside) { try { $h=(Get-FileHash -Algorithm SHA256 -LiteralPath $f.FullName).Hash; $outsideHashes[$h]=$true } catch {} }

  $bytesToFree = 0; $dupes = @()
  foreach ($f in $inside) {
    try {
      $h = (Get-FileHash -Algorithm SHA256 -LiteralPath $f.FullName).Hash
      if ($outsideHashes.ContainsKey($h)) {
        $dupes += $f; $bytesToFree += $f.Length
        if ($Commit) { Remove-Item -LiteralPath $f.FullName -Force -WhatIf:$false }
        else         { Remove-Item -LiteralPath $f.FullName -Force -WhatIf }
      }
    } catch {}
  }
  Write-Host ("_tmp dupes found: {0}  | Potential freed: {1} GB" -f $dupes.Count, (Show-GB $bytesToFree))
} else {
  Write-Host "No _tmp folder present."
}

Write-Host "`n=== Done ==="
