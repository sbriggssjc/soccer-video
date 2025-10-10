[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Build a fresh file list for cataloging (recursive; ignore MASTER and _quarantine)
$repoRoot = $root
$atomicDir = Join-Path $repoRoot 'out\atomic_clips'
$listFile = Join-Path $atomicDir 'list.txt'

if (-not (Test-Path $atomicDir)) {
    throw "Atomic clips directory not found: $atomicDir"
}

Get-ChildItem $atomicDir -Recurse -File -Include *.mp4,*.mov |
  Where-Object { $_.Name -notmatch '^(MASTER|master)\.(mp4|mov)$' -and $_.FullName -notmatch '\\_quarantine(\\|$)' } |
  ForEach-Object { $_.FullName } | Set-Content -Encoding UTF8 $listFile

Write-Host 'Scanning atomic clips and refreshing catalogs...' -ForegroundColor Cyan
$clipCount = (Get-Content $listFile).Count
Write-Host ("Found {0} clip(s) to catalog (recursive)." -f $clipCount)

$python = 'python'
$script = Join-Path $root 'tools\catalog.py'

$process = & $python $script --scan-list $listFile
Write-Host $process

Write-Host 'Catalog refresh complete.' -ForegroundColor Green
