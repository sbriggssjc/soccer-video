[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$catalogDir = Join-Path $root 'out\catalog'
$sidecarDir = Join-Path $catalogDir 'sidecar'

if (-not (Test-Path $catalogDir)) {
    New-Item -ItemType Directory -Path $catalogDir | Out-Null
}
if (-not (Test-Path $sidecarDir)) {
    New-Item -ItemType Directory -Path $sidecarDir | Out-Null
}

Write-Host 'Rebuilding atomic clip catalog...' -ForegroundColor Cyan

$python = 'python'
$script = Join-Path $root 'tools\catalog.py'

$output = & $python $script --rebuild-atomic-index
$output | ForEach-Object { Write-Host $_ }

if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Catalog rebuild completed with probe failures.'
    exit 1
}

Write-Host 'Catalog refresh complete.' -ForegroundColor Green
exit 0

