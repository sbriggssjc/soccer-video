[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = 'python'
$script = Join-Path $root 'tools\catalog.py'

Write-Host 'Scanning atomic clips and refreshing catalogs...' -ForegroundColor Cyan

$process = & $python $script --scan-atomic
Write-Host $process

Write-Host 'Catalog refresh complete.' -ForegroundColor Green
