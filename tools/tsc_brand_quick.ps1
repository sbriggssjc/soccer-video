param(
  [Parameter(Mandatory=$true)][string]$In,
  [Parameter(Mandatory=$true)][string]$Out,
  [string]$Title = ''
)

$ErrorActionPreference = 'Stop'

& "$PSScriptRoot\tsc_brand.ps1" -In $In -Out $Out -Title $Title -Watermark -EndCard -Aspect '16x9'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
