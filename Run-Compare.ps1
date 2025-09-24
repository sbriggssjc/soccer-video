param(
  [string]$In  = (Join-Path (Join-Path $PSScriptRoot 'recipes/compare') 'samples/clip.mp4'),
  [string]$Vars = (Join-Path (Join-Path $PSScriptRoot 'recipes/compare') 'samples/clip_zoom.ps1vars'),
  [string]$Out  = (Join-Path (Join-Path $PSScriptRoot 'recipes/compare') 'samples/out.mp4'),
  [string]$VF   = (Join-Path (Join-Path $PSScriptRoot 'recipes/compare') 'samples/tmp.vf')
)

$scriptPath = Join-Path $PSScriptRoot 'recipes/compare/Run-Compare.ps1'
if (-not (Test-Path -LiteralPath $scriptPath)) {
  throw "Unable to locate recipe runner at $scriptPath"
}

& $scriptPath -In $In -Vars $Vars -Out $Out -VF $VF
