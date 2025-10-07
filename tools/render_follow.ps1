param(
  [Parameter(Mandatory=$true)] [string]$In,
  [string]$Out,
  [ValidateSet("Cinematic","Gentle","RealZoom")]
  [string]$Preset = "Cinematic",
  [switch]$Flip180,
  [switch]$Portrait,
  [int]$Fps = 30,
  [string]$ExtraArgs = ""
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ".").Path

function Find-Python {
  $py = Get-Command python -ErrorAction SilentlyContinue
  if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
  if (-not $py) { throw "Python not found. Install Python 3.10+ or add it to PATH." }
  return $py.Source
}

function Find-Script([string]$name) {
  $candidates = @(
    (Join-Path $root $name),
    (Join-Path $root ("scripts\" + $name)),
    (Join-Path $root ("archive\scripts\2025-10-07\" + $name))
  )
  foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
  return $null
}

$map = @{
  "Cinematic" = @("render_follow_autoz_cinematic.py","render_follow_cinematic.py","render_follow.py")
  "Gentle"    = @("render_follow_autoz_gentle.py","render_follow_gentle.py","render_follow.py")
  "RealZoom"  = @("render_follow_autoz_realzoom.py","render_follow_realzoom.py","render_follow.py")
}

$chosen = $null
foreach ($cand in $map[$Preset]) {
  $p = Find-Script $cand
  if ($p) { $chosen = $p; break }
}
if (-not $chosen) {
  throw "Could not find any render_follow script for preset '$Preset'. Looked for: $($map[$Preset] -join ', ')"
}

$py = Find-Python
$inAbs = (Resolve-Path $In).Path
if (-not $Out) {
  $Out = [IO.Path]::ChangeExtension($inAbs, $null) + ("__{0}.mp4" -f $Preset.ToUpper())
}
$outAbs = $Out

$flags = @()
if ($Flip180) { $flags += @("--flip180") }
if ($Portrait){ $flags += @("--portrait", "1080x1920") }
$flags += @("--fps", "$Fps")

if ($ExtraArgs) {
  $flags += $ExtraArgs.Split(' ')
}

Write-Host "Preset: $Preset"
Write-Host "Script: $chosen"
Write-Host "In    : $inAbs"
Write-Host "Out   : $outAbs"
Write-Host "Args  : $($flags -join ' ')"

& $py $chosen --in "$inAbs" --out "$outAbs" @flags
if ($LASTEXITCODE -ne 0) { throw "Render failed (exit $LASTEXITCODE)" }

Write-Host "Done -> $outAbs"
