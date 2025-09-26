param(
  [Parameter(Mandatory=$true)][string]$In,
  [Parameter(Mandatory=$true)][string]$Vars,
  [string]$Out = '',
  [int]$Fps = 24,
  [int]$GoalLeft = 840,
  [int]$GoalRight = 1080,
  [int]$PadPx = 40,
  [double]$StartRampSec = 3.0,
  [double]$TGoalSec = 10.0,
  [double]$CeleSec = 2.0,
  [double]$CeleTight = 1.80,
  [double]$YMin = 360,
  [double]$YMax = 780,
  [switch]$ScaleFirst
)

if (-not (Test-Path $In)) {
    throw "Input file not found: $In"
}

if (-not (Test-Path $Vars)) {
    throw "Vars file not found: $Vars"
}

if ([string]::IsNullOrWhiteSpace($Out)) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($In)
    $dir = Split-Path $In -Parent
    if (-not $dir) { $dir = '.' }
    $Out = Join-Path $dir ($stem + '.goalzoom.mp4')
}

Import-Module "$PSScriptRoot\..\tools\Autoframe.psm1" -Force

$vf = New-VFChain `
    -Vars $Vars `
    -fps $Fps `
    -goalLeft $GoalLeft `
    -goalRight $GoalRight `
    -padPx $PadPx `
    -startRampSec $StartRampSec `
    -tGoalSec $TGoalSec `
    -celeSec $CeleSec `
    -celeTight $CeleTight `
    -yMin $YMin `
    -yMax $YMax `
    -ScaleFirst:$ScaleFirst

if ([string]::IsNullOrWhiteSpace($Out)) {
    throw 'Output path cannot be empty.'
}

& ffmpeg -y -i $In -vf $vf -c:v libx264 -crf 19 -preset veryfast $Out
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg failed with exit code $LASTEXITCODE"
}
