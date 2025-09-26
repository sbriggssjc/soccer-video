param(
    [Parameter(Mandatory=$true)][string]$In,
    [Parameter(Mandatory=$true)][string]$Out,
    [Parameter(Mandatory=$true)][Alias('Vars')][string]$Vars,
    [Parameter(Mandatory=$true)][int]$Fps,
    [double]$GoalLeft,
    [double]$GoalRight,
    [double]$PadPx,
    [double]$StartRampSec,
    [double]$TGoalSec,
    [double]$CeleSec,
    [double]$CeleTight,
    [double]$YMin,
    [double]$YMax,
    [switch]$ScaleFirst
)

$ErrorActionPreference = 'Stop'

Import-Module "$PSScriptRoot\..\tools\Autoframe.psm1" -Force

$vfArgs = @{
    VarsPath = $Vars
    fps = $Fps
}

if ($PSBoundParameters.ContainsKey('GoalLeft')) { $vfArgs.goalLeft = $GoalLeft }
if ($PSBoundParameters.ContainsKey('GoalRight')) { $vfArgs.goalRight = $GoalRight }
if ($PSBoundParameters.ContainsKey('PadPx')) { $vfArgs.padPx = $PadPx }
if ($PSBoundParameters.ContainsKey('StartRampSec')) { $vfArgs.startRampSec = $StartRampSec }
if ($PSBoundParameters.ContainsKey('TGoalSec')) { $vfArgs.tGoalSec = $TGoalSec }
if ($PSBoundParameters.ContainsKey('CeleSec')) { $vfArgs.celeSec = $CeleSec }
if ($PSBoundParameters.ContainsKey('CeleTight')) { $vfArgs.celeTight = $CeleTight }
if ($PSBoundParameters.ContainsKey('YMin')) { $vfArgs.yMin = $YMin }
if ($PSBoundParameters.ContainsKey('YMax')) { $vfArgs.yMax = $YMax }
if ($ScaleFirst.IsPresent) { $vfArgs.ScaleFirst = $true }

$vf = New-VFChain @vfArgs

if ([string]::IsNullOrWhiteSpace($Out)) {
    throw 'Output path cannot be empty.'
}

$ffmpegArgs = @(
    '-y',
    '-i', $In,
    '-vf', $vf,
    '-c:v', 'libx264',
    '-crf', '19',
    '-preset', 'veryfast',
    $Out
)

& ffmpeg @ffmpegArgs
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg failed with exit code $LASTEXITCODE"
}
