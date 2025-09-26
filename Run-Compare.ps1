param(
  [Parameter(Mandatory=$true)] [string]$In,
  [Parameter(Mandatory=$true)] [string]$Vars,
  [Parameter(Mandatory=$true)] [string]$Out,
  [string]$VF = ".\out\autoframe_work\compare_autoframe.vf",
  [int]$FPS = 24,
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

$modulePath = "$PSScriptRoot\..\tools\Autoframe.psm1"
if (-not (Test-Path $modulePath)) {
  $modulePath = Join-Path $PSScriptRoot 'tools/Autoframe.psm1'
}
Import-Module $modulePath -Force

if (-not (Test-Path $In)) {
  throw "Input file not found: $In"
}

if (-not (Test-Path $Vars)) {
  throw "Vars file not found: $Vars"
}

if ([string]::IsNullOrWhiteSpace($Out)) {
  throw 'Output path cannot be empty.'
}

$vfArgs = @{
  VarsPath = $Vars
  fps = $FPS
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

$vfChain = New-VFChain @vfArgs

$filter = "[0:v]split=2[left][right];[right]$vfChain[right_portrait];[left][right_portrait]hstack=inputs=2,format=yuv420p"

$vfDir = Split-Path -Path $VF -Parent
if ($vfDir -and -not (Test-Path $vfDir)) {
  New-Item -ItemType Directory -Force -Path $vfDir | Out-Null
}

[IO.File]::WriteAllText($VF, $filter, [System.Text.UTF8Encoding]::new($false))

Write-Host "`n--- VF Chain ---"
Write-Host $vfChain
Write-Host "`n--- Filter Script ($VF) ---"
Write-Host $filter

$ffmpegArgs = @(
  '-hide_banner',
  '-y',
  '-nostdin',
  '-i', $In,
  '-filter_complex_script', $VF,
  '-c:v', 'libx264',
  '-crf', '20',
  '-preset', 'veryfast',
  '-movflags', '+faststart',
  '-an',
  $Out
)

& ffmpeg @ffmpegArgs
if ($LASTEXITCODE -ne 0) {
  throw "ffmpeg failed with exit code $LASTEXITCODE"
}
