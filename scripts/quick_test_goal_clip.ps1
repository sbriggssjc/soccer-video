param(
  [string]$In = '.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4',
  [string]$Vars = '.\out\autoframe_work\004__GOAL__t266.50-t283.10_zoom.ps1vars',
  [string]$Out = '.\out\reels\tiktok\COMPARE__004__pipeline_apply.mp4',
  [int]$Fps = 24,
  [int]$GoalLeft = 840,
  [int]$GoalRight = 1080,
  [int]$PadPx = 40,
  [double]$StartRampSec = 4.0,
  [double]$TGoalSec = 13.0,
  [double]$CeleSec = 2.6,
  [double]$CeleTight = 1.80,
  [double]$YMin = 360,
  [double]$YMax = 780,
  [switch]$ScaleFirst
)

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

Write-Host "VF: $vf"
ffmpeg -y -i $In -vf $vf -c:v libx264 -crf 19 -preset veryfast $Out
