# Per-shot SMART enhancer runner
param(
  [Parameter(Mandatory=$true)][string]$In,
  [double]$SceneThresh = 0.30,
  [double]$MinShotSec  = 2.0,
  [int]$MaxShots       = 180,
  [double]$SampleStart = 0.2,
  [double]$SampleDur   = 4.0,
  [int]$Crf = 20,
  [string]$Preset = "veryfast",
  [switch]$Recurse,
  [switch]$NoWB,
  [switch]$DryRun
)
$script = Join-Path $PSScriptRoot "tools\auto_enhance\auto_enhance_smartshots.ps1"
& $script -In $In -SceneThresh $SceneThresh -MinShotSec $MinShotSec -MaxShots $MaxShots `
  -SampleStart $SampleStart -SampleDur $SampleDur -Crf $Crf -Preset $Preset -Recurse:$Recurse -TryWB:(!$NoWB) -DryRun:$DryRun
