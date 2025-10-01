# Run the SMART analyzer on a file or folder.
param(
  [Parameter(Mandatory=$true)][string]$In,
  [double]$SampleStart = 1.0,
  [double]$SampleDur   = 8.0,
  [int]$Crf = 20,
  [string]$Preset = "veryfast",
  [switch]$Recurse,
  [switch]$NoWB,     # set to disable whitebalance attempts
  [switch]$DryRun
)
$script = Join-Path $PSScriptRoot "tools\auto_enhance\auto_enhance_smart.ps1"
& $script -In $In -SampleStart $SampleStart -SampleDur $SampleDur -Crf $Crf -Preset $Preset -Recurse:$Recurse -TryWB:(!$NoWB) -DryRun:$DryRun
