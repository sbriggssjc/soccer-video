# Wrapper so we can run:  .\enhance.ps1 "path\to\fileOrFolder" [-Recurse] [-Profile rec709_smart|rec709_basic|punchy] [-Crf 20] [-Preset veryfast]
param(
  [Parameter(Mandatory=$true)][string]$In,
  [string]$Profile = "rec709_smart",
  [int]$Crf = 20,
  [string]$Preset = "veryfast",
  [switch]$Recurse
)
$script = Join-Path $PSScriptRoot "tools\auto_enhance\auto_enhance.ps1"
& $script -In $In -Profile $Profile -Crf $Crf -Preset $Preset -Recurse:$Recurse
