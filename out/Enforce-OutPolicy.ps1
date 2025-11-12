param(
  [string]$Root="C:\Users\scott\soccer-video\out",
  [string]$PolicyPath = ($Root + "\OutPolicy.psd1"),
  [switch]$Commit
)

Import-Module "$PSScriptRoot\..\tools\OutHousekeeping.psm1" -Force

Invoke-OutPolicy -Root $Root -PolicyPath $PolicyPath -Commit:$Commit
