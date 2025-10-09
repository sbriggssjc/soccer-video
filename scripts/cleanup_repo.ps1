Param(
    [Parameter(Mandatory=$true)] [string]$TargetDir,
    [switch]$WhatIf
)
$ErrorActionPreference = "Stop"
if (!(Test-Path $TargetDir)) { throw "TargetDir not found: $TargetDir" }

# Keep only canonical outputs: *__OPENER.mp4
$toRemove = Get-ChildItem -Path $TargetDir -File -Recurse |
    Where-Object { $_.Extension -in ".mp4",".mov",".mkv" -and $_.Name -notmatch "__OPENER.mp4$" }

if ($WhatIf) {
    Write-Host "Would remove $($toRemove.Count) files:" -ForegroundColor Yellow
    $toRemove | ForEach-Object { Write-Host " $_" }
} else {
    $toRemove | Remove-Item -Force
    Write-Host "Removed $($toRemove.Count) non-canonical video files from $TargetDir"
}
