[CmdletBinding()]
param(
  [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
  [string]$VaultRoot = "C:\Users\scott\OneDrive\SoccerVideoMedia",
  [int]$Days = 14,
  [string]$TaskName = "SoccerVideo_OneDrive_Offload"
)

$ErrorActionPreference = "Stop"

$ps = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$script = Join-Path $RepoRoot "tools\Offload-OneDriveMedia.ps1"
if (-not (Test-Path -LiteralPath $script)){ throw "Missing script: $script" }

# Weekly: Sunday 3:30 AM
$actionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -VaultRoot `"$VaultRoot`" -Days $Days -RepoRoot `"$RepoRoot`""
$action = New-ScheduledTaskAction -Execute $ps -Argument $actionArgs
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3:30AM

# Run with highest privileges helps with filesystem attribute toggles
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Force | Out-Null

Write-Host "Installed scheduled task: $TaskName"
Write-Host "To run it now: Start-ScheduledTask -TaskName $TaskName"
Write-Host "To remove it:  Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
