param(
  [string]$Dir = ".\out\autoframe_work",
  [int]$Fps = 24, [int]$GoalLeft = 840, [int]$GoalRight = 1080,
  [int]$PadPx = 40, [double]$StartRampSec = 4.0,
  [double]$TGoalSec = 13.0, [double]$CeleSec = 2.6, [double]$CeleTight = 1.8,
  [double]$YMin = 360, [double]$YMax = 780, [switch]$ScaleFirst
)
Import-Module "$PSScriptRoot\..\tools\Autoframe.psm1" -Force
$bad = @()
Get-ChildItem $Dir -Filter *.ps1vars | ForEach-Object {
  try {
    $vf = New-VFChain -VarsPath $_.FullName -fps $Fps -goalLeft $GoalLeft -goalRight $GoalRight `
      -padPx $PadPx -startRampSec $StartRampSec -tGoalSec $TGoalSec -celeSec $CeleSec `
      -celeTight $CeleTight -yMin $YMin -yMax $YMax -ScaleFirst:$ScaleFirst
    if ($vf -match 'clip\(' -or $vf -match '(^|[^a-zA-Z])t([^a-zA-Z]|$)') {
      $bad += $_.Name
      Write-Warning "VF contains disallowed token(s): $($_.Name)"
    }
  } catch { $bad += $_.Name; Write-Warning "Failed VF: $($_.Name) — $($_.Exception.Message)" }
}
if ($bad.Count -eq 0) { Write-Host "All good ✅" -ForegroundColor Green } else { Write-Host "Issues in: $($bad -join ', ')" -ForegroundColor Yellow }
