param(
  [string]$Root="C:\Users\scott\soccer-video\out",
  [string]$PolicyPath = ($Root + "\OutPolicy.psd1"),
  [switch]$Commit
)

Import-Module "$PSScriptRoot\..\tools\OutHousekeeping.psm1" -Force

# Load policy
$P = Import-PowerShellDataFile -LiteralPath $PolicyPath

# 1) Remove banned portrait artifacts
$portraitReport = Get-NoisyPortraitCandidates -Root $Root -Patterns @($P.BannedPortrait)
if ($portraitReport) {
  Show-PortraitReports -PortraitReport $portraitReport -Commit:$Commit | Out-Null
}

# 2) Prune old root backups (PS5-friendly splat)
$pruneSplat = @{ Root = $Root; Keep = [int]$P.Prune.RootBackupsKeep }
if ($Commit){ $pruneSplat['Commit'] = $true }
Prune-OldBackups @pruneSplat

# 3) Remove empty dirs
if ($P.Prune.EmptyDirs) {
  $empty = Get-ChildItem -LiteralPath $Root -Recurse -Directory -Force -EA SilentlyContinue |
           Where-Object { @(Get-ChildItem -LiteralPath $_.FullName -Force -EA SilentlyContinue).Count -eq 0 }
  Write-Host ("Empty dirs: {0}" -f $empty.Count)
  if ($Commit -and $empty.Count){ $empty | Remove-Item -Force }
}

# 4) Expire _tmp files
if ($P.Prune.TmpDaysOld -gt 0) {
  $tmp = Join-Path $Root '_tmp'
  if (Test-Path $tmp) {
    $old = Get-ChildItem -LiteralPath $tmp -Recurse -File -Force -EA SilentlyContinue |
           Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-[int]$P.Prune.TmpDaysOld) }
    Write-Host ("_tmp old files: {0}" -f $old.Count)
    if ($Commit -and $old.Count){ $old | Remove-Item -Force }
  }
}

# 5) Expire quarantine leftovers
if ($P.Prune.QuarantineDays -gt 0) {
  $qr = Join-Path $Root '_quarantine'
  if (Test-Path $qr) {
    $old = Get-ChildItem -LiteralPath $qr -Recurse -File -Force -EA SilentlyContinue |
           Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-[int]$P.Prune.QuarantineDays) }
    Write-Host ("Quarantine old files: {0}" -f $old.Count)
    if ($Commit -and $old.Count){ $old | Remove-Item -Force }
  }
}

# 6) One-pass report (re-uses your existing script)
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot '..\OutTidyAndReport.ps1') -OutRoot $Root -Top 25
