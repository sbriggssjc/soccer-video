param([string]$Root="C:\\Users\\scott\\soccer-video\\out",[string]$PolicyPath=("$Root\\OutPolicy.psd1"),[switch]$Commit)
$P = Import-PowerShellDataFile -LiteralPath $PolicyPath

# 1) Remove banned portrait artifacts
$portrait = (Resolve-JunctionTarget (Join-Path $Root 'portrait_1080x1920'))
if ($portrait) {
  $ban = $P.BannedPortrait
  $cands = Get-ChildItem -LiteralPath $portrait -Recurse -File -Force -EA SilentlyContinue |
           Where-Object { $n=$_.FullName; $ban | Where-Object { $n -like $_ } }
  $gb = [math]::Round((($cands | Measure-Object Length -Sum).Sum)/1GB,2)
  Write-Host ("Portrait bans: {0} files (~{1} GB)" -f $cands.Count,$gb)
  if ($Commit) { $cands | Remove-Item -Force }
}

# 2) Prune old root backups
Prune-OldBackups -Root $Root -Keep $P.Prune.RootBackupsKeep @($Commit ? @{Commit=$true} : @{})

# 3) Remove empty dirs
if ($P.Prune.EmptyDirs) {
  $empty = Get-ChildItem -LiteralPath $Root -Recurse -Directory |
           Where-Object { @(Get-ChildItem -LiteralPath $_.FullName -Force -EA SilentlyContinue).Count -eq 0 }
  Write-Host ("Empty dirs: {0}" -f $empty.Count)
  if ($Commit) { $empty | Remove-Item -Force }
}

# 4) Expire _tmp files
if ($P.Prune.TmpDaysOld -gt 0) {
  $tmp = Join-Path $Root '_tmp'
  if (Test-Path $tmp) {
    $old = Get-ChildItem -LiteralPath $tmp -Recurse -File -EA SilentlyContinue |
           Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$P.Prune.TmpDaysOld) }
    Write-Host ("_tmp old files: {0}" -f $old.Count)
    if ($Commit) { $old | Remove-Item -Force }
  }
}

# 5) Expire quarantine leftovers
if ($P.Prune.QuarantineDays -gt 0) {
  $qr = Join-Path $Root '_quarantine'
  if (Test-Path $qr) {
    $old = Get-ChildItem -LiteralPath $qr -Recurse -File -EA SilentlyContinue |
           Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$P.Prune.QuarantineDays) }
    Write-Host ("Quarantine old files: {0}" -f $old.Count)
    if ($Commit) { $old | Remove-Item -Force }
  }
}

# 6) Size rollup (eyes-on)
Out-Null; powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\..\OutTidyAndReport.ps1" -OutRoot $Root -Top 25
