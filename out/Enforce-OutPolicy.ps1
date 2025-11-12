param(
  [string]$Root="C:\\Users\\scott\\soccer-video\\out",
  [string]$PolicyPath=("$Root\\OutPolicy.psd1"),
  [switch]$Commit
)

# Helper: match any wildcard in a set
function Test-MatchesAny([string]$Path,[string[]]$Patterns){
  foreach($p in $Patterns){ if($Path -like $p){ return $true } }
  return $false
}

# Load policy
$P = Import-PowerShellDataFile -LiteralPath $PolicyPath

# 1) Remove banned portrait artifacts
$portrait = (Resolve-JunctionTarget (Join-Path $Root 'portrait_1080x1920'))
if ($portrait) {
  $ban = @($P.BannedPortrait)
  $cands = Get-ChildItem -LiteralPath $portrait -Recurse -File -Force -EA SilentlyContinue |
           Where-Object { Test-MatchesAny -Path $_.FullName -Patterns $ban }
  $gb = [math]::Round((($cands | Measure-Object Length -Sum).Sum)/1GB,2)
  Write-Host ("Portrait bans: {0} files (~{1} GB)" -f $cands.Count,$gb)
  if ($Commit -and $cands.Count) { $cands | Remove-Item -Force }
}

# 2) Prune old root backups (PowerShell 5 friendly splat)
$pruneSplat = @{ Root = $Root; Keep = $P.Prune.RootBackupsKeep }
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

# 6) Size rollup (eyes-on)
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot '..\\OutTidyAndReport.ps1') -OutRoot $Root -Top 25
