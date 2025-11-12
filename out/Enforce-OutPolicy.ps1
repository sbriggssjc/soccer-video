param(
  [string]$Root="C:\Users\scott\soccer-video\out",
  [string]$PolicyPath = ($Root + "\OutPolicy.psd1"),
  [switch]$Commit
)

# ---------- Local helpers (PS5-safe) ----------
# Only define if not already present in session
if (-not (Get-Command Resolve-JunctionTarget -ErrorAction SilentlyContinue)) {
  function Resolve-JunctionTarget {
    param([Parameter(Mandatory)][string]$Path)
    try {
      $it = Get-Item -LiteralPath $Path -Force
      $t  = $null
      try { $t = $it.Target } catch {}
      if ($t -and (Test-Path -LiteralPath $t)) { return $t }
      $guess = Join-Path (Split-Path $Path -Parent) 'reels\portrait_1080x1920'
      if (Test-Path -LiteralPath $guess) { return $guess }
    } catch {}
    return $null
  }
}

if (-not (Get-Command Prune-OldBackups -ErrorAction SilentlyContinue)) {
  function Prune-OldBackups {
    param(
      [string]$Root = "C:\Users\scott\soccer-video\out",
      [int]$Keep = 3,
      [switch]$Commit
    )
    $dirs = Get-ChildItem -LiteralPath $Root -Force -Directory -EA SilentlyContinue |
            Where-Object { $_.Name -like "root_backups_*" } |
            Sort-Object LastWriteTime -Desc
    if ($dirs.Count -le $Keep) { Write-Host "Nothing to prune. Found $($dirs.Count), keeping $Keep."; return }
    $toRemove = $dirs | Select-Object -Skip $Keep
    $bytes = 0
    foreach($d in $toRemove){
      $sum = (Get-ChildItem -LiteralPath $d.FullName -Recurse -File -Force -EA SilentlyContinue | Measure-Object Length -Sum).Sum
      if ($sum) { $bytes += $sum }
    }
    $gb = [math]::Round($bytes/1GB,2)
    Write-Host ("Would remove {0} backup dirs (~{1} GB):" -f $toRemove.Count, $gb)
    $toRemove | Select Name,LastWriteTime | Format-Table -Auto
    if ($Commit) {
      $toRemove | Remove-Item -Recurse -Force
      Write-Host "Pruned."
    } else {
      Write-Host "Re-run with -Commit to actually delete."
    }
  }
}

function Test-MatchesAny([string]$Path,[string[]]$Patterns){
  foreach($p in $Patterns){ if($Path -like $p){ return $true } }
  return $false
}
# ----------------------------------------------

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
