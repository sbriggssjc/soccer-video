param(
  [ValidateSet("DryRun","Execute")]
  [string]$Mode = "DryRun",

  # Where to send moved files (created if missing)
  [string]$Trash = "out\_TRASH_{0:yyyy-MM-dd_HHmmss}" -f (Get-Date),

  # Optional: keep these exact relative paths no matter what
  [string[]]$KeepList = @(),

  # Optional: additional roots to scan for de-versioning (semicolon-separated)
  [string]$DevVersionRoots = "out\atomic_clips;out\portrait_reels;branded"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path ".").Path
$OutDir = Join-Path $Root "out"
$LogPath = Join-Path $OutDir ("cleanup_actions_{0:yyyyMMdd_HHmmss}.csv" -f (Get-Date))
$MovedBytes = 0
$MovedCount = 0
$Plan = @()

function Ensure-Dir($p) {
  $d = Split-Path -Parent $p
  if ($d -and -not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

function RelPath($abs) {
  $abs = [IO.Path]::GetFullPath($abs)
  return $abs.Replace([IO.Path]::GetFullPath($Root) + [IO.Path]::DirectorySeparatorChar, '').Replace('\','/')
}

function AbsPath($rel) {
  return (Join-Path $Root ($rel -replace '/','\'))
}

function Safe-Move($absSrc, $absDst) {
  if ($Mode -eq "DryRun") { return }
  Ensure-Dir $absDst
  Move-Item -Force -LiteralPath $absSrc -Destination $absDst
}

function Add-Plan($action, $relSrc, $relDst, $bytes, $reason) {
  $global:Plan += [PSCustomObject]@{
    action = $action
    src    = $relSrc
    dst    = $relDst
    bytes  = $bytes
    reason = $reason
  }
}

# --- Load CSVs if present ---
$DupCsvPath     = Join-Path $OutDir "duplicates.csv"
$OrphanCsvPath  = Join-Path $OutDir "orphan_signals.csv"

$DupRows    = @()
$OrphanRows = @()

if (Test-Path $DupCsvPath)    { $DupRows    = Import-Csv $DupCsvPath }
if (Test-Path $OrphanCsvPath) { $OrphanRows = Import-Csv $OrphanCsvPath }

# --- Normalize KeepList to a set of repo-relpaths ---
$KeepSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($k in $KeepList) { [void]$KeepSet.Add(($k -replace '\\','/')) }

# === Rule A: Exact duplicate clusters (keep representative, move the rest) ===
foreach ($row in $DupRows) {
  # Tolerate different header casings
  $type  = $row.type  ?? $row.Type
  $rep   = $row.representative_file ?? $row.representative ?? $row.Representative
  $mems  = $row.members ?? $row.Members

  if (-not $type -or -not $rep -or -not $mems) { continue }
  if ($type -ne "exact") { continue }

  $repRel = $rep.Trim().Replace('\','/')
  $members = ($mems -split ';|,') | ForEach-Object { $_.Trim() } | Where-Object { $_ }

  foreach ($m in $members) {
    $rel = $m.Replace('\','/')
    if ($rel -eq $repRel) { continue }
    if ($KeepSet.Contains($rel)) { continue }

    $abs = AbsPath $rel
    if (Test-Path $abs) {
      $size = (Get-Item $abs).Length
      $dstRel = "DUPES/" + ($rel -replace '[:\\/]','_')
      $dstAbs = Join-Path $Trash $dstRel
      Add-Plan "move" $rel $dstRel $size "duplicate_exact"
    }
  }
}

# === Rule B: Orphans with strong reasons -> move (safe; reversible) ===
# We’ll move only if reasons include "no_refs" or "unused" or "isolated".
$strongSignals = @("no_refs","unused","isolated","not_imported")
foreach ($row in $OrphanRows) {
  $rel = ($row.repo_relpath ?? $row.path ?? $row.file ?? "").Trim().Replace('\','/')
  if (-not $rel) { continue }
  if ($KeepSet.Contains($rel)) { continue }

  $reasons = ($row.reasons ?? $row.reason ?? "") -split ';|,'
  $reasons = $reasons | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ }

  $hasStrong = $false
  foreach ($r in $reasons) { if ($strongSignals -contains $r) { $hasStrong = $true; break } }
  if (-not $hasStrong) { continue }

  $abs = AbsPath $rel
  if (Test-Path $abs) {
    $size = (Get-Item $abs).Length
    $dstRel = "ORPHANS/" + $rel.Replace('/','_')
    $dstAbs = Join-Path $Trash $dstRel
    Add-Plan "move" $rel $dstRel $size ("orphan:" + ($reasons -join '|'))
  }
}

# === Rule C: De-version clip families -> keep highest __vN only ===
$roots = $DevVersionRoots -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
foreach ($rootRel in $roots) {
  $rootAbs = AbsPath $rootRel
  if (-not (Test-Path $rootAbs)) { continue }

  Get-ChildItem -Recurse -File -Path $rootAbs -Include *.mp4 | ForEach-Object {
    $fn = $_.Name
    $dirRel = RelPath $_.Directory.FullName
    if ($fn -match "^(?<base>.+?)__v(?<ver>\d+)\.mp4$") {
      $base = $Matches["base"]
      $siblings = Get-ChildItem -File -Path $_.Directory -Filter ($base + "__v*.mp4")
      $max = $siblings | Sort-Object { [int]([regex]::Match($_.Name, "__v(\d+)\.mp4$").Groups[1].Value) } -Descending | Select-Object -First 1
      foreach ($s in $siblings) {
        if ($s.FullName -ne $max.FullName) {
          $rel = Join-Path $dirRel $s.Name
          $rel = $rel.Replace('\','/')
          if ($KeepSet.Contains($rel)) { continue }
          $dstRel = "DEVERSION/" + $rel.Replace('/','_')
          Add-Plan "move" $rel $dstRel $s.Length "deversion_keep_latest"
        }
      }
    }
  }
}

# === Execute plan ===
if ($Plan.Count -eq 0) {
  Write-Host "No actions planned. Nothing to do."
  return
}

# Ensure Trash exists if executing
if ($Mode -eq "Execute") {
  if (-not (Test-Path $Trash)) { New-Item -ItemType Directory -Force -Path $Trash | Out-Null }
}

# Print summary & run
"{0} planned actions -> Mode: {1}" -f $Plan.Count, $Mode | Write-Host
foreach ($item in $Plan) {
  $srcAbs = AbsPath $item.src
  $dstAbs = Join-Path $Trash $item.dst
  if (Test-Path $srcAbs) {
    Write-Host ("{0}: {1} -> {2} [{3}] {4} bytes" -f $Mode, $item.src, ("$Trash\" + $item.dst), $item.reason, $item.bytes)
    Safe-Move $srcAbs $dstAbs
    if ($Mode -eq "Execute") {
      $global:MovedBytes += [int64]$item.bytes
      $global:MovedCount += 1
    }
  }
}

# Write action log (CSV)
Ensure-Dir $LogPath
$Plan | Export-Csv -NoTypeInformation -Path $LogPath -Encoding UTF8
if ($Mode -eq "Execute") {
  Write-Host ("Moved files: {0}   Reclaimed: {1:N2} MB" -f $MovedCount, ($MovedBytes/1MB))
  Write-Host ("Log written: {0}" -f $LogPath)
  Write-Host ("Trash dir  : {0}" -f (Resolve-Path $Trash))
} else {
  Write-Host ("DryRun only. Planned actions exported: {0}" -f $LogPath)
}
