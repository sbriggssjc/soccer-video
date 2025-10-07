param(
  [ValidateSet("DryRun","Execute")]
  [string]$Mode = "DryRun",
  [string]$Trash = ("out\_TRASH_{0:yyyy-MM-dd_HHmmss}" -f (Get-Date)),
  [string[]]$KeepList = @(),
  [string]$DevVersionRoots = "out\atomic_clips;out\portrait_reels;branded"
)

$ErrorActionPreference = "Stop"
$Root   = (Resolve-Path ".").Path
$OutDir = Join-Path $Root "out"
$LogPath = Join-Path $OutDir ("cleanup_actions_{0:yyyyMMdd_HHmmss}.csv" -f (Get-Date))

# Use a strong, append-safe list for the plan (PS 5.1 friendly)
$script:Plan = New-Object System.Collections.Generic.List[object]
$script:MovedBytes = 0
$script:MovedCount = 0

function Ensure-Dir($p) {
  $d = Split-Path -Parent $p
  if ($d -and -not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}
function RelPath($abs) {
  $abs = [IO.Path]::GetFullPath($abs)
  return $abs.Replace([IO.Path]::GetFullPath($Root) + [IO.Path]::DirectorySeparatorChar, '').Replace('\','/')
}
function AbsPath($rel) { return (Join-Path $Root ($rel -replace '/','\')) }
function Safe-Move($absSrc, $absDst) { if ($Mode -ne "DryRun") { Ensure-Dir $absDst; Move-Item -Force -LiteralPath $absSrc -Destination $absDst } }
function Add-Plan($action,$relSrc,$relDst,$bytes,$reason) {
  $obj = [PSCustomObject]@{ action=$action; src=$relSrc; dst=$relDst; bytes=[int64]$bytes; reason=$reason }
  $script:Plan.Add($obj) | Out-Null
}
function Get-Col($row, [string[]]$names) {
  foreach ($n in $names) {
    $prop = $row.PSObject.Properties[$n]
    if ($prop -and $null -ne $prop.Value -and "$($prop.Value)".Trim() -ne "") { return "$($prop.Value)" }
  }
  return ""
}

# Load inputs (no ternary/??)
$DupCsvPath    = Join-Path $OutDir "duplicates.csv"
$OrphanCsvPath = Join-Path $OutDir "orphan_signals.csv"
$DupRows    = @()
$OrphanRows = @()
if (Test-Path $DupCsvPath)    { $DupRows    = Import-Csv $DupCsvPath }
if (Test-Path $OrphanCsvPath) { $OrphanRows = Import-Csv $OrphanCsvPath }

# Keep set
$KeepSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($k in $KeepList) { [void]$KeepSet.Add(($k -replace '\\','/')) }

# Rule A: exact duplicate clusters
foreach ($row in $DupRows) {
  $type = Get-Col $row @('type','Type')
  $rep  = Get-Col $row @('representative_file','representative','Representative')
  $mems = Get-Col $row @('members','Members')
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
      Add-Plan "move" $rel $dstRel $size "duplicate_exact"
    }
  }
}

# Rule B: strong-signal orphans
$strongSignals = @("no_refs","unused","isolated","not_imported")
foreach ($row in $OrphanRows) {
  $rel = Get-Col $row @('repo_relpath','path','file'); if (-not $rel) { continue }
  $rel = $rel.Trim().Replace('\','/')
  if ($KeepSet.Contains($rel)) { continue }
  $reasonsStr = Get-Col $row @('reasons','reason','Reasons','Reason')
  $reasons = @()
  if ($reasonsStr) { $reasons = ($reasonsStr -split ';|,') | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ } }
  $hasStrong = $false
  foreach ($r in $reasons) { if ($strongSignals -contains $r) { $hasStrong = $true; break } }
  if (-not $hasStrong) { continue }
  $abs = AbsPath $rel
  if (Test-Path $abs) {
    $size = (Get-Item $abs).Length
    $dstRel = "ORPHANS/" + $rel.Replace('/','_')
    Add-Plan "move" $rel $dstRel $size ("orphan:" + ($reasons -join '|'))
  }
}

# Rule C: de-version clip families (keep highest __vN)
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
        if ($max -and $s.FullName -ne $max.FullName) {
          $rel = (Join-Path $dirRel $s.Name).Replace('\','/')
          if ($KeepSet.Contains($rel)) { continue }
          $dstRel = "DEVERSION/" + $rel.Replace('/','_')
          Add-Plan "move" $rel $dstRel $s.Length "deversion_keep_latest"
        }
      }
    }
  }
}

# Execute
if ($script:Plan.Count -eq 0) { Write-Host "No actions planned. Nothing to do."; return }
if ($Mode -eq "Execute" -and -not (Test-Path $Trash)) { New-Item -ItemType Directory -Force -Path $Trash | Out-Null }

("{0} planned actions -> Mode: {1}" -f $script:Plan.Count, $Mode) | Write-Host
foreach ($item in $script:Plan) {
  $srcAbs = AbsPath $item.src
  $dstAbs = Join-Path $Trash $item.dst
  if (Test-Path $srcAbs) {
    Write-Host ("{0}: {1} -> {2} [{3}] {4} bytes" -f $Mode, $item.src, ("$Trash\" + $item.dst), $item.reason, $item.bytes)
    Safe-Move $srcAbs $dstAbs
    if ($Mode -eq "Execute") {
      $script:MovedBytes += [int64]$item.bytes
      $script:MovedCount += 1
    }
  }
}

Ensure-Dir $LogPath
$script:Plan | Export-Csv -NoTypeInformation -Path $LogPath -Encoding UTF8
if ($Mode -eq "Execute") {
  Write-Host ("Moved files: {0}   Reclaimed: {1:N2} MB" -f $script:MovedCount, ($script:MovedBytes/1MB))
  Write-Host ("Log written: {0}" -f $LogPath)
  Write-Host ("Trash dir  : {0}" -f (Resolve-Path $Trash))
} else {
  Write-Host ("DryRun only. Planned actions exported: {0}" -f $LogPath)
}
