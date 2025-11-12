function Resolve-JunctionTarget {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$Path
  )
  try {
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    $target = $null
    try { $target = $item.Target } catch {}
    if ($target -and (Test-Path -LiteralPath $target)) { return $target }
    $guess = Join-Path (Split-Path $Path -Parent) 'reels\portrait_1080x1920'
    if (Test-Path -LiteralPath $guess) { return $guess }
  } catch {}
  return $null
}

function Get-NoisyPortraitCandidates {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$Root,
    [string]$PortraitSubPath = 'portrait_1080x1920',
    [string[]]$Patterns = @()
  )
  $portraitTarget = Resolve-JunctionTarget (Join-Path $Root $PortraitSubPath)
  if (-not $portraitTarget) { return $null }
  if (-not $Patterns -or $Patterns.Count -eq 0) {
    return [pscustomobject]@{
      PortraitRoot = $portraitTarget
      Candidates   = @()
    }
  }
  try {
    $files = Get-ChildItem -LiteralPath $portraitTarget -Recurse -File -Force -EA SilentlyContinue |
             Where-Object {
               $fullName = $_.FullName
               foreach ($pattern in $Patterns) {
                 if ($fullName -like $pattern) { return $true }
               }
               return $false
             }
  } catch {
    $files = @()
  }
  return [pscustomobject]@{
    PortraitRoot = $portraitTarget
    Candidates   = @($files)
  }
}

function Get-PortraitReport {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$Root,
    [string]$PortraitSubPath = 'portrait_1080x1920',
    [string[]]$Patterns = @(),
    [switch]$Commit
  )
  $portraitReport = Get-NoisyPortraitCandidates -Root $Root -PortraitSubPath $PortraitSubPath -Patterns $Patterns
  if (-not $portraitReport) { return $null }
  $candidates = @($portraitReport.Candidates)
  $count = $candidates.Count
  $bytes = 0
  if ($count -gt 0) {
    $sum = ($candidates | Measure-Object Length -Sum).Sum
    if ($sum) { $bytes = $sum }
  }
  $gb = [math]::Round($bytes/1GB,2)
  Write-Host ("Portrait bans: {0} files (~{1} GB)" -f $count, $gb)
  if ($Commit -and $count -gt 0) {
    $candidates | Remove-Item -Force
  }
  return [pscustomobject]@{
    PortraitRoot = $portraitReport.PortraitRoot
    Candidates   = $candidates
    Count        = $count
    SizeBytes    = $bytes
  }
}

function Clear-OldBackups {
  [CmdletBinding()]
  param(
    [string]$Root = "C:\\Users\\scott\\soccer-video\\out",
    [int]$Keep = 3,
    [switch]$Commit
  )
  $dirs = Get-ChildItem -LiteralPath $Root -Force -Directory -EA SilentlyContinue |
          Where-Object { $_.Name -like "root_backups_*" } |
          Sort-Object LastWriteTime -Desc
  if ($dirs.Count -le $Keep) {
    Write-Host "Nothing to prune. Found $($dirs.Count), keeping $Keep."
    return @()
  }
  $toRemove = $dirs | Select-Object -Skip $Keep
  $bytes = 0
  foreach ($d in $toRemove) {
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
  return $toRemove
}

function Restore-FromManifest {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$ManifestPath,
    [Parameter(Mandatory)][string]$DestinationRoot,
    [string]$SourceRoot,
    [switch]$Commit
  )
  if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
  }
  if (-not (Test-Path -LiteralPath $DestinationRoot)) {
    throw "Destination root not found: $DestinationRoot"
  }
  $canonicalRoot = "C:\\Users\\scott\\soccer-video\\out"
  $rows = Import-Csv -LiteralPath $ManifestPath
  $entries = foreach ($row in $rows) {
    foreach ($column in 'stabilized_path','brand_path','post_path') {
      $value = $row.$column
      if ([string]::IsNullOrWhiteSpace($value)) { continue }
      $normalized = $value -replace '/', '\\'
      $relative = if ($normalized -like "$canonicalRoot*") {
        $normalized.Substring($canonicalRoot.Length).TrimStart('\\')
      } else {
        $normalized
      }
      if (-not [string]::IsNullOrWhiteSpace($relative)) {
        [pscustomobject]@{
          RelativePath = $relative
          Destination  = Join-Path $DestinationRoot $relative
          Source       = if ($SourceRoot) { Join-Path $SourceRoot $relative } else { $null }
        }
      }
    }
  }
  $missing = $entries | Where-Object { -not (Test-Path -LiteralPath $_.Destination) }
  Write-Host ("Manifest entries: {0} | Missing at destination: {1}" -f $entries.Count, $missing.Count)
  if ($Commit) {
    if (-not $SourceRoot) {
      Write-Warning "Commit requested but no SourceRoot provided. Skipping copy operations."
    } else {
      foreach ($entry in $missing) {
        if (-not (Test-Path -LiteralPath $entry.Source)) {
          Write-Warning ("Source missing: {0}" -f $entry.Source)
          continue
        }
        $destDir = Split-Path $entry.Destination -Parent
        if ($destDir -and -not (Test-Path -LiteralPath $destDir)) {
          New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Copy-Item -LiteralPath $entry.Source -Destination $entry.Destination -Force
      }
    }
  }
  return $missing
}

function Invoke-OutPolicy {
  [CmdletBinding()]
  param(
    [string]$Root = "C:\\Users\\scott\\soccer-video\\out",
    [string]$PolicyPath = (Join-Path $Root 'OutPolicy.psd1'),
    [switch]$Commit
  )
  if (-not (Test-Path -LiteralPath $PolicyPath)) {
    throw "Policy file not found: $PolicyPath"
  }
  $policy = Import-PowerShellDataFile -LiteralPath $PolicyPath

  if ($policy.BannedPortrait) {
    Get-PortraitReport -Root $Root -Patterns @($policy.BannedPortrait) -Commit:$Commit | Out-Null
  }

  $keep = if ($policy.Prune -and $policy.Prune.RootBackupsKeep) {
    [int]$policy.Prune.RootBackupsKeep
  } else {
    3
  }
  $pruneSplat = @{ Root = $Root; Keep = $keep }
  if ($Commit) { $pruneSplat['Commit'] = $true }
  Clear-OldBackups @pruneSplat | Out-Null

  if ($policy.Prune -and $policy.Prune.EmptyDirs) {
    $empty = Get-ChildItem -LiteralPath $Root -Recurse -Directory -Force -EA SilentlyContinue |
             Where-Object { @(Get-ChildItem -LiteralPath $_.FullName -Force -EA SilentlyContinue).Count -eq 0 }
    Write-Host ("Empty dirs: {0}" -f $empty.Count)
    if ($Commit -and $empty.Count) { $empty | Remove-Item -Force }
  }

  $tmpDays = if ($policy.Prune) { [int]$policy.Prune.TmpDaysOld } else { 0 }
  if ($tmpDays -gt 0) {
    $tmp = Join-Path $Root '_tmp'
    if (Test-Path $tmp) {
      $old = Get-ChildItem -LiteralPath $tmp -Recurse -File -Force -EA SilentlyContinue |
             Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$tmpDays) }
      Write-Host ("_tmp old files: {0}" -f $old.Count)
      if ($Commit -and $old.Count) { $old | Remove-Item -Force }
    }
  }

  $quarantineDays = if ($policy.Prune) { [int]$policy.Prune.QuarantineDays } else { 0 }
  if ($quarantineDays -gt 0) {
    $qr = Join-Path $Root '_quarantine'
    if (Test-Path $qr) {
      $old = Get-ChildItem -LiteralPath $qr -Recurse -File -Force -EA SilentlyContinue |
             Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$quarantineDays) }
      Write-Host ("Quarantine old files: {0}" -f $old.Count)
      if ($Commit -and $old.Count) { $old | Remove-Item -Force }
    }
  }

  $reportScript = Join-Path $PSScriptRoot '..\OutTidyAndReport.ps1'
  powershell -ExecutionPolicy Bypass -File $reportScript -OutRoot $Root -Top 25
}

Export-ModuleMember -Function Clear-OldBackups, Get-PortraitReport, Invoke-OutPolicy, Resolve-JunctionTarget
