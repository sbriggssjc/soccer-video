<# 
  Clean-SoccerVideo.ps1
  Safe cleanup for large repos with video/media artifacts.

  Features:
    - Duplicates by SHA256 (size-filtered first for speed)
    - Unreferenced large files (not mentioned anywhere in code/docs)
    - Quarantine (move) or Delete actions (both optional) 
    - CSV reports for review before action

  Usage examples (run from repo root):
    # DRY RUN – create reports only
    .\tools\cleanup\Clean-SoccerVideo.ps1 -Path . -MinSizeMB 50 -DryRun

    # Move duplicates & unreferenced to quarantine (review later)
    .\tools\cleanup\Clean-SoccerVideo.ps1 -Path . -MinSizeMB 50 -Quarantine -QuarantineDaysToKeep 14

    # Permanently delete the items that appear in the last reports
    .\tools\cleanup\Clean-SoccerVideo.ps1 -Path . -MinSizeMB 50 -Delete

  Notes:
    - Excludes common dependency/metadata dirs by default (.git, .venv, node_modules, .idea, .vscode, __pycache__).
    - “Unreferenced” = filename or relative path not found in any text source files (ps1/py/json/csv/md/etc.).
    - Quarantine path: .\.cleanup_trash\YYYYMMDD_HHMMSS\
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [string]$Path,

  [int]$MinSizeMB = 100,

  # Text files we scan to detect references (add more if you like)
  [string[]]$ReferenceExt = @(
    '.ps1','.psm1','.psd1','.py','.json','.jsonl','.csv','.tsv',
    '.md','.txt','.yml','.yaml','.ini','.cfg','.bat','.ps1vars'
  ),

  # Media / binary types we might purge if unreferenced/duplicate
  [string[]]$TargetExt = @(
    '.mp4','.mov','.mkv','.avi','.wav','.mp3','.flac','.aac',
    '.jpg','.jpeg','.png','.webp','.gif','.zip','.7z','.tar','.gz'
  ),

  # Common big-output folders (heuristic only; still treated safely)
  [string[]]$LikelyArtifactDirs = @('out','output','.cache','clips','reels','build','dist'),

  [switch]$Quarantine,              # move files into .\.cleanup_trash\<timestamp>\
  [switch]$Delete,                  # permanently delete files discovered by the scan
  [switch]$DryRun,                  # just write reports (default behavior if neither -Quarantine nor -Delete)
  [int]$QuarantineDaysToKeep = 30   # older quarantine folders you may remove later
)

function New-ReportDir {
  param([string]$Base)
  $report = Join-Path $Base "cleanup_reports"
  New-Item -ItemType Directory -Force -Path $report | Out-Null
  return $report
}

function Get-RepoFiles {
  param([string]$Root)

  $excludeDirs = @(
    '\.git($|\\)','\\\.venv($|\\)','\\node_modules($|\\)',
    '\\\.idea($|\\)','\\\.vscode($|\\)','\\__pycache__($|\\)'
  )

  Get-ChildItem -Path $Root -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object {
      $full = $_.FullName
      -not ($excludeDirs | ForEach-Object { $full -match $_ } | Where-Object { $_ })
    }
}

function Get-TextFiles {
  param([System.IO.FileInfo[]]$All, [string[]]$Exts)
  $extSet = $Exts | ForEach-Object { $_.ToLowerInvariant() }
  $All | Where-Object { $extSet -contains $_.Extension.ToLowerInvariant() }
}

function Get-TargetFiles {
  param([System.IO.FileInfo[]]$All, [string[]]$Exts, [int]$MinSizeMB)
  $minBytes = $MinSizeMB * 1MB
  $extSet = $Exts | ForEach-Object { $_.ToLowerInvariant() }
  $All | Where-Object {
    $_.Length -ge $minBytes -and ($extSet -contains $_.Extension.ToLowerInvariant())
  }
}

function Build-ReferenceIndex {
  param([System.IO.FileInfo[]]$TextFiles, [string]$Root)

  Write-Host "Indexing references from $($TextFiles.Count) text files..." -ForegroundColor Cyan
  $refs = New-Object 'System.Collections.Generic.HashSet[string]'

  foreach ($tf in $TextFiles) {
    try {
      $content = Get-Content -Raw -Encoding UTF8 -LiteralPath $tf.FullName
    } catch {
      # Try default encoding if UTF8 fails
      try { $content = Get-Content -Raw -LiteralPath $tf.FullName } catch { continue }
    }

    # Add any substring that looks like filename or path pieces (simple heuristics)
    # 1) full relative paths found in repo
    # 2) bare basenames (clip names etc.)
    # We don't get fancy: we’ll just add every token with a dot/extension-ish pattern
    $tokens = [System.Text.RegularExpressions.Regex]::Matches($content, '(?:[A-Za-z0-9_\-\\/\. ]+\.(?:mp4|mov|mkv|avi|wav|mp3|flac|aac|jpg|jpeg|png|webp|gif|zip|7z|tar|gz|csv|json|ps1vars))', 'IgnoreCase')
    foreach ($m in $tokens) {
      $val = $m.Value.Trim()
      if (![string]::IsNullOrWhiteSpace($val)) { [void]$refs.Add($val) }
      # also add just the basename variant
      try {
        $bn = [System.IO.Path]::GetFileName($val)
        if ($bn) { [void]$refs.Add($bn) }
      } catch {}
    }
  }

  # Also include file names that appear as plain words like "clip001.mp4" without any path separator
  # (covered by the regex above already)

  return $refs
}

function Find-Duplicates {
  param([System.IO.FileInfo[]]$Files)

  Write-Host "Finding duplicates among $($Files.Count) candidate files..." -ForegroundColor Cyan

  # First group by size to avoid hashing everything
  $bySize = $Files | Group-Object Length | Where-Object { $_.Count -gt 1 }

  $results = @()
  foreach ($grp in $bySize) {
    $sameSizeFiles = $grp.Group
    # Hash only this group
    $hashes = foreach ($f in $sameSizeFiles) {
      try {
        $h = Get-FileHash -Algorithm SHA256 -LiteralPath $f.FullName
        [PSCustomObject]@{
          FullName = $f.FullName
          Length   = $f.Length
          Hash     = $h.Hash
          LastWriteTime = $f.LastWriteTime
        }
      } catch {
        # if hashing fails, skip
      }
    }

    # group by hash
    $byHash = $hashes | Group-Object Hash | Where-Object { $_.Count -gt 1 }
    foreach ($hgrp in $byHash) {
      $dups = $hgrp.Group | Sort-Object LastWriteTime -Descending
      # Keep the newest file, mark the rest as duplicates
      $keep = $dups[0]
      $rest = $dups | Select-Object -Skip 1
      foreach ($r in $rest) {
        $results += [PSCustomObject]@{
          Duplicate = $r.FullName
          Keep      = $keep.FullName
          SizeBytes = $r.Length
          Hash      = $r.Hash
          DuplicateLastWrite = $r.LastWriteTime
          KeepLastWrite      = $keep.LastWriteTime
        }
      }
    }
  }

  return $results
}

function Find-Unreferenced {
  param(
    [System.IO.FileInfo[]]$Files,
    [System.Collections.Generic.HashSet[string]]$Refs,
    [string]$Root
  )

  Write-Host "Scanning for unreferenced large files..." -ForegroundColor Cyan

  $rootFull = (Resolve-Path $Root).Path
  $unref = @()

  foreach ($f in $Files) {
    $rel = $f.FullName.Replace($rootFull,'').TrimStart('\','/')
    $bn  = $f.Name

    $isReferenced =
      $Refs.Contains($bn) -or
      $Refs.Contains($rel) -or
      $Refs.Contains($rel -replace '\\','/') -or
      $Refs.Contains($rel -replace '/','\')

    if (-not $isReferenced) {
      $unref += [PSCustomObject]@{
        FullName = $f.FullName
        Relative = $rel
        SizeBytes = $f.Length
        LastWriteTime = $f.LastWriteTime
        InLikelyArtifactDir = ($rel -split '[\\/]' | Where-Object { $_ -in $LikelyArtifactDirs }).Count -gt 0
      }
    }
  }

  return $unref
}

# --- Main ---
$root = Resolve-Path $Path
$reportDir = New-ReportDir -Base $root

$allFiles = Get-RepoFiles -Root $root
$textFiles = Get-TextFiles -All $allFiles -Exts $ReferenceExt
$targets   = Get-TargetFiles -All $allFiles -Exts $TargetExt -MinSizeMB $MinSizeMB

Write-Host ("Total files: {0} | Text files to index: {1} | Target large files: {2}" -f `
  $allFiles.Count, $textFiles.Count, $targets.Count) -ForegroundColor Yellow

$refIndex = Build-ReferenceIndex -TextFiles $textFiles -Root $root

$dupRows  = Find-Duplicates -Files $targets
$unrefRows = Find-Unreferenced -Files $targets -Refs $refIndex -Root $root

# Write reports
$dupCsv   = Join-Path $reportDir ("duplicates_{0}.csv" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$unrefCsv = Join-Path $reportDir ("unreferenced_{0}.csv" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

$dupRows  | Sort-Object SizeBytes -Descending | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $dupCsv
$unrefRows| Sort-Object SizeBytes -Descending | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $unrefCsv

Write-Host "Reports written:" -ForegroundColor Green
Write-Host "  Duplicates   -> $dupCsv"
Write-Host "  Unreferenced -> $unrefCsv"

# Stop here if dry run or no action flags
if ($DryRun -or (-not $Quarantine -and -not $Delete)) {
  Write-Host "`nDry run complete. Review the CSVs before taking action." -ForegroundColor Cyan
  exit 0
}

# Action phase
$actionList = @()

if ($dupRows.Count -gt 0) {
  $actionList += ($dupRows | Select-Object @{n='FullName';e={$_.Duplicate}}, SizeBytes)
}
if ($unrefRows.Count -gt 0) {
  $actionList += ($unrefRows | Select-Object FullName, SizeBytes)
}

if ($actionList.Count -eq 0) {
  Write-Host "Nothing to act on." -ForegroundColor Green
  exit 0
}

if ($Quarantine) {
  $qBase = Join-Path $root (".cleanup_trash")
  New-Item -ItemType Directory -Force -Path $qBase | Out-Null
  $qDir = Join-Path $qBase (Get-Date -Format 'yyyyMMdd_HHmmss')
  New-Item -ItemType Directory -Force -Path $qDir | Out-Null

  Write-Host "Quarantining $($actionList.Count) files to $qDir ..." -ForegroundColor Yellow

  foreach ($i in $actionList) {
    $src = $i.FullName
    if (Test-Path -LiteralPath $src) {
      $rel = (Resolve-Path $src).Path.Replace((Resolve-Path $root).Path,'').TrimStart('\','/')
      $dest = Join-Path $qDir $rel
      $destDir = Split-Path $dest -Parent
      New-Item -ItemType Directory -Force -Path $destDir | Out-Null
      try {
        Move-Item -LiteralPath $src -Destination $dest -Force
      } catch {
        Write-Warning "Failed to move: $src ($_)" 
      }
    }
  }

  Write-Host "Quarantine complete." -ForegroundColor Green
  Write-Host "You can delete the quarantine folder later if nothing breaks." -ForegroundColor DarkYellow
  Write-Host "Tip: remove quarantine folders older than $QuarantineDaysToKeep days." -ForegroundColor DarkYellow
  exit 0
}

if ($Delete) {
  Write-Host "Deleting $($actionList.Count) files (permanent)..." -ForegroundColor Red
  foreach ($i in $actionList) {
    $src = $i.FullName
    if (Test-Path -LiteralPath $src) {
      try {
        Remove-Item -LiteralPath $src -Force
      } catch {
        Write-Warning "Failed to delete: $src ($_)" 
      }
    }
  }
  Write-Host "Delete complete." -ForegroundColor Green
}
