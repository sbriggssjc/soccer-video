# REPOCLEAN WRAPPER (safe, self-contained)
# tools\repo_cleanup_unified\RepoClean.ps1
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('Inventory')]
  [string]$Mode,

  [Parameter(Mandatory=$true)]
  [string]$Root,

  [switch]$ComputeHashes,
  [switch]$DedupExact
)

$ErrorActionPreference = 'Stop'

# --- Safe defaults if caller didn't predefine these globals ---
if (-not $global:excludeFolders) {
  $global:excludeFolders = @(
    'out\_trash\','out\_trash\dedupe_exact\','out\scratch\',
    '.git\','node_modules\','venv\','env\','site-packages\','__pycache__\'
  )
}
if (-not $global:targetExtensions) {
  $global:targetExtensions = @('.mp4','.mov','.m4v','.mkv','.avi')
}

function Test-Excluded([string]$root, [string]$fullPath){
  $rel = ($fullPath.Substring($root.Length)).TrimStart('\\','/')
  foreach($ex in $global:excludeFolders){
    $ex = ($ex -replace '/','\\').TrimEnd('\\')
    if ($rel -like "$ex*") { return $true }
  }
  return $false
}

function Get-Bucket([string]$rel){
  $rel = ($rel -replace '/','\\')
  if ($rel -like 'out\\reels\\portrait_1080x1920\\*') { return 'portrait_export' }
  if ($rel -like 'out\\reels\\portrait_branded\\*')    { return 'portrait_branded' }
  if ($rel -like 'out\\atomic_clips\\*')              { return 'atomic_clips' }
  if ($rel -like 'out\\games\\*')                     { return 'games' }
  if ($rel -like 'out\\masters\\*')                   { return 'masters' }
  if ($rel -like 'out\\follow_diag\\*')               { return 'follow_diag' }
  return 'other'
}

function Get-Score($rel,$ts){ return (Get-Date $ts).ToFileTimeUtc() } # newest first

function Get-InventoryRecords([string]$RootPath){
  $resolvedRoot = (Resolve-Path $RootPath).Path
  $files = Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Force

  $records = [System.Collections.Generic.List[psobject]]::new()
  foreach($f in $files){
    if (Test-Excluded -root $resolvedRoot -fullPath $f.FullName) { continue }
    $rel  = ($f.FullName.Substring($resolvedRoot.Length)).TrimStart('\\','/')
    $hash = ''
    if ($ComputeHashes -and ($global:targetExtensions -contains $f.Extension.ToLower())){
      try { $hash = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash } catch {}
    }
    $records.Add([pscustomobject]@{
      RelativePath  = ($rel -replace '/','\\')
      FullPath      = $f.FullName
      SizeBytes     = $f.Length
      LastWriteTime = $f.LastWriteTime
      Hash          = $hash
    })
  }
  return $records
}

if ($Mode -eq 'Inventory') {
  $outDir = Join-Path $Root 'out\inventory'
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $csv = Join-Path $outDir 'repo_inventory.csv'

  $recs = Get-InventoryRecords -RootPath $Root
  $recs | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $csv
  Write-Host "[Inventory] Wrote $(@($recs).Count) rows -> $csv"

  if ($DedupExact) {
    $live = $recs | Where-Object {
      $_.Hash -and $_.Hash.Trim() -ne '' -and
      $_.FullPath -and $_.FullPath.Trim() -ne '' -and
      (($_.FullPath -replace '/','\\') -notlike '*\\out\\_trash\\*') -and
      (Test-Path -LiteralPath $_.FullPath)
    }
    $dupGroups = $live | Group-Object Hash | Where-Object Count -gt 1
    if ($dupGroups.Count -gt 0) {
      $stamp = Get-Date -Format yyyyMMdd_HHmmss
      $quar  = Join-Path $Root "out\_trash\dedupe_exact\$stamp"
      $log   = Join-Path $outDir "RepoClean_dedupe_$stamp.csv"

      $priority = @('portrait_branded','portrait_export','masters','games','atomic_clips','follow_diag','other')
      $prioMap=@{}; 0..($priority.Count-1) | % { $prioMap[$priority[$_]] = $_ }

      $toMove=@()
      foreach($g in $dupGroups){
        $grp = $g.Group
        $buckets    = $grp | ForEach-Object { Get-Bucket $_.RelativePath } | Sort-Object -Unique
        $keepBucket = ($buckets | Sort-Object { if($prioMap.ContainsKey($_)){ $prioMap[$_] } else { 999 } } | Select-Object -First 1)
        $keepers    = $grp | Where-Object { (Get-Bucket $_.RelativePath) -eq $keepBucket }
        $keep = $keepers | Sort-Object `
          @{e={ -1 * (Get-Score $_.RelativePath $_.LastWriteTime) }},
          @{e={$_.SizeBytes}; Descending=$true},
          @{e={$_.RelativePath}} | Select-Object -First 1

        foreach($x in ($grp | Where-Object { $_.FullPath -ne $keep.FullPath })){
          $toMove += [pscustomobject]@{
            Hash         = $x.Hash
            KeepRel      = $keep.RelativePath
            RelativePath = $x.RelativePath
            FullPath     = $x.FullPath
            SizeBytes    = $x.SizeBytes
            Reason       = "keep-$keepBucket"
          }
        }
      }

      if ($toMove.Count -gt 0) {
        $toMove | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $log
        $failed=@()
        foreach($r in $toMove){
          if(-not (Test-Path -LiteralPath $r.FullPath)) { continue }
          $dst = Join-Path $quar ($r.RelativePath -replace '/','\\')
          New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($dst)) | Out-Null
          try { Move-Item "\\\\?\$($r.FullPath)" "\\\\?\$dst" -Force -ErrorAction Stop }
          catch { $failed += [pscustomobject]@{ FullPath=$r.FullPath; Error=$_.Exception.Message } }
        }
        if($failed.Count){
          $failCsv = Join-Path $outDir "RepoClean_dedupe_failed_$stamp.csv"
          $failed | Export-Csv $failCsv -NoTypeInformation -Encoding UTF8
          Write-Host "Move failures: $($failed.Count) -> $failCsv"
        } else {
          Write-Host "Dedup move complete."
        }
      } else {
        Write-Host "No duplicate hashes to move."
      }
    } else {
      Write-Host "No live duplicate hashes found."
    }
  }

  exit 0
}

throw "Unknown -Mode '$Mode'."

