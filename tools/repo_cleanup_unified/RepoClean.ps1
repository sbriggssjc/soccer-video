if (-not $global:excludeFolders) {
  $global:excludeFolders = @('out_trash\','out_trash\dedupe_exact\','out\scratch\',
                             '.git','node_modules','venv','env','site-packages','__pycache__')
}
if (-not $global:targetExtensions) {
  $global:targetExtensions = @('.mp4','.mov','.m4v','.mkv','.avi')
}

[CmdletBinding()]
param(
  [ValidateSet('Inventory')]
  [string]$Mode = 'Inventory',
  [Parameter(Mandatory=$true)]
  [string]$Root,
  [ValidateSet('none','sha256','size')]
  [string]$HashMode = 'sha256',
  [string]$InventoryDirectory = $null,
  [string[]]$ExcludeFolders = @('out_trash\','out_trash\dedupe_exact\'),
  [switch]$ComputeHashes,
  [switch]$DedupExact
  [switch]$QuarantineZeroByte
)

if (-not $script:targetExtensions) { $script:targetExtensions = $global:targetExtensions }
if (-not $script:excludeFolders)   { $script:excludeFolders   = $global:excludeFolders }

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir 'RepoClean.Core.ps1')

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($PSBoundParameters.ContainsKey('ExcludeFolders')) {
    if ($null -eq $ExcludeFolders) {
        $script:excludeFolders = @()
    } else {
        $script:excludeFolders = @($ExcludeFolders)
    }
}

if (-not (Test-Path -LiteralPath $Root)) {
    throw "Root path '$Root' was not found."
}

$resolvedRoot = (Resolve-Path -LiteralPath $Root).ProviderPath
if (-not $InventoryDirectory) {
    $InventoryDirectory = Join-Path -Path (Join-Path -Path $resolvedRoot -ChildPath 'out') -ChildPath 'inventory'
}

$script:InvDir = $InventoryDirectory
New-Item -ItemType Directory -Force -Path $script:InvDir | Out-Null

switch ($Mode) {
    'Inventory' {
        $exclude = @()
        if ($script:excludeFolders) {
            $exclude += $script:excludeFolders
        }

        function ShouldSkip([string]$p) {
            $pp = ($p -replace '/', '\\')
            foreach ($x in $exclude) {
                if ($pp -like "*$x*") {
                    return $true
                }
            }
            return $false
        }

        $files = Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -ErrorAction SilentlyContinue |
                 Where-Object { -not (ShouldSkip $_.FullName) }

        $records = Get-InventoryRecords -Files $files -RootPath $resolvedRoot -HashMode $HashMode

        if ($ComputeHashes) {
          $wantExt = $global:targetExtensions | ForEach-Object { $_.ToLower() }
          $i = 0
          foreach ($r in $records) {
            if (-not $r.FullPath) { continue }
            $ext = [IO.Path]::GetExtension($r.FullPath).ToLower()
            if ($wantExt -contains $ext -and (Test-Path -LiteralPath $r.FullPath)) {
              $i++; if ($i % 500 -eq 0) { "…hashed $i files" | Out-Host }
              try { $r.Hash = (Get-FileHash -LiteralPath $r.FullPath -Algorithm SHA256).Hash } catch {}
            }
          }
        if ($QuarantineZeroByte) {
            $EMPTY_HASH = 'E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855'
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            $quarRoot = Join-Path -Path $Root -ChildPath 'out_trash'
            $quarRoot = Join-Path -Path $quarRoot -ChildPath 'zero_byte'
            $quar = Join-Path -Path $quarRoot -ChildPath $stamp
            $zeros = @($records | Where-Object {
                $_.Hash -eq $EMPTY_HASH -and $_.FullPath -and $_.FullPath.Trim() -ne '' -and
                (($_.FullPath -replace '/','\') -notlike '*\out_trash\*')
            })
            if ($zeros.Count) {
                foreach ($r in $zeros) {
                    if (-not (Test-Path -LiteralPath $r.FullPath)) { continue }
                    $dst = Join-Path -Path $quar -ChildPath ($r.RelativePath -replace '/','\')
                    $dstDir = [IO.Path]::GetDirectoryName($dst)
                    if ($dstDir) { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
                    try {
                        Move-Item "\\?\$($r.FullPath)" "\\?\$dst" -Force -ErrorAction Stop
                    } catch {
                    }
                }
                Write-Host ("[Inventory] Quarantined {0:n0} zero-byte files -> {1}" -f $zeros.Count, $quar)
            }
        }

        $invCsv = Join-Path $script:InvDir 'repo_inventory.csv'
        $records | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $invCsv
        Add-Content -Path $invCsv -Value "# generated: $(Get-Date -Format o)"

        Write-Host ("[Inventory] Wrote {0:n0} rows -> {1}" -f $records.Count, $invCsv)

        if ($DedupExact) {
          $Root   = (Resolve-Path $Root).Path
          $InvCsv = Join-Path $Root 'out\inventory\repo_inventory.csv'
          $inv    = Import-Csv -LiteralPath $InvCsv | Where-Object {
            $_.Hash -and $_.Hash.Trim() -ne '' -and $_.FullPath -and $_.FullPath.Trim() -ne '' `
            -and (($_.FullPath -replace '/','\') -notlike '*\out\_trash\*') -and (Test-Path -LiteralPath $_.FullPath)
          }

          function Get-Bucket([string]$rel){
            $rel = ($rel -replace '/','\')
            if ($rel -like 'out\reels\portrait_1080x1920\*') { return 'portrait_export' }
            if ($rel -like 'out\reels\portrait_branded\*')    { return 'portrait_branded' }
            if ($rel -like 'out\atomic_clips\*')              { return 'atomic_clips' }
            if ($rel -like 'out\games\*')                     { return 'games' }
            if ($rel -like 'out\masters\*')                   { return 'masters' }
            if ($rel -like 'out\follow_diag\*')               { return 'follow_diag' }
            return 'other'
          }
          function Get-Score($rel,$ts){ return (Get-Date $ts).ToFileTimeUtc() }  # newest first

          $priority = @('portrait_branded','portrait_export','masters','games','atomic_clips','follow_diag','other')
          $prioMap  = @{}; 0..($priority.Count-1) | % { $prioMap[$priority[$_]] = $_ }

          $groups = $inv | Group-Object Hash | Where-Object Count -gt 1
          if ($groups.Count) {
            $stamp  = Get-Date -Format yyyyMMdd_HHmmss
            $quar   = Join-Path $Root "out\_trash\dedupe_exact\$stamp"
            $logCsv = Join-Path $Root "out\inventory\R_keep_one_per_hash_$stamp.csv"
            $toMove = @()

            foreach($g in $groups){
              $grp = $g.Group
              $buckets = $grp | ForEach-Object { Get-Bucket $_.RelativePath } | Sort-Object -Unique
              $keepBucket = ($buckets | Sort-Object { if($prioMap.ContainsKey($_)){ $prioMap[$_] } else { 999 } } | Select-Object -First 1)
              $keepers = $grp | Where-Object { (Get-Bucket $_.RelativePath) -eq $keepBucket }
              $keep = $keepers | Sort-Object `
                        @{e={ -1 * (Get-Score $_.RelativePath $_.LastWriteTime) }}, `
                        @{e={$_.SizeBytes}; Descending=$true}, `
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

            if ($toMove.Count) {
              New-Item -ItemType Directory -Force -Path $quar | Out-Null
              $toMove | Export-Csv $logCsv -NoTypeInformation -Encoding UTF8 | Out-Null
              foreach($r in $toMove){
                if(-not (Test-Path -LiteralPath $r.FullPath)) { continue }
                $dst = Join-Path $quar ($r.RelativePath -replace '/','\')
                New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($dst)) | Out-Null
                try { Move-Item "\\?\$($r.FullPath)" "\\?\$dst" -Force -ErrorAction Stop } catch {}
              }
            }
          }
        }
    }
    default {
        throw "Mode '$Mode' is not implemented in this build."
    }
}

function Get-ContentHash {
    param(
        [string]$Path,
        [string]$HashMode = 'sha256'
    )
    try {
        switch ($HashMode) {
            'sha256' { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash }
            'size'   { return (Get-Item -LiteralPath $Path).Length.ToString() }
            default  { return '' }
        }
    } catch { return '' }
}

function Get-InventoryRecords {
    param(
        [Parameter(Mandatory)][System.Collections.IEnumerable]$Files,
        [Parameter(Mandatory)][string]$RootPath,
        [string]$HashMode = 'sha256'
    )

    $records = [System.Collections.Generic.List[psobject]]::new()

    foreach ($file in $Files) {
        $fullPath = $file.FullName
        $relPath  = [IO.Path]::GetRelativePath($RootPath, $fullPath)
        $size     = $file.Length
        $mtime    = $file.LastWriteTimeUtc

        $rec = [PSCustomObject]@{
            RelativePath  = $relPath
            FullPath      = $fullPath
            SizeBytes     = $size
            LastWriteTime = $mtime
            Extension     = ''
            Codec         = $null
            Width         = $null
            Height        = $null
            Hash          = $null
            Status        = $null
            Notes         = $null
        }

        $ext = [IO.Path]::GetExtension($fullPath)
        if ([string]::IsNullOrEmpty($ext)) { $ext = '' }
        $ext = $ext.ToLowerInvariant()
        $rec.Extension = $ext

        $rec.Hash = ''

        [void]$records.Add($rec)
    }

    return $records
}
