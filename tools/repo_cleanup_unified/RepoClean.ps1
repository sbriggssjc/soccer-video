[CmdletBinding()]
param(
    [ValidateSet('Inventory')]
    [string]$Mode = 'Inventory',

    [Parameter(Mandatory)]
    [string]$Root,

    [string]$InventoryDirectory = $null,

    [string]$HashAlgorithm = 'MD5',

    [string[]]$ExcludeFolders = @('out_trash\', 'out_trash\dedupe_exact\')
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir 'RepoClean.Core.ps1')

# --- Self-contained defaults / guards ---

# exclude folders: define if missing
if (-not (Get-Variable -Name excludeFolders -Scope Script -ErrorAction SilentlyContinue) -and
    -not (Get-Variable -Name excludeFolders -Scope Global -ErrorAction SilentlyContinue)) {
  $script:excludeFolders = @('out_trash\','out_trash\dedupe_exact\')
} elseif (-not (Get-Variable -Name excludeFolders -Scope Script -ErrorAction SilentlyContinue)) {
  $script:excludeFolders = $global:excludeFolders
}

# target extensions: define in script: scope if missing
if (-not (Get-Variable -Name targetExtensions -Scope Script -ErrorAction SilentlyContinue) -and
    -not (Get-Variable -Name targetExtensions -Scope Global -ErrorAction SilentlyContinue)) {
  $script:targetExtensions = @('.mp4','.mov','.m4v','.mkv','.avi')
} elseif (-not (Get-Variable -Name targetExtensions -Scope Script -ErrorAction SilentlyContinue)) {
  $script:targetExtensions = $global:targetExtensions
}

# inventory directory: ensure exists (adjust if you store inventory elsewhere)
if (-not (Get-Variable -Name InvDir -Scope Script -ErrorAction SilentlyContinue)) {
  $script:InvDir = Join-Path $Root 'out\inventory'
  New-Item -ItemType Directory -Force -Path $script:InvDir | Out-Null
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($null -eq $ExcludeFolders) {
    $script:excludeFolders = @()
} else {
    $script:excludeFolders = @($ExcludeFolders)
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

        $records = Get-InventoryRecords -Files $files -RootPath $resolvedRoot -HashAlgo $HashAlgorithm -Extensions $script:targetExtensions

        $rows = [System.Collections.Generic.List[psobject]]::new()
        foreach ($record in $records) {
            [void]$rows.Add($record)
        }

        $csvPath = Join-Path -Path $InventoryDirectory -ChildPath 'repo_inventory.csv'
        New-Item -ItemType Directory -Force -Path $InventoryDirectory | Out-Null

        $rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8 -Force
        Add-Content -Path $csvPath -Value "# generated: $(Get-Date -Format o)"
    }
    default {
        throw "Mode '$Mode' is not implemented in this build."
    }
}

function Get-InventoryRecords {
    param(
        [Parameter(Mandatory)][System.Collections.IEnumerable]$Files,
        [Parameter(Mandatory)][string]$RootPath,
        [hashtable]$HashCache = $null,
        [string]$HashAlgo = 'MD5',
        [string[]]$Extensions = $script:targetExtensions
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
        $rec.Extension = $ext.ToLowerInvariant()

        $shouldHash = $true
        if ($size -le 0) {
            $shouldHash = $false
        } elseif ($Extensions -and ($Extensions -notcontains $rec.Extension)) {
            $shouldHash = $false
        }

        if ($HashCache -and $shouldHash) {
            $cacheKey = "$relPath|$size|$mtime"
            if ($HashCache.ContainsKey($cacheKey)) {
                $rec.Hash = $HashCache[$cacheKey].Hash
            } else {
                try {
                    $hasher = [System.Security.Cryptography.HashAlgorithm]::Create($HashAlgo)
                    $fs = [IO.File]::OpenRead($fullPath)
                    try {
                        $hash = ($hasher.ComputeHash($fs) | ForEach-Object { $_.ToString('x2') }) -join ''
                        $rec.Hash = $hash
                        $HashCache[$cacheKey] = [PSCustomObject]@{
                            Path          = $cacheKey
                            SizeBytes     = $size
                            LastWriteTime = $mtime
                            Hash          = $hash
                        }
                    } finally { $fs.Dispose() }
                } catch { }
            }
        }

        [void]$records.Add($rec)
    }

    return $records
}
