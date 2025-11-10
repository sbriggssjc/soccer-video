[CmdletBinding()]
param(
    [ValidateSet('Inventory')]
    [string]$Mode = 'Inventory',

    [Parameter(Mandatory)]
    [string]$Root,

    [string]$InventoryDirectory = $null,

    [ValidateSet('none','sha256','size')]
    [string]$HashMode = 'sha256',

    [string[]]$ExcludeFolders = @('out_trash\', 'out_trash\dedupe_exact\')
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir 'RepoClean.Core.ps1')

# Defaults (self-contained)
if (-not (Get-Variable -Name excludeFolders -Scope Script -ErrorAction SilentlyContinue) -and
    -not (Get-Variable -Name excludeFolders -Scope Global -ErrorAction SilentlyContinue)) {
  $script:excludeFolders = @('out_trash\','out_trash\dedupe_exact\')
} elseif (-not (Get-Variable -Name excludeFolders -Scope Script -ErrorAction SilentlyContinue)) {
  $script:excludeFolders = $global:excludeFolders
}

if (-not (Get-Variable -Name targetExtensions -Scope Script -ErrorAction SilentlyContinue) -and
    -not (Get-Variable -Name targetExtensions -Scope Global -ErrorAction SilentlyContinue)) {
  $script:targetExtensions = @('.mp4','.mov','.m4v','.mkv','.avi')
} elseif (-not (Get-Variable -Name targetExtensions -Scope Script -ErrorAction SilentlyContinue)) {
  $script:targetExtensions = $global:targetExtensions
}

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

        $records = Get-InventoryRecords -Files $files -RootPath $resolvedRoot -HashMode $HashMode -Extensions $script:targetExtensions

        $invCsv = Join-Path $script:InvDir 'repo_inventory.csv'
        $records | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $invCsv
        Add-Content -Path $invCsv -Value "# generated: $(Get-Date -Format o)"

        Write-Host ("[Inventory] Wrote {0:n0} rows -> {1}" -f $records.Count, $invCsv)
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
        [string]$HashMode = 'sha256',
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

        $hash = ''
        if ($shouldHash) {
            $hash = Get-ContentHash -Path $fullPath -HashMode $HashMode
        }
        $rec.Hash = $hash

        [void]$records.Add($rec)
    }

    return $records
}
