[CmdletBinding()]
param(
  [ValidateSet('Inventory')]
  [string]$Mode = 'Inventory',
  [Parameter(Mandatory=$true)]
  [string]$Root,
  [ValidateSet('none','sha256','size')]
  [string]$HashMode = 'sha256',
  [string]$InventoryDirectory = $null,
  [string[]]$ExcludeFolders = @('out_trash\','out_trash\dedupe_exact\')
)

if (-not $script:targetExtensions) { $script:targetExtensions = @('.mp4','.mov','.m4v','.mkv','.avi') }
if (-not $script:excludeFolders)   { $script:excludeFolders   = @('out_trash\','out_trash\dedupe_exact\') }

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

        $hash = ''
        if ($script:targetExtensions -contains $ext) {
            $hash = Get-ContentHash -Path $fullPath -HashMode $HashMode
        }
        $rec.Hash = $hash

        [void]$records.Add($rec)
    }

    return $records
}
