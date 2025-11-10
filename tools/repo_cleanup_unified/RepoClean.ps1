[CmdletBinding()]
param(
    [ValidateSet('Inventory')]
    [string]$Mode = 'Inventory',

    [Parameter(Mandatory)]
    [string]$Root,

    [string]$InventoryDirectory = $null,

    [string]$HashAlgorithm = 'MD5'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir 'RepoClean.Core.ps1')

if (-not (Test-Path -LiteralPath $Root)) {
    throw "Root path '$Root' was not found."
}

$resolvedRoot = (Resolve-Path -LiteralPath $Root).ProviderPath
if (-not $InventoryDirectory) {
    $InventoryDirectory = Join-Path -Path (Join-Path -Path $resolvedRoot -ChildPath 'out') -ChildPath 'inventory'
}

switch ($Mode) {
    'Inventory' {
        $exclude = @('out_trash\', 'out_trash\dedupe_exact\')
        if ($global:excludeFolders) {
            $exclude += $global:excludeFolders
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

        $records = Get-InventoryRecords -Files $files -RootPath $resolvedRoot -HashAlgo $HashAlgorithm

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
