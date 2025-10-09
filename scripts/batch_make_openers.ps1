Param(
    [Parameter(Mandatory=$true)] [string]$RosterCsv,
    [Parameter(Mandatory=$true)] [string]$OutDir
)
$ErrorActionPreference = "Stop"
$makeOpenerScript = Join-Path $PSScriptRoot 'make_opener.ps1'

Import-Csv $RosterCsv | ForEach-Object {
    $row = $_

    & $makeOpenerScript `
        -PlayerName   $row.PlayerName `
        -PlayerNumber ([int]$row.PlayerNumber) `
        -PlayerPhoto  $row.PlayerPhoto `
        -OutDir       $OutDir
}
