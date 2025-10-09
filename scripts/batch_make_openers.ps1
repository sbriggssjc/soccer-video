Param(
    [Parameter(Mandatory=$true)] [string]$RosterCsv,
    [Parameter(Mandatory=$true)] [string]$OutDir
)
$ErrorActionPreference = "Stop"
Import-Csv $RosterCsv | ForEach-Object {
    & "$PSScriptRoot\make_opener.ps1" `
        -PlayerName   $_.PlayerName `
        -PlayerNumber ([int]$_.PlayerNumber) `
        -PlayerPhoto  $_.PlayerPhoto `
        -OutDir       $OutDir
}
