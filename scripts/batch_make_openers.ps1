Param(
    [Parameter(Mandatory=$true)] [string]$RosterCsv,
    [Parameter(Mandatory=$true)] [string]$OutDir
)
$ErrorActionPreference = "Stop"
Import-Csv $RosterCsv | ForEach-Object {
    $name = $_.name
    $number = [int]$_.number
    $photo = $_.photo
    pwsh -File (Join-Path $PSScriptRoot "make_opener.ps1") `
        -PlayerName $name -PlayerNumber $number -PlayerPhoto $photo -OutDir $OutDir
}
