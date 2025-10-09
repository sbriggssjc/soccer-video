Param(
    [Parameter(Mandatory=$true)] [string]$RosterCsv,
    [Parameter(Mandatory=$true)] [string]$OutDir
)
$ErrorActionPreference = "Stop"
$makeOpenerScript = Join-Path $PSScriptRoot 'make_opener.ps1'

Import-Csv $RosterCsv | ForEach-Object {
    $row = $_

    # build a safe, informative filename: "14__Claire_Briggs.mp4"
    $safeName = ($row.PlayerName -replace '[^\w\- ]','').Trim() -replace '\s+','_'
    if ([string]::IsNullOrWhiteSpace($safeName)) {
        $safeName = "Player"
    }
    $outFile = Join-Path $OutDir ("{0}__{1}.mp4" -f $row.PlayerNumber, $safeName)

    Write-Host "Rendering opener -> $outFile"

    & $makeOpenerScript `
        -PlayerName   $row.PlayerName `
        -PlayerNumber "\#${($row.PlayerNumber)}" `
        -PlayerPhoto  $row.PlayerPhoto `
        -OutDir       $OutDir `
        -OutFile      $outFile
}
