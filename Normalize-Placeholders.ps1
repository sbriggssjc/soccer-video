param(
  [string]$CinematicRoot = ".\out\autoframe_work\cinematic",
  [string]$BrandedRoot   = ".\out\portrait_reels\branded",
  [switch]$WhatIf
)

# Share the same pattern across scripts
. "$PSScriptRoot\FolderRegex.ps1"

$BrandedRegex = '^(?<idx>\d{3})__(?:(?<date>\d{4}-\d{2}-\d{2})__)?(?<home>.+?)_vs_(?<away>.+?)__(?<label>.+?)__t(?<t1>\d+(?:\.\d+)?)(?:-t?|_)(?<t2>\d+(?:\.\d+)?).*_portrait_(?:BRAND|POST)\.mp4$'

# Build a lookup from branded files keyed by (label|t1|t2)
$brandedMap = @{}
Get-ChildItem -LiteralPath $BrandedRoot -File -Recurse | ForEach-Object {
  if ($_.Name -match $BrandedRegex) {
    $labelToken = $Matches.label
    $tStartToken = $Matches.t1
    $tEndToken = $Matches.t2
    $key = '{0}|{1}|{2}' -f $labelToken, $tStartToken, $tEndToken
    $brandedMap[$key] = @{
      matchDate = $Matches.date
      homeTeam  = $Matches.home
      awayTeam  = $Matches.away
    }
  }
}

Get-ChildItem -LiteralPath $CinematicRoot -Directory | ForEach-Object {
  $folderName = $_.Name
  if ($folderName -match $FolderRegex) {
    $matchDate = $Matches.date
    $homeTeam  = $Matches.home
    $awayTeam  = $Matches.away
    $clipIndex = $Matches.idx
    $labelToken = $Matches.label
    $tStartToken = $Matches.t1
    $tEndToken = $Matches.t2
    $needsFix = ($matchDate -eq '1970-01-01') -or ($homeTeam -eq 'HOME') -or ($awayTeam -eq 'AWAY')

    if ($needsFix) {
      $key = '{0}|{1}|{2}' -f $labelToken, $tStartToken, $tEndToken
      if ($brandedMap.ContainsKey($key)) {
        $meta = $brandedMap[$key]
        $newName = ('{0}__{1}__{2}_vs_{3}__{4}__t{5}-t{6}_portrait_FINAL' -f `
          $clipIndex, $meta.matchDate, $meta.homeTeam, $meta.awayTeam, $labelToken, $tStartToken, $tEndToken)
        if ($newName -ne $folderName) {
          if ($WhatIf) {
            "Would rename: $folderName -> $newName"
          } else {
            Rename-Item -LiteralPath $_.FullName -NewName $newName
            "Renamed: $folderName -> $newName"
          }
        }
      }
    }
  }
}

"Re-scan next:  powershell -NoProfile -ExecutionPolicy Bypass -File .\Scan-AtomicClips.ps1"
