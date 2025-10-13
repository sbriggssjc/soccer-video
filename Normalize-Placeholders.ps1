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
    $k = ('{0}|{1}|{2}' -f $Matches.label, $Matches.t1, $Matches.t2)
    $brandedMap[$k] = @{
      date = $Matches.date
      home = $Matches.home
      away = $Matches.away
    }
  }
}

Get-ChildItem -LiteralPath $CinematicRoot -Directory | ForEach-Object {
  $name = $_.Name
  if ($name -match $FolderRegex) {
    $date = $Matches.date; $home = $Matches.home; $away = $Matches.away
    $needsFix = ($date -eq '1970-01-01') -or ($home -eq 'HOME') -or ($away -eq 'AWAY')

    if ($needsFix) {
      $k = ('{0}|{1}|{2}' -f $Matches.label, $Matches.t1, $Matches.t2)
      if ($brandedMap.ContainsKey($k)) {
        $meta = $brandedMap[$k]
        $newName = ('{0}__{1}__{2}_vs_{3}__{4}__t{5}-t{6}_portrait_FINAL' -f `
          $Matches.idx, $meta.date, $meta.home, $meta.away, $Matches.label, $Matches.t1, $Matches.t2)
        if ($newName -ne $name) {
          if ($WhatIf) {
            "Would rename: $name -> $newName"
          } else {
            Rename-Item -LiteralPath $_.FullName -NewName $newName
            "Renamed: $name -> $newName"
          }
        }
      }
    }
  }
}

"Re-scan next:  powershell -NoProfile -ExecutionPolicy Bypass -File .\Scan-AtomicClips.ps1"
