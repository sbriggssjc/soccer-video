param(
  [string]$OutDir = ".\out"
)

$ErrorActionPreference = "Stop"

$rawPath = Join-Path $OutDir "events_raw.csv"
$selPath = Join-Path $OutDir "events_selected.csv"
$summaryPath = Join-Path $OutDir "review_summary.json"

if (-not (Test-Path $rawPath)) { throw "Missing events_raw.csv at $rawPath" }
if (-not (Test-Path $selPath)) { throw "Missing events_selected.csv at $selPath" }

$rawRows = Import-Csv $rawPath
$selRows = Import-Csv $selPath
$summary = $null
if (Test-Path $summaryPath) {
  $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
}

$goalTotal = 0
$goalIncluded = 0
$shotTotal = 0
$shotIncluded = 0
$stoppageSelected = 0

if ($summary -ne $null) {
  $goalTotal = [int]($summary.coverage.goals_total)
  $goalIncluded = [int]($summary.coverage.goals_included)
  $shotTotal = [int]($summary.coverage.shots_total)
  $shotIncluded = [int]($summary.coverage.shots_included)
} else {
  $goalTotal = ($rawRows | Where-Object { ($_.type -eq 'GOAL') }).Count
  $goalIncluded = ($selRows | Where-Object { ($_.type -eq 'GOAL') -or (($_.all_types) -and ($_.all_types -like '*GOAL*')) }).Count
  $shotTotal = ($rawRows | Where-Object { ($_.type -eq 'SHOT') }).Count
  $shotIncluded = ($selRows | Where-Object { ($_.type -eq 'SHOT') -or (($_.all_types) -and ($_.all_types -like '*SHOT*')) }).Count
}

$stoppageSelected = ($selRows | Where-Object { ($_.type -eq 'STOPPAGE') -or (($_.all_types) -and ($_.all_types -like '*STOPPAGE*')) }).Count

Write-Host ("[test_selection] Goals included: {0}/{1}" -f $goalIncluded, $goalTotal)
Write-Host ("[test_selection] Shots included: {0}/{1}" -f $shotIncluded, $shotTotal)
if ($summary -ne $null) {
  $saveIncluded = [int]($summary.coverage.saves_included)
  $saveTotal = [int]($summary.coverage.saves_total)
  $defIncluded = [int]($summary.coverage.def_included)
  $offIncluded = [int]($summary.coverage.off_included)
  Write-Host ("[test_selection] Saves included: {0}/{1}" -f $saveIncluded, $saveTotal)
  Write-Host ("[test_selection] Offensive build-ups included: {0}/{1}" -f $offIncluded, [int]($summary.coverage.off_total))
  Write-Host ("[test_selection] Defensive actions included: {0}/{1}" -f $defIncluded, [int]($summary.coverage.def_total))
}
Write-Host ("[test_selection] Stoppage clips in final: {0}" -f $stoppageSelected)

$failures = @()
if ($goalTotal -gt 0 -and $goalIncluded -lt $goalTotal) { $failures += "Missing goals" }
if ($shotTotal -gt 0 -and $shotIncluded -lt $shotTotal) { $failures += "Missing shots" }
if ($stoppageSelected -gt 0) { $failures += "Stoppage segments present" }

if ($failures.Count -gt 0) {
  Write-Error ("test_selection failed: " + ($failures -join '; '))
  exit 1
}

Write-Host "[test_selection] All recall checks passed."
exit 0
