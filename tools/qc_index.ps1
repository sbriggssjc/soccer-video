param([string]$Csv = ".\atomic_clips_index.rebuilt.csv")

$rows = Import-Csv -LiteralPath $Csv

"Rows: {0}" -f $rows.Count
$rows | Group-Object date | Sort-Object Name | Select Name,Count | Format-Table

$proxyCSV      = ($rows | Where-Object { $_.proxy_exists      -eq 'true' }).Count
$trfCSV        = ($rows | Where-Object { $_.trf_exists        -eq 'true' }).Count
$stabilizedCSV = ($rows | Where-Object { $_.stabilized_exists -eq 'true' }).Count
$brandedCSV    = ($rows | Where-Object { $_.has_branded       -eq 'true' }).Count

"Proxy exists (CSV):      {0}" -f $proxyCSV
"Transforms exists (CSV): {0}" -f $trfCSV
"Stabilized exists (CSV): {0}" -f $stabilizedCSV
"Has branded (CSV):       {0}" -f $brandedCSV

$bad = $rows | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.date) -or $_.date -eq '1970-01-01' }
if ($bad.Count -gt 0) {
  Write-Host "`nERROR: placeholder dates detected:" -ForegroundColor Red
  $bad | Select-Object -First 10 index,homeTeam,awayTeam,date | Format-Table
  exit 1
} else {
  Write-Host "`nDate check OK (no placeholders)." -ForegroundColor Green
}
