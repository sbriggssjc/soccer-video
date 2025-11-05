param(
  [Parameter(Mandatory=$true)][string]$Csv,
  [Parameter(Mandatory=$true)][string]$Inventory
)

if(!(Test-Path -LiteralPath $Csv)){ throw "CSV not found: $Csv" }
if(!(Test-Path -LiteralPath $Inventory)){
  Set-Content -LiteralPath $Inventory -Value "game_label,id,brand,master_path,master_start,master_end,playtag,phase,side,formation,notes,tag_right_left,score_impact,player,assist" -Encoding UTF8
}

$rows = Import-Csv -LiteralPath $Csv
$rows | Where-Object {
  $_.master_start -and $_.master_end
} | ForEach-Object {
  # normalize and append
  $line = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}" -f `
    $_.game_label,$_.id,$_.brand,$_.master_path,$_.master_start,$_.master_end,$_.playtag,$_.phase,$_.side,$_.formation,($_.notes -replace ',',';'),$_.tag_right_left,$_.score_impact,$_.player,$_.assist
  Add-Content -LiteralPath $Inventory -Value $line
}
Write-Host "Inventory updated: $Inventory" -ForegroundColor Green
