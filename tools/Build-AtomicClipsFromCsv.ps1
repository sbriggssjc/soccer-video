param(
  [Parameter(Mandatory=$true)][string]$Csv,
  [Parameter(Mandatory=$true)][string]$OutRoot,
  [string]$Brand = "tsc"
)

$FFMPEG = "ffmpeg"
$rows = Import-Csv -LiteralPath $Csv
foreach($r in $rows){
  if([string]::IsNullOrWhiteSpace($r.master_start) -or [string]::IsNullOrWhiteSpace($r.master_end)){ continue }
  $id    = "{0}" -f $r.id
  $g     = $r.game_label
  $dstDir= Join-Path $OutRoot $g
  if(!(Test-Path $dstDir)){ New-Item -ItemType Directory -Path $dstDir | Out-Null }
  $clip  = Join-Path $dstDir ("{0}__{1}__{2}-{3}.mp4" -f $id,$r.playtag,([string]$r.master_start).Replace('.','p'),([string]$r.master_end).Replace('.','p'))
  & $FFMPEG -hide_banner -y -ss ([double]$r.master_start) -to ([double]$r.master_end) -i "$($r.master_path)" -c copy "$clip"
  if($LASTEXITCODE -ne 0){
    # Fallback re-encode if stream copy fails
    & $FFMPEG -hide_banner -y -ss ([double]$r.master_start) -to ([double]$r.master_end) -i "$($r.master_path)" `
      -vf "scale=1920:-2:flags=lanczos,setsar=1" -r 60 -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k -movflags +faststart "$clip"
  }
  Write-Host "Wrote $clip"
}
