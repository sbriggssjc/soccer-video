param(
  [Parameter(Mandatory=$true)] [string]$RosterCsv,                             # PlayerName,PlayerNumber,PlayerPhoto
  [Parameter(Mandatory=$true)] [string]$OutFile,                                # e.g. out\opener\TSC__Team_Montage.mp4
  [string]$BadgeSolid = "C:\\Users\\scott\\soccer-video\\brand\\tsc\\badge_clean.png",
  [string]$BadgeHole  = "C:\\Users\\scott\\soccer-video\\brand\\tsc\\badge_hole.png",
  [string]$Background = "C:\\Users\\scott\\soccer-video\\brand\\tsc\\end_card_1080x1920.png",
  [string]$CoachPhoto = "",                                                     # optional – add coach at end
  [int]$FPS = 30,
  [double]$PerCardDur = 0.7,                                                    # “rapid fire” feel
  [double]$Crossfade = 0.0,                                                     # keep at 0 for hard cuts; can set to 0.12 if you want fades
  [int]$BadgeW = 900,
  [int]$BadgeY = 520,
  [double]$HoleScale = 0.58,
  [int]$FaceYOffset = -30
)

# --- prep
$ErrorActionPreference = "Stop"
$segDir = Join-Path ([System.IO.Path]::GetDirectoryName($OutFile)) "_montage_segments"
New-Item -ItemType Directory -Force -Path $segDir | Out-Null

# inner circle box
$HoleBox = [Math]::Round($BadgeW * $HoleScale)

# base filtergraph (no name/number text; identical geometry; small solid->hole crossfade)
$xfStart = 0.20
$xfDur   = 0.35

$fc = @"
[2:v]format=rgba,setsar=1,scale=${BadgeW}:-1[solidRef];
[3:v]format=rgba,setsar=1[holeRaw];
[holeRaw][solidRef]scale2ref=w=iw:h=ih[holeMatch][solidMatch];

[solidMatch]fade=t=in:st=0:d=0.15:alpha=1,fade=t=out:st=${xfStart}:d=${xfDur}:alpha=1[badgeSolid];
[holeMatch] fade=t=in:st=${xfStart}:d=${xfDur}:alpha=1[badgeHole];

[1:v]scale=${HoleBox}:-1:force_original_aspect_ratio=increase,
     crop=${HoleBox}:${HoleBox},format=rgba,
     fade=t=in:st=${xfStart}:d=${xfDur}:alpha=1[face];

[0:v][face]overlay=x='(W-w)/2':y='${BadgeY} - h/2 + ${FaceYOffset}':shortest=1[b1];
[b1][badgeSolid]overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[b2];
[b2][badgeHole] overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[vout]
"@

# --- build per-person segments
$rows = Import-Csv $RosterCsv
$segments = @()

foreach ($row in $rows) {
  $name = [string]$row.PlayerName
  $photo = [string]$row.PlayerPhoto
  if (-not (Test-Path $photo)) { Write-Warning "Missing photo for $name -> $photo"; continue }

  $safeName = ($name -replace '[^A-Za-z0-9_ -]','_') -replace '\s+','_'
  $seg = Join-Path $segDir ("{0}.mp4" -f $safeName)
  $segments += $seg

  ffmpeg -y `
    -loop 1 -t $PerCardDur -i "$Background" `
    -loop 1 -t $PerCardDur -i "$photo" `
    -loop 1 -t $PerCardDur -i "$BadgeSolid" `
    -loop 1 -t $PerCardDur -i "$BadgeHole" `
    -f lavfi -t $PerCardDur -i "anullsrc=channel_layout=stereo:sample_rate=48000" `
    -filter_complex $fc `
    -map "[vout]" -map 4:a -r $FPS -c:v libx264 -pix_fmt yuv420p -c:a aac "$seg"
}

# optional coach
if ($CoachPhoto -and (Test-Path $CoachPhoto)) {
  $seg = Join-Path $segDir "_Coach.mp4"
  $segments += $seg
  ffmpeg -y `
    -loop 1 -t $PerCardDur -i "$Background" `
    -loop 1 -t $PerCardDur -i "$CoachPhoto" `
    -loop 1 -t $PerCardDur -i "$BadgeSolid" `
    -loop 1 -t $PerCardDur -i "$BadgeHole" `
    -f lavfi -t $PerCardDur -i "anullsrc=channel_layout=stereo:sample_rate=48000" `
    -filter_complex $fc `
    -map "[vout]" -map 4:a -r $FPS -c:v libx264 -pix_fmt yuv420p -c:a aac "$seg"
}

# --- concat the segments
$listFile = Join-Path $segDir "files.txt"
Set-Content -Path $listFile -Value ($segments | ForEach-Object { "file '$($_.Replace("'", "''"))'" })

# Use -c copy since all segments share the same encoding/params
ffmpeg -y -safe 0 -f concat -i "$listFile" -c copy "$OutFile"
Write-Host "Team montage created -> $OutFile"
