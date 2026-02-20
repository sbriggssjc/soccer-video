param(
  [Parameter(Mandatory=$true)] [string]$RosterCsv,
  [Parameter(Mandatory=$true)] [string]$OutFile,
  [int]$FPS = 30,
  [double]$MaxDuration = 5.0
)

$ErrorActionPreference = 'Stop'

# ---- Paths you already standardized ----
$RepoRoot    = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BG          = Join-Path $RepoRoot "brand\tsc\end_card_1080x1920.png"
$BADGE_SOL   = Join-Path $RepoRoot "brand\tsc\badge_clean.png"
$BADGE_HOLE  = Join-Path $RepoRoot "brand\tsc\badge_hole.png"
$FontName    = "Arial Bold"   # use the font family instead of a file path
$FontPath    = "C\:/Windows/Fonts/arialbd.ttf"

# ---- Timings ----
$introDur    = 0.5            # solid logo -> hole logo crossfade

# ---- Prep workspace ----
$work = Join-Path ([System.IO.Path]::GetDirectoryName($OutFile)) "_team_short_tmp"
New-Item -ItemType Directory -Force -Path $work | Out-Null

# ---- Load roster and coerce types ----
$rows = Import-Csv $RosterCsv | ForEach-Object {
  [pscustomobject]@{
    PlayerName   = [string]$_.'PlayerName'
    PlayerNumber = ([int]([regex]::Match([string]$_.'PlayerNumber','\d+').Value))
    PlayerPhoto  = [string]$_.'PlayerPhoto'
  }
} | Sort-Object PlayerNumber

# ---- Append coach if present in the photo folder ----
$photoFolder = Split-Path -Parent ($rows[0].PlayerPhoto)
$coachPhoto = Get-ChildItem -Path $photoFolder -File -Include *Coach*.* -ErrorAction SilentlyContinue | Select-Object -First 1
if ($coachPhoto) {
  $rows += [pscustomobject]@{
    PlayerName   = 'COACH'
    PlayerNumber = $null
    PlayerPhoto  = $coachPhoto.FullName
  }
}

if (!$rows -or $rows.Count -eq 0) { throw "No roster rows found." }

# ---- Compute per-face duration within MaxDuration budget ----
$faces = $rows.Count
$faceDur = [math]::Max(0.25, ($MaxDuration - $introDur) / $faces)  # clamp to at least 0.25s
$faceDur = [math]::Round($faceDur, 2)

$invCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Escape-Drawtext([string]$text) {
  if ($null -eq $text) { return "" }
  $escaped = $text -replace ":", "\\:"
  $escaped = $escaped -replace "'", "\\'"
  return $escaped
}

Write-Host "Intro: $introDur s | Each face: $faceDur s | People: $faces | Target <= $MaxDuration s"

# ---- 1) Build the INTRO segment (solid -> hole; no face yet) ----
$introMp4 = Join-Path $work "_00_intro.mp4"
$fcIntro = @"
[0:v]format=rgba,setsar=1[bg];
[1:v]format=rgba,setsar=1,scale=900:-1[solid];
[2:v]format=rgba,setsar=1,scale=900:-1[hole];

# 0.00–0.25s: solid badge; 0.25–0.50s: hole badge
[solid]fade=t=in:st=0:d=0.12:alpha=1,fade=t=out:st=0.13:d=0.12:alpha=1[solidA];
[hole] fade=t=in:st=0.25:d=0.12:alpha=1,fade=t=out:st=0.38:d=0.12:alpha=1[holeA];

[bg][solidA]overlay=x='(W-w)/2':y='520 - h/2':shortest=1[tmp];
[tmp][holeA] overlay=x='(W-w)/2':y='520 - h/2':shortest=1[vout]
"@

ffmpeg -hide_banner -y `
  -loop 1 -t $introDur -i "$BG" `
  -loop 1 -t $introDur -i "$BADGE_SOL" `
  -loop 1 -t $introDur -i "$BADGE_HOLE" `
  -f lavfi -t $introDur -i "anullsrc=channel_layout=stereo:sample_rate=48000" `
  -filter_complex $fcIntro `
  -map "[vout]" -map 3:a -r $FPS -c:v libx264 -pix_fmt yuv420p -c:a aac "$introMp4" | Out-Null

# ---- 2) Build each FACE segment under the logo-with-hole ----
$tempParts = New-Object System.Collections.Generic.List[string]
$tempParts.Add($introMp4)

$idx = 1
foreach ($row in $rows) {
  $name = [string]$row.PlayerName
  $num  = $row.PlayerNumber
  $photo = [string]$row.PlayerPhoto

  $safeNumText = $(if ($null -ne $num) { "\#$num" } else { " " })
  $seg = Join-Path $work ("_{0:d2}_{1}.mp4" -f $idx, ($name -replace '\s','_'))

  $faceDurStr = [string]::Format($invCulture, "{0:0.##}", $faceDur)
  if ([string]::IsNullOrEmpty($name)) {
    $nameUpper = ""
  }
  else {
    $nameUpper = $name.ToUpperInvariant()
  }
  $escapedName = Escape-Drawtext($nameUpper)
  $escapedNum = Escape-Drawtext($safeNumText)
  $fcFace = @"
[0:v]format=rgba,setsar=1[bg];
[3:v]format=rgba,setsar=1,scale=900:-1[badgeHole];

# keep only the hole on the background
[bg][badgeHole]overlay=x='(W-w)/2':y='520 - h/2':shortest=1[bgHole];

# face in the ring (no fades)
[1:v]scale=522:-1:force_original_aspect_ratio=increase,crop=522:522,format=rgba[face];
[bgHole][face]overlay=x='(W-w)/2':y='520 - h/2 - 30':shortest=1[b1];

# name (no fades)
color=c=black@0.0:s=1080x1920:d=${faceDurStr}[t1];
[t1]drawtext=fontfile='$FontPath':text='${escapedName}':fontsize=52:fontcolor=0xFFFFFF:
    x=(w-text_w)/2:y=1030,format=rgba[nameL];
[b1][nameL]overlay=0:0:shortest=1[b2];

# number (no fades)
color=c=black@0.0:s=1080x1920:d=${faceDurStr}[t2];
[t2]drawtext=fontfile='$FontPath':text='${escapedNum}':fontsize=48:fontcolor=0x9B1B33:
    x=(w-text_w)/2:y=1110,format=rgba[numL];
[b2][numL]overlay=0:0:shortest=1[vout]
"@

  ffmpeg -hide_banner -y `
    -loop 1 -t $faceDur -i "$BG" `
    -loop 1 -t $faceDur -i "$photo" `
    -loop 1 -t $faceDur -i "$BADGE_SOL" `
    -loop 1 -t $faceDur -i "$BADGE_HOLE" `
    -f lavfi -t $faceDur -i "anullsrc=channel_layout=stereo:sample_rate=48000" `
    -filter_complex $fcFace `
    -map "[vout]" -map 4:a -r $FPS -c:v libx264 -pix_fmt yuv420p -c:a aac "$seg" | Out-Null

  $tempParts.Add($seg)
  $idx++
}

# ---- 3) Concat everything ----
$listPath = Join-Path $work "concat_list.txt"
Set-Content -Path $listPath -Value ($tempParts | ForEach-Object { "file '$($_)'" })

ffmpeg -hide_banner -y -f concat -safe 0 -i $listPath -c:v libx264 -pix_fmt yuv420p -r $FPS -c:a aac "$OutFile"

Write-Host "Done -> $OutFile"

# Optional: cleanup temp
# Remove-Item -Recurse -Force $work
