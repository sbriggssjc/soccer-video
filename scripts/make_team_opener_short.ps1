param(
  [Parameter(Mandatory=$true)] [string]$RosterCsv,
  [Parameter(Mandatory=$true)] [string]$OutFile,
  [int]$FPS = 30,
  [double]$MaxDuration = 5.0
)

$ErrorActionPreference = 'Stop'

# ---- Paths you already standardized ----
$BG          = "C:\Users\scott\soccer-video\brand\tsc\end_card_1080x1920.png"
$BADGE_SOL   = "C:\Users\scott\soccer-video\brand\tsc\badge_clean.png"
$BADGE_HOLE  = "C:\Users\scott\soccer-video\brand\tsc\badge_hole.png"
$FontName    = "Arial Bold"   # use the font family instead of a file path
$FontPath    = "C\:/Windows/Fonts/arialbd.ttf"

# ---- Layout (kept consistent with your opener) ----
$BadgeW      = 900
$BadgeY      = 520
$HoleScale   = 0.58
$HoleBox     = [int][math]::Round($BadgeW * $HoleScale)
$FaceYOffset = -30

# ---- Timings ----
$introDur    = 0.5            # solid logo -> hole logo crossfade
$fadeTxtIn   = 0.15
$fadeTxtOut  = 0.15

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
[1:v]format=rgba,setsar=1,scale=${BadgeW}:-1[solidRef];
[2:v]format=rgba,setsar=1[holeRaw];
[holeRaw][solidRef]scale2ref=w=iw:h=ih[holeMatch][solidMatch];
[solidMatch]format=rgba,fade=t=in:st=0:d=0.25:alpha=1,fade=t=out:st=0.25:d=0.25:alpha=1[badgeSolid];
[holeMatch]format=rgba,fade=t=in:st=0.25:d=0.25:alpha=1[badgeHole];
[0:v][badgeSolid]overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[b1];
[b1][badgeHole]overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[vout]
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
  $fadeTxtInStr = [string]::Format($invCulture, "{0:0.##}", $fadeTxtIn)
  $fadeTxtOutStr = [string]::Format($invCulture, "{0:0.##}", $fadeTxtOut)
  $faceFadeOutStart = [math]::Round([math]::Max(0, $faceDur - 0.05), 2)
  $faceFadeOutStartStr = [string]::Format($invCulture, "{0:0.##}", $faceFadeOutStart)
  $textFadeOutStart = [math]::Round([math]::Max(0, $faceDur - $fadeTxtOut), 2)
  $textFadeOutStartStr = [string]::Format($invCulture, "{0:0.##}", $textFadeOutStart)

  if ([string]::IsNullOrEmpty($name)) {
    $nameUpper = ""
  }
  else {
    $nameUpper = $name.ToUpperInvariant()
  }
  $faceCenterY = $BadgeY + $FaceYOffset
  $escapedName = Escape-Drawtext($nameUpper)
  $escapedNum = Escape-Drawtext($safeNumText)

  $nameFilter = "[t1]drawtext=fontfile='$FontPath':text='${escapedName}':fontsize=52:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=1030,format=rgba,fade=t=in:st=0:d=${fadeTxtInStr}:alpha=1,fade=t=out:st=${textFadeOutStartStr}:d=${fadeTxtOutStr}:alpha=1[nameL];[b1][nameL]overlay=0:0:shortest=1[b2];"
  $numFilter = "[t2]drawtext=fontfile='$FontPath':text='${escapedNum}':fontsize=48:fontcolor=0x9B1B33:x=(w-text_w)/2:y=1110,format=rgba,fade=t=in:st=0:d=${fadeTxtInStr}:alpha=1,fade=t=out:st=${textFadeOutStartStr}:d=${fadeTxtOutStr}:alpha=1[numL];[b2][numL]overlay=0:0:shortest=1[vout]"

  $fcTemplate = @"
[2:v]format=rgba,setsar=1,scale=${BadgeW}:-1[solid];
[3:v]format=rgba,setsar=1,scale=${BadgeW}:-1[hole];

[0:v][solid]overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[introA];
[introA][hole]overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[intro];

[3:v][0:v]scale2ref=w=iw:h=ih[holeMatch][bgRef];
[bgRef][holeMatch]overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[bgHole];

[1:v]scale=${HoleBox}:-1:force_original_aspect_ratio=increase,crop=${HoleBox}:${HoleBox},format=rgba,fade=t=in:st=0:d=0.05:alpha=1,fade=t=out:st=${faceFadeOutStartStr}:d=0.05:alpha=1[face];
[bgHole][face]overlay=x='(W-w)/2':y='${faceCenterY} - h/2':shortest=1[b1];

color=c=black@0.0:s=1080x1920:d=${faceDurStr}[t1];
{0}

color=c=black@0.0:s=1080x1920:d=${faceDurStr}[t2];
{1}
"@

  $fcFace = [string]::Format($fcTemplate, $nameFilter, $numFilter)

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
