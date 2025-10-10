param(
  [Parameter(Mandatory=$true)] [string]$RosterCsv,
  [Parameter(Mandatory=$true)] [string]$OutFile,
  [string]$CoachPhoto = ""
)

# === constants to match your locked-in opener ===
$FPS       = 30
$DUR       = 5.0          # total length
$tX        = 0.40         # solid->hole crossfade start
$xfDur     = 0.40         # crossfade duration

$BadgeW    = 900          # EXACT width for BOTH logos
$BadgeY    = 520          # vertical center
$HoleScale = 0.58         # inner-circle fraction of BadgeW
$HoleBox   = [int]([math]::Round($BadgeW * $HoleScale))
$FaceYOffset = -30        # nudge face if needed
$HoleDX = 0               # fine nudge if ring art is off-center
$HoleDY = 0

# paths (adjust if yours differ)
$BG         = "C:\Users\scott\soccer-video\brand\tsc\end_card_1080x1920.png"
$BADGE_SOL  = "C:\Users\scott\soccer-video\brand\tsc\badge_clean.png"
$BADGE_HOLE = "C:\Users\scott\soccer-video\brand\tsc\badge_hole.png"

# --- build ordered face list (players by jersey number, then coach if provided) ---
$rows = Import-Csv $RosterCsv | Sort-Object { [int](([regex]::Match([string]$_.'PlayerNumber','\d+')).Value) }
$faces = @()
$faces += ($rows | ForEach-Object { $_.PlayerPhoto }) | Where-Object { $_ -and (Test-Path $_) }
if ($CoachPhoto -and (Test-Path $CoachPhoto)) { $faces += $CoachPhoto }

if ($faces.Count -eq 0) {
  throw "No valid face images were found from $RosterCsv (and CoachPhoto='$CoachPhoto')."
}

# target ~4.0s of faces after the 0.4s+0.4s intro (you can tweak)
$faceWindow = [math]::Max(0.1, $DUR - ($tX + $xfDur))   # ~4.2s
$perFace = [math]::Round($faceWindow / $faces.Count, 2) # ~0.38-0.42s typical
if ($perFace -lt 0.12) { $perFace = 0.12 }              # clamp to sane minimum

# concat list for ffmpeg (demuxer requires repeating last file without duration)
$tempDir  = New-Item -ItemType Directory -Force -Path (Join-Path $env:TEMP "tsc_team_opener") | Select-Object -ExpandProperty FullName
$listFile = Join-Path $tempDir "faces.txt"
$lines = @()
for ($i = 0; $i -lt $faces.Count; $i++) {
  $p = $faces[$i]
  $escaped = $p -replace '''',''''''
  $lines += "file '$escaped'"
  # duration lines for all but the very last *listed* entry; we will repeat the last file after the loop
  $lines += "duration $perFace"
}
# repeat last file once per concat-demuxer rules
$lines += "file '$(($faces[-1]) -replace '''','''''')'"

Set-Content -Path $listFile -Value $lines -Encoding Ascii

# --- filter graph (no inline comments; same geometry as your opener) ---
$fc = @"
[2:v]format=rgba,setsar=1,scale=${BadgeW}:-1[solidRef];
[3:v]format=rgba,setsar=1[holeRaw];
[holeRaw][solidRef]scale2ref=w=iw:h=ih[holeMatch][solidMatch];

[solidMatch]fade=t=in:st=0:d=0.4:alpha=1,fade=t=out:st=${tX}:d=${xfDur}:alpha=1[badgeSolid];
[holeMatch] fade=t=in:st=${tX}:d=${xfDur}:alpha=1[badgeHole];

[0:v]fps=${FPS},scale=${HoleBox}:-1:force_original_aspect_ratio=increase,crop=${HoleBox}:${HoleBox},format=rgba[facesSized];

[1:v][facesSized]overlay=x='(W-w)/2 + ${HoleDX}':y='${BadgeY} - h/2 + ${FaceYOffset} + ${HoleDY}':shortest=1[b1];
[b1][badgeSolid]overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[b2];
[b2][badgeHole] overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[vout]
"@

# --- run ffmpeg in one pass: faces concat (in#0) + BG (in#1) + solid (in#2) + hole (in#3) + silent audio (in#4) ---
$bgDur = $DUR.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$cmd = @(
  '-y',
  '-f','concat','-safe','0','-i', $listFile,
  '-loop','1','-t', $bgDur, '-i', $BG,
  '-loop','1','-t', $bgDur, '-i', $BADGE_SOL,
  '-loop','1','-t', $bgDur, '-i', $BADGE_HOLE,
  '-f','lavfi','-t', $bgDur, '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
  '-filter_complex', $fc,
  '-map','[vout]','-map','4:a',
  '-r',"$FPS", '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',
  $OutFile
)

Write-Host "Creating short team opener -> $OutFile"
ffmpeg @cmd
