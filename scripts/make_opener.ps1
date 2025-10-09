Param(
    [Parameter(Mandatory=$true)] [string]$PlayerName,
    [Parameter(Mandatory=$true)] [int]$PlayerNumber,
    [Parameter(Mandatory=$true)] [string]$PlayerPhoto,
    [Parameter(Mandatory=$true)] [string]$OutDir,

    [string]$BGPath = "C:\Users\scott\soccer-video\brand\tsc\end_card_1080x1920.png",
    [string]$BadgeSolidPath = "C:\Users\scott\soccer-video\brand\tsc\badge_clean.png",
    [string]$BadgeHolePath = "C:\Users\scott\soccer-video\brand\tsc\badge_hole.png",

    [int]$FPS = 30,
    [double]$DUR = 6.0,
    [double]$tX = 1.00,
    [double]$xfDur = 1.00,

    [int]$BadgeW = 900,
    [int]$BadgeY = 520,
    [double]$HoleScale = 0.58,
    [int]$FaceYOffset = -30,
    [int]$HoleDX = 0,
    [int]$HoleDY = 0,

    [string]$FontPath = "C:/WINDOWS/Fonts/arialbd.ttf",
    [double]$FadeInTxt = 0.60,
    [double]$FadeOutTxt = 0.60
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$HoleBox = [int]([math]::Round($BadgeW * $HoleScale))
$TxtOutStart = [math]::Round(($DUR - $FadeOutTxt),2)
$SanName = ($PlayerName -replace '[^\w]+','').Trim('')
$OUT = Join-Path $OutDir "$SanName__OPENER.mp4"

# Filter graph (single-line stages; no inline comments)
$fc = @"
[2:v]format=rgba,setsar=1,scale=${BadgeW}:-1[solidRef];
[3:v]format=rgba,setsar=1[holeRaw];
[holeRaw][solidRef]scale2ref=w=iw:h=ih[holeMatch][solidMatch];
[solidMatch]fade=t=in:st=0:d=0.6:alpha=1,fade=t=out:st=${tX}:d=${xfDur}:alpha=1[badgeSolid];
[holeMatch]fade=t=in:st=${tX}:d=${xfDur}:alpha=1[badgeHole];
[1:v]scale=${HoleBox}:-1:force_original_aspect_ratio=increase,crop=${HoleBox}:${HoleBox},format=rgba,fade=t=in:st=${tX}:d=${xfDur}:alpha=1[face];
[0:v][face]overlay=x='(W-w)/2 + ${HoleDX}':y='${BadgeY} - h/2 + ${FaceYOffset} + ${HoleDY}':shortest=1[b1];
[b1][badgeSolid]overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[b2];
[b2][badgeHole]overlay=x='(W-w)/2':y='${BadgeY} - h/2':shortest=1[vbase];
color=c=black@0.0:s=1080x1920:d=${DUR}[t1];
[t1]drawtext=fontfile='${FontPath}':text='$($PlayerName.ToUpper())':fontsize=72:fontcolor=0xFFFFFF:x=(w-text_w)/2:y=1030,format=rgba,fade=t=in:st=0:d=${FadeInTxt}:alpha=1,fade=t=out:st=${TxtOutStart}:d=${FadeOutTxt}:alpha=1[nameL];
[vbase][nameL]overlay=0:0:shortest=1[b3];
color=c=black@0.0:s=1080x1920:d=${DUR}[t2];
[t2]drawtext=fontfile='${FontPath}':text='#${PlayerNumber}':fontsize=66:fontcolor=0x9B1B33:x=(w-text_w)/2:y=1110,format=rgba,fade=t=in:st=0:d=${FadeInTxt}:alpha=1,fade=t=out:st=${TxtOutStart}:d=${FadeOutTxt}:alpha=1[numL];
[b3][numL]overlay=0:0:shortest=1[vout]
"@

# Run ffmpeg
$ff = @(
    "-y",
    "-loop","1","-t",$DUR,"-i",$BGPath,
    "-loop","1","-t",$DUR,"-i",$PlayerPhoto,
    "-loop","1","-t",$DUR,"-i",$BadgeSolidPath,
    "-loop","1","-t",$DUR,"-i",$BadgeHolePath,
    "-f","lavfi","-t",$DUR,"-i","anullsrc=channel_layout=stereo:sample_rate=48000",
    "-filter_complex",$fc,
    "-map","[vout]","-map","4:a","-r",$FPS,"-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",$OUT
)

Write-Host "Rendering opener -> $OUT"
ffmpeg @ff
