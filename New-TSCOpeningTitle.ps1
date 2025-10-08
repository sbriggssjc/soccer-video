[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)] [string]$PlayerImg,
  [Parameter(Mandatory=$true)] [string]$PlayerName,
  [Parameter(Mandatory=$true)] [string]$PlayerNo,
  [string]$ReelIn,
  [string]$ReelOut,
  [string]$OutOpener = "C:\Users\scott\soccer-video\out\opener\TSC_Opener__${PlayerNo}_${PlayerName}.mp4",
  [int]$FPS = 30,
  [double]$Dur = 3.5,
  [int]$BadgeSize = 900,
  [int]$BadgeY = 520,
  [int]$FaceBox = 1080,
  [int]$FaceOffsetX = 0,
  [int]$FaceOffsetY = 0
)

# If FaceBox wasn't passed, match the badge size by default
if (-not $PSBoundParameters.ContainsKey('FaceBox')) { $FaceBox = $BadgeSize }

# Brand assets
$BG    = "C:\Users\scott\soccer-video\brand\tsc\end_card_1080x1920.png"
$Badge = "C:\Users\scott\soccer-video\brand\tsc\badge_clean.png"

# Fonts (Montserrat ExtraBold preferred; fallback Arial Bold)
$BoldFontFF = (Join-Path $env:WINDIR 'Fonts\Montserrat-ExtraBold.ttf')
if (!(Test-Path $BoldFontFF)) { $BoldFontFF = (Join-Path $env:WINDIR 'Fonts\arialbd.ttf') }
$BoldFontFF = $BoldFontFF.Replace('\','/') -replace ':','\:'  # e.g., C\:/Windows/Fonts/arialbd.ttf

# Brand colors
$White = "0xFFFFFF"
$Red   = "0x9B1B33"

# Validate inputs
foreach ($p in @($PlayerImg, $BG, $Badge)) {
  if (!(Test-Path $p)) { throw "Missing required file: $p" }
}

$DurFrames    = [int]([math]::Round($Dur * $FPS))
$FadeInFace   = 0.8
$FadeOutFace  = 0.7
$FadeInBadge  = 0.6
$FadeOutBadge = 0.6

$NameY = [int]($BadgeY + $BadgeSize/2 + 60)
$NumY  = [int]($BadgeY + $BadgeSize/2 + 140)
$TextFade = 0.35

$null = New-Item -ItemType Directory -Force -Path (Split-Path $OutOpener)

ffmpeg -y `
 -loop 1 -t $Dur -i "$BG" `
 -loop 1 -t $Dur -i "$PlayerImg" `
 -loop 1 -t $Dur -i "$Badge" `
 -f lavfi -t $Dur -i anullsrc=channel_layout=stereo:sample_rate=48000 `
 -filter_complex "
   [1]scale=${FaceBox}:${FaceBox}:force_original_aspect_ratio=increase,
      crop=${FaceBox}:${FaceBox},
      fps=${FPS},
      zoompan=z='min(1.10,1.0+0.03*on/${DurFrames})':d=${DurFrames}:s=${FaceBox}x${FaceBox},
      format=rgba,
      fade=t=in:st=0:d=${FadeInFace}:alpha=1,
      fade=t=out:st=$(($Dur-$FadeOutFace)):d=${FadeOutFace}:alpha=1[face];

   [2]scale=${BadgeSize}:-1,
      format=rgba,
      colorkey=0xFFFFFF:0.12:0.02,
      fade=t=in:st=0:d=${FadeInBadge}:alpha=1,
      fade=t=out:st=$(($Dur-$FadeOutBadge)):d=${FadeOutBadge}:alpha=1[badgecut];

   [0][face]overlay=x='(W-w)/2+${FaceOffsetX}':y='${BadgeY}-h/2+${FaceOffsetY}':shortest=1[bgface];

   [bgface][badgecut]overlay=x='(W-w)/2':y='${BadgeY}-h/2':shortest=1,
     drawtext=fontfile='${BoldFontFF}':text='$($PlayerName.ToUpper())':fontsize=72:fontcolor=${White}:x=(w-text_w)/2:y=${NameY}:alpha='if(lt(t,${TextFade}),t/${TextFade},if(lt(t,${Dur}-${TextFade}),1,max(0,(${Dur}-t)/${TextFade})))',
     drawtext=fontfile='${BoldFontFF}':text='\#$($PlayerNo)':fontsize=66:fontcolor=${Red}:x=(w-text_w)/2:y=${NumY}:alpha='if(lt(t,${TextFade}),t/${TextFade},if(lt(t,${Dur}-${TextFade}),1,max(0,(${Dur}-t)/${TextFade})))'
 " `
 -map 3:a -c:a aac -shortest `
 -c:v libx264 -r $FPS -pix_fmt yuv420p "$OutOpener"

if ($LASTEXITCODE -ne 0) { throw "FFmpeg failed for $PlayerName #$PlayerNo (exit $LASTEXITCODE)" }

Write-Host "Opener created -> $OutOpener"

if ($ReelIn) {
  if (!(Test-Path $ReelIn)) { throw "ReelIn not found: $ReelIn" }
  if (-not $ReelOut) {
    $ReelOut = [IO.Path]::ChangeExtension($ReelIn, $null) + "__WITH_OPENER.mp4"
  }
  ffmpeg -y -i "$OutOpener" -i "$ReelIn" -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]" -map "[v]" -map "[a]" `
    -c:v libx264 -r $FPS -pix_fmt yuv420p -c:a aac "$ReelOut"
  if ($LASTEXITCODE -ne 0) { throw "Concat failed: $LASTEXITCODE" }
  Write-Host "Done! Reel with opener -> $ReelOut"
}

