param(
  [Parameter(Mandatory=$true)] [string]$HighlightsDir,
  [Parameter(Mandatory=$true)] [string]$FinalsDir,
  [Parameter(Mandatory=$true)] [string]$OpenersDir,               # where individual openers live
  [Parameter(Mandatory=$true)] [string]$TeamOpener,               # TSC__Team_Montage.mp4
  [int]$FPS = 30
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $FinalsDir | Out-Null

Get-ChildItem $HighlightsDir -File -Filter *.mp4 | ForEach-Object {
  $hl = $_
  $useOpener = $TeamOpener

  if ($hl.BaseName -match '^(?<num>\d{1,2})__?(?<first>[A-Za-z]+)_(?<last>[A-Za-z]+)') {
    $num   = $matches.num
    $name  = "$($matches.first)_$($matches.last)"
    $candidate = Join-Path $OpenersDir "$num__$name.mp4"
    if (Test-Path $candidate) { $useOpener = $candidate }
  }

  $final = Join-Path $FinalsDir ($hl.BaseName + "__WITH_OPENER.mp4")
  Write-Host "Stitching -> $final"
  ffmpeg -y -i "$useOpener" -i "$($hl.FullName)" `
  -filter_complex `
  "[0:v]fps=$FPS,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[v0]; `
   [1:v]fps=$FPS,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[v1]; `
   [0:a]aresample=48000,apad[a0]; [1:a]aresample=48000[a1]; `
   [v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]" `
  -map "[v]" -map "[a]" -c:v libx264 -pix_fmt yuv420p -c:a aac -r $FPS "$final"
}
