param(
  [string]$Concat = ".\out\concat_goals_plus_top.txt",
  [string]$Preset = "tiktok_9x16",
  [string]$OutFile = ".\out\reel.mp4",
  [string]$PresetsJson = ".\config\reels.json"
)
$cfg = Get-Content $PresetsJson | ConvertFrom-Json
$p = $cfg.presets.$Preset
if ($null -eq $p) { throw "Unknown preset: $Preset" }
$vf = $p.vf; if ([string]::IsNullOrEmpty($vf)) { $vf = "scale=$($p.width):-2:flags=lanczos" }
ffmpeg -f concat -safe 0 -i $Concat -r $($p.fps) -g ($p.fps*2) -vf $vf -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -c:a aac -ar 48000 -af $p.afilters -movflags +faststart -y $OutFile
Write-Host "[rendered] $Preset -> $OutFile"
