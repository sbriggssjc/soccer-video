# ==== Run-Compare-Batch.ps1 ====
# Requires: ffmpeg/ffprobe in PATH, your per-clip *.ps1vars files

# -- helper: read FPS as a number (e.g., "24")
function Get-Fps([string]$inPath) {
  $r = ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate `
        -of default=nw=1:nk=1 --% "$inPath"
  if ($LASTEXITCODE -ne 0 -or -not $r) { return 24 } # fallback
  if ($r -match '^\s*(\d+)\s*/\s*(\d+)\s*$') { return [int]([double]$Matches[1]/[double]$Matches[2] + 0.5) }
  return [int]$r
}

# -- helpers: normalize expressions
function Convert-SciToDecimal([string]$s) {
  $p='([+-]?(?:\d+(?:\.\d+)?|\.\d+))[eE]([+-]?\d+)'
  [regex]::Replace($s,$p,{
    param($m)
    $base=[double]::Parse($m.Groups[1].Value,[Globalization.CultureInfo]::InvariantCulture)
    $exp=[int]$m.Groups[2].Value
    ($base * [math]::Pow(10,$exp)).ToString("0.############################",[Globalization.CultureInfo]::InvariantCulture)
  })
}
function Sanitize([string]$s){ Convert-SciToDecimal ($s -replace '\s+','') }
function SubN([string]$s,[int]$fps){ $s -replace '\bn\b',"(t*$fps)" }
function ExpandClip([string]$s){
  $pat='clip\((?<v>(?>[^()]+|\((?<o>)|\)(?<-o>))*(?(o)(?!))),(?<a>(?>[^()]+|\((?<o2>)|\)(?<-o2>))*(?(o2)(?!))),(?<b>(?>[^()]+|\((?<o3>)|\)(?<-o3>))*(?(o3)(?!)))\)'
  while($s -match $pat){
    $s=[regex]::Replace($s,$pat,{ param($m) "min(max($($m.Groups['v'].Value),$($m.Groups['a'].Value)),$($m.Groups['b'].Value))" })
  }; $s
}
# escape commas ONLY inside min(...) or max(...)
function EscapeCommasInMinMax([string]$s){
  return [regex]::Replace($s,'\b(min|max)\((?<inner>(?>[^()]+|\((?<o>)|\)(?<-o>))*(?(o)(?!)))\)',{
    param($m)
    $inner = ($m.Groups['inner'].Value -replace ',', '\,')
    "$($m.Groups[1].Value)($inner)"
  })
}

# -- I/O roots
$inDir   = ".\out\atomic_clips"
$varsDir = ".\out\autoframe_work"
$outDir  = ".\out\reels\tiktok"
New-Item -Force -ItemType Directory $outDir | Out-Null

# -- loop all atomic clips
Get-ChildItem $inDir -File -Filter "*.mp4" | ForEach-Object {
  $in  = $_.FullName
  $fps = Get-Fps $in

  # map:  "...\X__NAME__tA-tB.mp4"  ->  "...\X__NAME__tA-tB_zoom.ps1vars"
  $stem   = [IO.Path]::GetFileNameWithoutExtension($_.Name)
  $vars   = Join-Path $varsDir ($stem + "_zoom.ps1vars")
  if (!(Test-Path $vars)) { Write-Warning "No vars for $($_.Name) -> $vars"; return }

  # load cxExpr, cyExpr, zExpr
  Invoke-Expression (Get-Content -Raw -Encoding UTF8 $vars)
  if (-not $cxExpr -or -not $cyExpr -or -not $zExpr) { Write-Warning "Bad vars in $vars"; return }

  # normalize inputs
  $cx = EscapeCommasInMinMax ( SubN ( Sanitize ( ExpandClip $cxExpr ) ) $fps )
  $cy = EscapeCommasInMinMax ( SubN ( Sanitize ( ExpandClip $cyExpr ) ) $fps )
  $z  = EscapeCommasInMinMax ( SubN ( Sanitize ( ExpandClip $zExpr  ) ) $fps )

  # build crop expressions (named args; ih/iw; even dims)
  $w  = "floor(((ih*9/16)/($z))/2)*2"
  $h  = "floor((ih/($z))/2)*2"
  $x  = "($cx)-(($w)/2)"
  $y  = "($cy)-(($h)/2)"

  # final filter (right = portrait crop; left = original; hstack)
  $vf = "[0:v]split=2[left][right];" +
        "[right]crop=w='$w':h='$h':x='$x':y='$y',scale=w=-2:h=1080:flags=lanczos,setsar=1[right_portrait];" +
        "[left][right_portrait]hstack=inputs=2,format=yuv420p"

  $out = Join-Path $outDir ("COMPARE__" + $stem + ".mp4")
  Write-Host "`n>> Processing $($_.Name)  (fps=$fps)"
  ffmpeg -hide_banner -y -nostdin -i "$in" -filter_complex "$vf" `
         -c:v libx264 -crf 20 -preset veryfast -movflags +faststart -an "$out"
}

Write-Host "`nAll done."
