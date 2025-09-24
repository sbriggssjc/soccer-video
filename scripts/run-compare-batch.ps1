# ==== Run-Compare-Batch.ps1 ====
# Requires: ffmpeg/ffprobe in PATH, your per-clip *.ps1vars files

# --- Make paths independent of current dir ---
$Root    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$inDir   = Join-Path $Root 'out\atomic_clips'
$varsDir = Join-Path $Root 'out\autoframe_work'
$outDir  = Join-Path $Root 'out\reels\tiktok'
New-Item -Force -ItemType Directory $outDir | Out-Null

# -- helper: read FPS as a number (e.g., "24")
function Get-Fps([string]$inPath) {
  $r = ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate `
        -of default=nw=1:nk=1 --% "$inPath"
  if ($LASTEXITCODE -ne 0 -or -not $r) { return 24 }
  if ($r -match '^\s*(\d+)\s*/\s*(\d+)\s*$') { return [int]([double]$Matches[1]/[double]$Matches[2] + 0.5) }
  return [int]$r
}

# --- Helpers (once per script) ---
function Convert-SciToDecimal([string]$s) {
  $pat = '([+-]?(?:\d+(?:\.\d+)?|\.\d+))[eE]([+-]?\d+)'
  [regex]::Replace($s, $pat, {
    param($m)
    $base = [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
    $exp  = [int]$m.Groups[2].Value
    ($base * [math]::Pow(10,$exp)).ToString("0.############################", [Globalization.CultureInfo]::InvariantCulture)
  })
}
function Sanitize([string]$s) { ($s -replace '\s+','') | Convert-SciToDecimal }
function SubN([string]$s, [int]$fps) { $s -replace '\bn\b', "(t*$fps)" }
function Expand-Clip([string]$s) {
  $pat = 'clip\((?<v>(?>[^()]+|\((?<o>)|\)(?<-o>))*(?(o)(?!))),(?<a>(?>[^()]+|\((?<o2>)|\)(?<-o2>))*(?(o2)(?!))),(?<b>(?>[^()]+|\((?<o3>)|\)(?<-o3>))*(?(o3)(?!)))\)'
  while ($s -match $pat) {
    $s = [regex]::Replace($s, $pat, { param($m) "min(max($($m.Groups['v'].Value),$($m.Groups['a'].Value)),$($m.Groups['b'].Value))" })
  }
  $s
}
function Escape-Commas-In-Parens([string]$s) {
  $sb = New-Object System.Text.StringBuilder
  $depth = 0
  foreach ($ch in $s.ToCharArray()) {
    if ($ch -eq '(') { $depth++; [void]$sb.Append($ch); continue }
    if ($ch -eq ')') { $depth=[Math]::Max(0,$depth-1); [void]$sb.Append($ch); continue }
    if ($ch -eq ',' -and $depth -gt 0) { [void]$sb.Append('\,'); continue }
    [void]$sb.Append($ch)
  }
  $sb.ToString()
}

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

  # normalize expressions (cx, cy, z)
  $cxN = Escape-Commas-In-Parens ( SubN ( Sanitize ( Expand-Clip $cxExpr ) ) $fps )
  $cyN = Escape-Commas-In-Parens ( SubN ( Sanitize ( Expand-Clip $cyExpr ) ) $fps )
  $zN  = Escape-Commas-In-Parens ( SubN ( Sanitize ( Expand-Clip $zExpr  ) ) $fps )

  # build crop expressions (named args; ih/iw; even dims)
  $w  = "floor((((ih*9/16)/($zN)))/2)*2"
  $h  = "floor(((ih/($zN)))/2)*2"
  $x  = "($cxN)-($w)/2"
  $y  = "($cyN)-($h)/2"

  # final escaping pass for any commas introduced by min/max
  $w = Escape-Commas-In-Parens $w
  $h = Escape-Commas-In-Parens $h
  $x = Escape-Commas-In-Parens $x
  $y = Escape-Commas-In-Parens $y

  # final filter (right = portrait crop; left = original; hstack)
  $vf = "[0:v]split=2[left][right];" +
        "[right]crop=w='$w':h='$h':x='$x':y='$y',scale=w=-2:h=1080:flags=lanczos,setsar=1[right_portrait];" +
        "[left][right_portrait]hstack=inputs=2,format=yuv420p"

  $inPath  = Join-Path $inDir  $_.Name
  $outPath = Join-Path $outDir ("COMPARE__" + $stem + ".mp4")

  Write-Host "`n>> Processing $($_.Name)  (fps=$fps)"
  & ffmpeg -hide_banner -y -nostdin -i $inPath -filter_complex $vf `
    -c:v libx264 -crf 20 -preset veryfast -movflags +faststart -an $outPath
}

Write-Host "`nAll done."
