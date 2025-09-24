# ==== Run-Compare-Batch.ps1 ====
# Requires: ffmpeg/ffprobe in PATH, your per-clip *.ps1vars files

# --- Stable roots/dirs ---
$ErrorActionPreference = 'Stop'
$Root    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ffmpeg  = 'ffmpeg'
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
function Sanitize([string]$s) { Convert-SciToDecimal (($s -replace '\s+','')) }
function SubN([string]$s, [int]$fps) { $s -replace '\bn\b', "(t*$fps)" }
function Expand-Clip([string]$s) {
  $pat = 'clip\((?<v>(?>[^()]+|\((?<o>)|\)(?<-o>))*(?(o)(?!))),(?<a>(?>[^()]+|\((?<o2>)|\)(?<-o2>))*(?(o2)(?!))),(?<b>(?>[^()]+|\((?<o3>)|\)(?<-o3>))*(?(o3)(?!)))\)'
  while ($s -match $pat) {
    $s = [regex]::Replace($s, $pat, {
      param($m)
      "min(max($($m.Groups['v'].Value),$($m.Groups['a'].Value)),$($m.Groups['b'].Value))"
    })
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

  # --- fps default ---
  if (-not $fps) { $fps = 24 }

  # --- 0) Start from raw strings; remove legacy '\,' if present ---
  $cxRaw = $cxExpr -replace '\\,', ','
  $cyRaw = $cyExpr -replace '\\,', ','
  $zRaw  = $zExpr  -replace '\\,', ','

  # --- 1) Expand clip(v,a,b) -> min(max(v,a),b) (handles nesting) ---
  $cx1 = Expand-Clip $cxRaw
  $cy1 = Expand-Clip $cyRaw
  $z1  = Expand-Clip $zRaw

  # --- 2) Sanitize numbers + replace n with t*fps ---
  $cx2 = SubN (Sanitize $cx1) $fps
  $cy2 = SubN (Sanitize $cy1) $fps
  $z2  = SubN (Sanitize $z1)  $fps

  # --- 3) Escape commas ONLY inside parentheses (so filter commas stay intact) ---
  $cxN = Escape-Commas-In-Parens $cx2
  $cyN = Escape-Commas-In-Parens $cy2
  $zN  = Escape-Commas-In-Parens $z2

  # Safety: make sure clip() really disappeared
  if ($cxN -match 'clip\(' -or $cyN -match 'clip\(' -or $zN -match 'clip\(') {
    throw "clip() still present after expansion"
  }

  # --- 4) Build portrait window using ih/iw (even sizes) ---
  $w = "floor((((ih*9/16)/($zN)))/2)*2"
  $h = "floor(((ih/($zN)))/2)*2"
  $x = "($cxN)-($w)/2"
  $y = "($cyN)-($h)/2"

  # Final escape pass (min/max introduced commas)
  $w = Escape-Commas-In-Parens $w
  $h = Escape-Commas-In-Parens $h
  $x = Escape-Commas-In-Parens $x
  $y = Escape-Commas-In-Parens $y

  # --- 5) Named crop + quotes; then scale/setsar/format ---
  $filter = "[0:v]crop=w='$w':h='$h':x='$x':y='$y',scale=w=-2:h=1080:flags=lanczos,setsar=1,format=yuv420p"

  # --- 6) IO paths ---
  $inPath  = Join-Path $inDir  $_.Name
  $outPath = Join-Path $outDir ("COMPARE__" + $stem + ".mp4")

  if (-not (Test-Path $inPath)) { Write-Warning "Missing input: $inPath"; return }
  Write-Host "`n>> Processing $($_.Name)  (fps=$fps)"
  Write-Host ">> filter: $filter"

  & $ffmpeg -hide_banner -y -nostdin -i $inPath -filter_complex $filter `
    -c:v libx264 -crf 20 -preset veryfast -movflags +faststart -an $outPath
  if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed on $($_.Name)" }
}

Write-Host "`nAll done."
