param(
  [switch]$SkipMissingVars,
  [switch]$DefaultIfMissing,
  [int]$Fps = 24,
  [double]$kSpeed = 0.02,
  [int]$LeadFrames = 6,
  [double]$zMin = 1.20,
  [double]$zMax = 2.8,
  [double]$zBias = 0.25
)

Write-Host ("Options: SkipMissingVars={0}  DefaultIfMissing={1}" -f `
  $SkipMissingVars.IsPresent, $DefaultIfMissing.IsPresent)

# ==== Run-Compare-Batch.ps1 ====
# Requires: ffmpeg/ffprobe in PATH, your per-clip *.ps1vars files

# --- Stable roots/dirs ---
$ErrorActionPreference = 'Continue'
$Root    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
$ffmpeg  = 'ffmpeg'
$inDir   = (Resolve-Path (Join-Path $Root 'out\atomic_clips')).Path
$varsDir = (Resolve-Path (Join-Path $Root 'out\autoframe_work')).Path
$outDir  = Join-Path $Root 'out\reels\tiktok'
New-Item -Force -ItemType Directory $outDir | Out-Null

# Behavior when a clip has no ps1vars
$Skipped = 0
$Defaulted = 0
$Failed = 0

# --- helpers (self-contained, no session priming needed) ---
function Unquote([string]$s) {
  if ($null -eq $s) { return $s }
  if ($s -match '^\s*"(.*)"\s*$') { return $Matches[1] }
  if ($s -match "^\s*'(.*)'\s*$") { return $Matches[1] }
  return $s
}
function Convert-SciToDecimal([string]$s) {
  $pat = '([+-]?(?:\d+(?:\.\d+)?|\.\d+))[eE]([+-]?\d+)'
  [regex]::Replace($s, $pat, {
    param($m)
    $base = [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
    $exp  = [int]$m.Groups[2].Value
    ($base * [math]::Pow(10,$exp)).ToString("0.############################",[Globalization.CultureInfo]::InvariantCulture)
  })
}
function Sanitize-Expr([string]$s) { Convert-SciToDecimal ($s -replace '\s+','') }
function SubN([string]$s, [int]$fps) { $s -replace '\bn\b', "(t*$fps)" }
function Unescape-Expr([string]$s) { $s -replace '\\,', ',' }
function Replace-ClipByScan([string]$s) {
  $s = $s -replace '\\,', ','
  $sb = [System.Text.StringBuilder]::new()
  $i = 0
  while ($i -lt $s.Length) {
    if ($i -le $s.Length-5 -and $s.Substring($i,5) -eq 'clip(') {
      $i += 5; $depth = 1; $inner = [System.Text.StringBuilder]::new()
      while ($i -lt $s.Length -and $depth -gt 0) {
        $ch = $s[$i]; if ($ch -eq '(') { $depth++ } elseif ($ch -eq ')') { $depth-- }
        if ($depth -gt 0) { $null = $inner.Append($ch) }; $i++
      }
      $inner = $inner.ToString()
      $lvl = 0; $first = -1; $second = -1
      for ($j=0; $j -lt $inner.Length; $j++) {
        $c = $inner[$j]
        if ($c -eq '(') { $lvl++ } elseif ($c -eq ')') { $lvl-- }
        elseif ($c -eq ',' -and $lvl -eq 0) { if ($first -lt 0) { $first=$j } elseif ($second -lt 0) { $second=$j; break } }
      }
      if ($first -lt 0 -or $second -lt 0) { throw "Could not parse clip(): $inner" }
      $v = $inner.Substring(0,$first); $a=$inner.Substring($first+1,$second-$first-1); $b=$inner.Substring($second+1)
      $null = $sb.Append("min(max($v,$a),$b)")
    } else { $null = $sb.Append($s[$i]); $i++ }
  }
  $sb.ToString()
}
function Normalize([string]$s, [int]$fps) {
  $s = Unescape-Expr $s
  $s = Sanitize-Expr $s
  $s = SubN $s $fps
  $s
}
function Get-CubicDeriv([string]$poly) {
  if ($poly -notmatch '^\s*\(\((?<a0>[-\deE\.]+)\)\+\((?<a1>[-\deE\.]+)\)\*n\+\((?<a2>[-\deE\.]+)\)\*n\*n\+\((?<a3>[-\deE\.]+)\)\*n\*n\*n\)\s*$') {
    return $null
  }
  $a1 = [double]$Matches.a1
  $a2 = [double]$Matches.a2
  $a3 = [double]$Matches.a3
  return "(($a1)+({0})*n+({1})*n*n)" -f (2 * $a2), (3 * $a3)
}
function Get-VarLine([string]$text, [string]$name) {
  $pat = '(?m)^\s*\$?' + [regex]::Escape($name) + '\s*=\s*(.+?)\s*$'
  $m = [regex]::Match($text, $pat)
  if ($m.Success) { $m.Groups[1].Value } else { $null }
}
function Load-ZoomVars([string]$varsPath) {
  if (-not (Test-Path $varsPath)) { return $null }
  $raw = Get-Content -Raw -Encoding UTF8 $varsPath
  $raw = $raw -replace "^\uFEFF","" -replace "\u200B|\u200E|\u200F",""
  $cxExpr = Unquote (Get-VarLine $raw 'cxExpr'); if (-not $cxExpr) { $cxExpr = Unquote (Get-VarLine $raw 'cx') }
  $cyExpr = Unquote (Get-VarLine $raw 'cyExpr'); if (-not $cyExpr) { $cyExpr = Unquote (Get-VarLine $raw 'cy') }
  $zExpr  = Unquote (Get-VarLine $raw 'zExpr' ); if (-not $zExpr ) { $zExpr  = Unquote (Get-VarLine $raw 'z' ) }
  if (-not $cxExpr -or -not $cyExpr -or -not $zExpr) { return $null }
  [pscustomobject]@{ cxExpr=$cxExpr; cyExpr=$cyExpr; zExpr=$zExpr }
}

# -- loop all atomic clips
Get-ChildItem $inDir -File -Filter "*.mp4" | ForEach-Object {
  $in  = $_.FullName

  # map:  "...\X__NAME__tA-tB.mp4"  ->  "...\X__NAME__tA-tB_zoom.ps1vars"
  $stem     = [IO.Path]::GetFileNameWithoutExtension($_.Name)
  $varsPath = Join-Path $varsDir ($stem + "_zoom.ps1vars")
  $vars = Load-ZoomVars $varsPath
  $name = $_.Name
  if ($null -eq $vars) {
    if ($DefaultIfMissing) {
      $vars = [pscustomobject]@{
        cxExpr = 'iw/2'
        cyExpr = 'ih/2'
        zExpr  = '1.35'
      }
      Write-Warning ("No vars for {0} - using defaults" -f $name)
      $Defaulted++
    }
    elseif ($SkipMissingVars) {
      Write-Warning "No vars for $name - skipping"
      $Skipped++
      return
    }
    else {
      Write-Error "Missing cxExpr/cyExpr/zExpr in $varsPath"
      $Failed++
      return
    }
  }

  Write-Host "Loaded:`n cxExpr=$($vars.cxExpr)`n cyExpr=$($vars.cyExpr)`n zExpr=$($vars.zExpr)" -ForegroundColor DarkCyan

  $fps = [int]$Fps

  $cxd = Get-CubicDeriv $($vars.cxExpr)
  $cyd = Get-CubicDeriv $($vars.cyExpr)

  $zN   = Normalize $($vars.zExpr)  $fps
  $cxN  = Normalize $($vars.cxExpr) $fps
  $cyN  = Normalize $($vars.cyExpr) $fps
  $cxpN = if ($cxd) { Normalize $cxd $fps } else { "0" }
  $cypN = if ($cyd) { Normalize $cyd $fps } else { "0" }

  $cxAgg = "($cxN)+($LeadFrames)*($cxpN)"
  $cyAgg = "($cyN)+($LeadFrames)*($cypN)"

  $speed = "sqrt(($cxpN)*($cxpN)+($cypN)*($cypN))"
  $zBase = "(min(max(($zN),$zMin),$zMax))"
  $zAgg  = "min(max((($zBase)+($zBias))+($kSpeed)*($speed),$zMin),$zMax)"

  $w = "floor(((ih*9/16)/($zAgg))/2)*2"
  $h = "floor((ih/($zAgg))/2)*2"
  $x = "clip((($cxAgg)-($w)/2),0,iw-($w))"
  $y = "clip((($cyAgg)-($h)/2),0,ih-($h))"

  $filter = "[0:v]crop=w='$w':h='$h':x='$x':y='$y',scale=w=-2:h=1080:flags=lanczos,setsar=1,format=yuv420p"
  $filter = Replace-ClipByScan $filter   # remove any lingering clip()

  # --- IO paths ---
  $inPath  = Join-Path $inDir  $_.Name
  $outPath = Join-Path $outDir ("COMPARE__" + $stem + ".mp4")

  if (-not (Test-Path $inPath)) { Write-Warning "Missing input: $inPath"; return }
  Write-Host "`n>> Processing $name  (fps=$fps)"
  Write-Host ">> filter: $filter"

  # run ffmpeg (inline filter avoids the script-file quoting mess)
  & $ffmpeg -hide_banner -y -nostdin `
    -i $inPath `
    -filter_complex $filter `
    -c:v libx264 -crf 20 -preset veryfast -movflags +faststart -an `
    $outPath
  if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed on $($_.Name)" }
}

Write-Host "`nSummary: Defaulted=$Defaulted  Skipped=$Skipped  Failed=$Failed"
Write-Host "All done."
