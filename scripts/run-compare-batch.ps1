# ==== Run-Compare-Batch.ps1 ====
# Requires: ffmpeg/ffprobe in PATH, your per-clip *.ps1vars files

# --- Stable roots/dirs ---
$ErrorActionPreference = 'Stop'
$Root    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
$ffmpeg  = 'ffmpeg'
$inDir   = (Resolve-Path (Join-Path $Root 'out\atomic_clips')).Path
$varsDir = (Resolve-Path (Join-Path $Root 'out\autoframe_work')).Path
$outDir  = Join-Path $Root 'out\reels\tiktok'
New-Item -Force -ItemType Directory $outDir | Out-Null

# --- Helpers (replace your existing ones with these) ---
function Convert-SciToDecimal([string]$s) {
  $pat = '([+-]?(?:\d+(?:\.\d+)?|\.\d+))[eE]([+-]?\d+)'
  [regex]::Replace($s, $pat, {
    param($m)
    $base = [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
    $exp  = [int]$m.Groups[2].Value
    ($base * [math]::Pow(10,$exp)).ToString("0.############################",[Globalization.CultureInfo]::InvariantCulture)
  })
}
function Sanitize-Expr([string]$s) {
  $s = $s -replace '\s+',''
  Convert-SciToDecimal $s
}
function SubN([string]$s, [int]$fps) { $s -replace '\bn\b', "(t*$fps)" }

# Unescape any historical '\\,' so 'clip(v\\,a\\,b)' becomes 'clip(v,a,b)'
function Unescape-Expr([string]$s) { $s -replace '\\,', ',' }

# Split a string on commas at depth 0 (ignoring nested parens)
function Split-TopLevel([string]$text) {
  $parts   = @()
  $current = [System.Text.StringBuilder]::new()
  $depth   = 0

  foreach ($ch in $text.ToCharArray()) {
    switch ($ch) {
      '(' { $depth++; [void]$current.Append($ch); continue }
      ')' { if ($depth -gt 0) { $depth-- }; [void]$current.Append($ch); continue }
      ',' {
        if ($depth -eq 0) {
          $parts += $current.ToString()
          [void]$current.Clear()
          continue
        }
      }
    }
    [void]$current.Append($ch)
  }

  $parts += $current.ToString()
  return $parts
}

function Replace-ClipByScan([string]$text) {
  if ([string]::IsNullOrWhiteSpace($text)) { return $text }

  $builder = [System.Text.StringBuilder]::new()
  $length  = $text.Length
  $index   = 0

  while ($index -lt $length) {
    if ($index -le $length - 4) {
      $segment = $text.Substring($index, 4)
      if ($segment.Equals('clip', [System.StringComparison]::OrdinalIgnoreCase)) {
        $prevIndex = $index - 1
        $prevValid = ($prevIndex -lt 0) -or -not ([char]::IsLetterOrDigit($text[$prevIndex]) -or $text[$prevIndex] -eq '_')
        $probe = $index + 4
        while ($probe -lt $length -and [char]::IsWhiteSpace($text[$probe])) { $probe++ }
        if ($prevValid -and $probe -lt $length -and $text[$probe] -eq '(') {
          $depth = 1
          $pos   = $probe + 1
          while ($pos -lt $length -and $depth -gt 0) {
            $ch = $text[$pos]
            if ($ch -eq '(') { $depth++ }
            elseif ($ch -eq ')') { $depth-- }
            $pos++
          }

          if ($depth -eq 0) {
            $argsText = if ($pos -gt $probe + 1) { $text.Substring($probe + 1, $pos - $probe - 2) } else { '' }
            $args = Split-TopLevel $argsText
            if ($args.Count -eq 3) {
              $valueExpr = Replace-ClipByScan ($args[0].Trim())
              $minExpr   = Replace-ClipByScan ($args[1].Trim())
              $maxExpr   = Replace-ClipByScan ($args[2].Trim())
              $replacement = "min(max(($valueExpr),($minExpr)),($maxExpr))"
              [void]$builder.Append($replacement)
              $index = $pos
              continue
            }
          }
        }
      }
    }

    [void]$builder.Append($text[$index])
    $index++
  }

  return $builder.ToString()
}

# clip(v,a,b)  ->  min(max(v,a), b)   (supports nesting)
function Expand-Clip([string]$s) {
  # Safe to assume commas are unescaped now
  $pat = 'clip\((?<v>(?>[^()]+|\((?<o>)|\)(?<-o>))*(?(o)(?!))),(?<a>(?>[^()]+|\((?<o2>)|\)(?<-o2>))*(?(o2)(?!))),(?<b>(?>[^()]+|\((?<o3>)|\)(?<-o3>))*(?(o3)(?!)))\)'
  while ($s -match $pat) {
    $s = [regex]::Replace($s, $pat, {
      param($m)
      "min(max($($m.Groups['v'].Value),$($m.Groups['a'].Value)),$($m.Groups['b'].Value))"
    })
  }
  $s
}

# One-stop normalization used for cx, cy, z
function Normalize([string]$s, [int]$fps) {
  $s = Unescape-Expr $s
  $s = Expand-Clip   $s
  $s = Sanitize-Expr $s
  $s = SubN          $s $fps
  return $s
}

# -- loop all atomic clips
Get-ChildItem $inDir -File -Filter "*.mp4" | ForEach-Object {
  $in  = $_.FullName

  # map:  "...\X__NAME__tA-tB.mp4"  ->  "...\X__NAME__tA-tB_zoom.ps1vars"
  $stem     = [IO.Path]::GetFileNameWithoutExtension($_.Name)
  $varsPath = Join-Path $varsDir ($stem + "_zoom.ps1vars")
  # --- Load fit vars robustly ---
  if (-not (Test-Path $varsPath)) {
    Write-Warning "No vars for $($_.Name) -> $varsPath"
    return
  }

  # --- Robust loader for cxExpr/cyExpr/zExpr from $varsPath
  #     Accepts $cxExpr=..., cxExpr=..., $cx=..., cx=...
  Remove-Variable cxExpr,cyExpr,zExpr,cx,cy,z -ErrorAction SilentlyContinue

  $raw = Get-Content -Raw -Encoding UTF8 $varsPath
  # strip BOM/zero-width junk just in case
  $raw = $raw -replace "^\uFEFF","" -replace "\u200B|\u200E|\u200F",""

  function Get-VarLine([string]$text, [string]$name) {
    # match "$name=..." OR "name=..." capturing the full RHS to end of line
    $m = [regex]::Match($text, "(?m)^\s*\$?$name\s*=\s*(.+?)\s*$")
    if ($m.Success) { $m.Groups[1].Value } else { $null }
  }

  $cxExpr = Get-VarLine $raw "cxExpr"
  $cyExpr = Get-VarLine $raw "cyExpr"
  $zExpr  = Get-VarLine $raw "zExpr"

  # fallbacks to cx/cy/z
  if (-not $cxExpr) { $cxExpr = Get-VarLine $raw "cx" }
  if (-not $cyExpr) { $cyExpr = Get-VarLine $raw "cy" }
  if (-not $zExpr)  { $zExpr  = Get-VarLine $raw "z"  }

  # verify
  $missing = @()
  foreach ($n in "cxExpr","cyExpr","zExpr") { if (-not (Get-Variable $n -ValueOnly -ErrorAction SilentlyContinue)) { $missing += $n } }
  if ($missing.Count) { throw "Missing $($missing -join '/')" + " in $varsPath" }

  Write-Host "Loaded:`n cxExpr=$cxExpr`n cyExpr=$cyExpr`n zExpr=$zExpr" -ForegroundColor DarkCyan

  $fps = 24  # lock if your ingest is always 24

  # normalize first (this kills clip(), sci notation, and n->t*fps)
  $zN  = Normalize $zExpr  $fps
  $cxN = Normalize $cxExpr $fps
  $cyN = Normalize $cyExpr $fps

  # build expressions (portrait 9:16 viewport width derived from ih)
  $w   = "floor(((ih*9/16)/($zN))/2)*2"
  $h   = "floor((ih/($zN))/2)*2"
  $x   = "($cxN)-($w)/2"
  $y   = "($cyN)-($h)/2"

  # assemble the final filter string (named args; each quoted)
  $filter = "[0:v]crop=w='$w':h='$h':x='$x':y='$y',scale=w=-2:h=1080:flags=lanczos,setsar=1,format=yuv420p"

  # --- IO paths ---
  $inPath  = Join-Path $inDir  $_.Name
  $outPath = Join-Path $outDir ("COMPARE__" + $stem + ".mp4")

  if (-not (Test-Path $inPath)) { Write-Warning "Missing input: $inPath"; return }
  Write-Host "`n>> Processing $($_.Name)  (fps=$fps)"
  Write-Host ">> filter: $filter"

  # run ffmpeg (inline filter avoids the script-file quoting mess)
  $filter = Replace-ClipByScan $filter
  if ($filter -match 'clip\(') { throw 'clip() still present after expansion' }
  & $ffmpeg -hide_banner -y -nostdin `
    -i $inPath `
    -filter_complex $filter `
    -c:v libx264 -crf 20 -preset veryfast -movflags +faststart -an `
    $outPath
  if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed on $($_.Name)" }
}

Write-Host "`nAll done."
