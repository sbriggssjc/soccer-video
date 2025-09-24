param(
  [Parameter(Mandatory=$true)] [string]$In,
  [Parameter(Mandatory=$true)] [string]$Vars,
  [Parameter(Mandatory=$true)] [string]$Out,
  [string]$VF = ".\out\autoframe_work\compare_autoframe.vf",
  [int]$FPS = 24  # freeze unless you want to auto-detect
)

Import-Module (Join-Path $PSScriptRoot 'tools/Autoframe.psm1') -Force

# 0) Load fits
Invoke-Expression (Get-Content -Raw -Encoding UTF8 $Vars)
if (-not $cxExpr -or -not $cyExpr -or -not $zExpr) {
  throw "Missing cx/cy/z in $Vars"
}

# 1) Pre-normalize base exprs (order is important!)
$cxN = Escape-Commas-In-Parens ( SubN ( Sanitize-Expr ( Expand-Clip $cxExpr ) ) $FPS )
$cyN = Escape-Commas-In-Parens ( SubN ( Sanitize-Expr ( Expand-Clip $cyExpr ) ) $FPS )
$zN  = Escape-Commas-In-Parens ( SubN ( Sanitize-Expr ( Expand-Clip $zExpr  ) ) $FPS )

# 2) Build derived exprs (use ih)
$cwExpr = "((ih*9/16)/($zN))"
$chExpr = "(ih/($zN))"
$wExpr  = "floor(($cwExpr)/2)*2"
$hExpr  = "floor(($chExpr)/2)*2"
$xCore  = "($cxN)-($wExpr)/2"
$yCore  = "($cyN)-($hExpr)/2"

# 3) Expand nested $ once
$expand = { param($s) $ExecutionContext.InvokeCommand.ExpandString($s) }
$wE = & $expand $wExpr
$hE = & $expand $hExpr
$xE = & $expand $xCore
$yE = & $expand $yCore

# 4) Final safety: escape commas inside parens
$wE = Escape-Commas-In-Parens $wE
$hE = Escape-Commas-In-Parens $hE
$xE = Escape-Commas-In-Parens $xE
$yE = Escape-Commas-In-Parens $yE

# 5) Write filtergraph (positional crop + ${} so PS doesn’t treat : as drive)
$vfText = @"
[0:v]split=2[left][right];
[right]crop=${wE}:${hE}:${xE}:${yE},scale=-2:1080:flags=lanczos,setsar=1[right_portrait];
[left][right_portrait]hstack=inputs=2,format=yuv420p
"@
[IO.File]::WriteAllText($VF, $vfText, (New-Object Text.UTF8Encoding($false)))

# 6) Log for reproducibility
"`n--- DEBUG ---" | Write-Host
"FPS=$FPS" | Write-Host
"W=$wE" | Write-Host
"H=$hE" | Write-Host
"X=$xE" | Write-Host
"Y=$yE" | Write-Host
"`n--- VF ($VF) ---" | Write-Host
(Get-Content $VF) | Write-Host

# 7) Render (keep using -filter_complex_script; deprecation msg is cosmetic)
ffmpeg -hide_banner -y -nostdin `
  -i $In `
  -filter_complex_script $VF `
  -c:v libx264 -crf 20 -preset veryfast -movflags +faststart -an `
  $Out
