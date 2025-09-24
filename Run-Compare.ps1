param(
  [string]$In    = ".\out\atomic_clips\001__GOAL__t155.50-t166.40.mp4",
  [string]$Vars  = ".\out\autoframe_work\001__GOAL__t155.50-t166.40_zoom.ps1vars",
  [string]$Out   = ".\out\reels\tiktok\COMPARE__001__GOAL__t155.50-t166.40.mp4",
  [string]$VF    = ".\out\autoframe_work\compare_autoframe.vf"
)

Import-Module "$PSScriptRoot\tools\Autoframe.psm1" -Force

$exprs = Load-ExprVars $Vars
$E     = Use-FFExprVars -inPath $In -vars $exprs

"--- DEBUG (computed) ---"
"FPS=$($E.FPS)"
"W=$($E.W)"
"H=$($E.H)"
"X=$($E.X)"
"Y=$($E.Y)"

$vfText = Write-CompareVF $VF $E
"--- VF file ---"
$vfText

Render-Compare $In $VF $Out
"Done: $Out"
