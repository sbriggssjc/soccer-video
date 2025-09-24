function Get-Fps ($path) {
  # safe: returns integer fps
  $ff = ffprobe -v error -select_streams v:0 `
        -show_entries stream=r_frame_rate -of default=nk=1:nw=1 "$path"
  if ($LASTEXITCODE -ne 0) { throw "ffprobe failed" }
  $num,$den = $ff -split '/'
  return [int][math]::Round(($num / [double]$den))
}

function Load-ExprVars ($varsPath) {
  Invoke-Expression (Get-Content -Raw -Encoding UTF8 $varsPath)
  if (-not $cxExpr -or -not $cyExpr -or -not $zExpr) {
    throw "Missing cx/cy/z in $varsPath"
  }
  [pscustomobject]@{ cx=$cxExpr; cy=$cyExpr; z=$zExpr }
}

function Sanitize-Expr([string]$s) {
  # 1) 1.23e-05 → (1.23*pow(10,-5))   2) strip spaces
  $s = $s -replace '([0-9]+\.[0-9]+|[0-9]+)[eE]\+?(-?[0-9]+)', '($1*pow(10,$2))'
  $s = ($s -replace '\s+', '')
  return $s
}

function Use-FFExprVars($inPath, $vars) {
  # IMPORTANT: use 'ih' (a.k.a input height) not 'in_h' or 'h'
  $cw = "((ih*9/16)/$($vars.z))"
  $ch = "(ih/$($vars.z))"
  $wE = "floor(($cw)/2)*2"
  $hE = "floor(($ch)/2)*2"
  $xE = "($($vars.cx))-($wE)/2"
  $yE = "($($vars.cy))-($hE)/2"

  $fps = Get-Fps $inPath
  $subN = { param($s,$fps) ($s -replace '\bn\b',"(t*$fps)") }

  # sanitize + n→t*fps
  foreach ($name in 'wE','hE','xE','yE') {
    $val = Get-Variable $name -ValueOnly
    $val = Sanitize-Expr (& $subN $val $fps)
    Set-Variable $name $val
  }

  # return a clean record
  [pscustomobject]@{
    W = $wE; H = $hE; X = $xE; Y = $yE; FPS = $fps
  }
}

function Write-CompareVF($vfPath, $E) {
  # NO quotes around expressions, NO backslashes, NO comma-escaping.
  $vf = @"
[0:v]split=2[left][right];
[right]crop=$($E.W):$($E.H):$($E.X):$($E.Y),scale=-2:1080:flags=lanczos,setsar=1[right_portrait];
[left][right_portrait]hstack=inputs=2,format=yuv420p
"@
  [IO.File]::WriteAllText($vfPath, $vf, (New-Object Text.UTF8Encoding($false)))
  return $vf
}

function Render-Compare($in,$vfPath,$out) {
  ffmpeg -hide_banner -y -nostdin -i "$in" -filter_complex_script "$vfPath" `
    -c:v libx264 -crf 20 -preset veryfast -movflags +faststart -an "$out"
  if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed" }
}
