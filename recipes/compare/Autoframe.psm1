function Assert-CommandAvailable {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name
  )

  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required executable '$Name' was not found in PATH."
  }
}

function Ensure-AutoframeTools {
  foreach ($tool in 'ffprobe', 'ffmpeg') {
    Assert-CommandAvailable -Name $tool
  }
}

function Get-Fps {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  # safe: returns integer fps
  Ensure-AutoframeTools
  $ff = ffprobe -v error -select_streams v:0 `
        -show_entries stream=r_frame_rate -of default=nk=1:nw=1 "$Path"
  if ($LASTEXITCODE -ne 0) { throw "ffprobe failed" }
  $num,$den = $ff -split '/'
  return [int][math]::Round(($num / [double]$den))
}

function Load-ExprVars {
  param(
    [Parameter(Mandatory = $true)]
    [string]$VarsPath
  )

  if (-not (Test-Path -LiteralPath $VarsPath)) {
    throw "Vars file not found: $VarsPath"
  }

  Invoke-Expression (Get-Content -Raw -Encoding UTF8 $VarsPath)
  if (-not $cxExpr -or -not $cyExpr -or -not $zExpr) {
    throw "Missing cx/cy/z in $VarsPath"
  }
  [pscustomobject]@{ cx=$cxExpr; cy=$cyExpr; z=$zExpr }
}

function Sanitize-Expr {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Expression
  )

  # 1) 1.23e-05 → (1.23*pow(10,-5))   2) strip spaces
  $sanitized = $Expression -replace '([0-9]+\.[0-9]+|[0-9]+)[eE]\+?(-?[0-9]+)', '($1*pow(10,$2))'
  $sanitized = ($sanitized -replace '\s+', '')
  return $sanitized
}

function Use-FFExprVars {
  param(
    [Parameter(Mandatory = $true)]
    [string]$InPath,
    [Parameter(Mandatory = $true)]
    $Vars
  )

  # IMPORTANT: use 'ih' (a.k.a input height) not 'in_h' or 'h'
  $cw = "((ih*9/16)/$($Vars.z))"
  $ch = "(ih/$($Vars.z))"
  $wE = "floor(($cw)/2)*2"
  $hE = "floor(($ch)/2)*2"
  $xE = "($($Vars.cx))-($wE)/2"
  $yE = "($($Vars.cy))-($hE)/2"

  $fps = Get-Fps -Path $InPath
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

function Write-CompareVF {
  param(
    [Parameter(Mandatory = $true)]
    [string]$VfPath,
    [Parameter(Mandatory = $true)]
    $E
  )

  # NO quotes around expressions, NO backslashes, NO comma-escaping.
  $vf = @"
[0:v]split=2[left][right];
[right]crop=$($E.W):$($E.H):$($E.X):$($E.Y),scale=-2:1080:flags=lanczos,setsar=1[right_portrait];
[left][right_portrait]hstack=inputs=2,format=yuv420p
"@
  $encoding = New-Object Text.UTF8Encoding($false)
  [IO.File]::WriteAllText($VfPath, $vf, $encoding)
  return $vf
}

function Render-Compare {
  param(
    [Parameter(Mandatory = $true)]
    [string]$In,
    [Parameter(Mandatory = $true)]
    [string]$VfPath,
    [Parameter(Mandatory = $true)]
    [string]$Out
  )

  Ensure-AutoframeTools
  ffmpeg -hide_banner -y -nostdin -i "$In" -filter_complex_script "$VfPath" `
    -c:v libx264 -crf 20 -preset veryfast -movflags +faststart -an "$Out"
  if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed" }
}

Export-ModuleMember -Function \
  Assert-CommandAvailable, \
  Ensure-AutoframeTools, \
  Get-Fps, \
  Load-ExprVars, \
  Sanitize-Expr, \
  Use-FFExprVars, \
  Write-CompareVF, \
  Render-Compare
