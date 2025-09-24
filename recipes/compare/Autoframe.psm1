$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$helperModulePath = Join-Path (Join-Path $repoRoot 'tools') 'Autoframe.psm1'
if (-not (Test-Path -LiteralPath $helperModulePath)) {
  throw "Autoframe helper module not found: $helperModulePath"
}
Import-Module $helperModulePath -Force

$script:FFmpegVersionLogged = $false
$script:FFmpegSelfTested = $false

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

  if (-not $script:FFmpegVersionLogged) {
    $versionOutput = & ffmpeg -hide_banner -version 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $versionOutput) {
      throw "Unable to query ffmpeg version"
    }
    $versionLine = ($versionOutput | Select-Object -First 1).Trim()
    if (-not $versionLine) { $versionLine = 'unknown' }
    Write-Host "[autoframe] ffmpeg version: $versionLine (watch for behavior changes after upgrades)"
    $script:FFmpegVersionLogged = $true
  }

  if (-not $script:FFmpegSelfTested) {
    Invoke-FFmpegSelfTest
    $script:FFmpegSelfTested = $true
  }
}

function Invoke-FFmpegSelfTest {
  $testArgs = @(
    '-hide_banner', '-loglevel', 'error', '-nostdin',
    '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=1',
    '-f', 'null', '-'
  )
  & ffmpeg @testArgs | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg self-test failed with exit code $LASTEXITCODE"
  }
  Write-Host '[autoframe] ffmpeg self-test: ok (1s synthetic input)'
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
    [string]$Expression,
    [Nullable[double]]$Fps
  )

  $expr = Convert-SciToDecimal -Expression $Expression
  $expr = ($expr -replace '\s+', '')
  if ($PSBoundParameters.ContainsKey('Fps') -and $null -ne $Fps) {
    $expr = SubN -Expression $expr -Fps $Fps
  }
  return $expr
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

  foreach ($name in 'wE','hE','xE','yE') {
    $val = Get-Variable -Name $name -ValueOnly
    $val = Sanitize-Expr -Expression $val -Fps $fps
    Set-Variable -Name $name -Value $val
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

  $clipX = Expand-Clip -Expression $E.X -Min '0' -Max "iw-($($E.W))"
  $clipY = Expand-Clip -Expression $E.Y -Min '0' -Max "ih-($($E.H))"
  $vfLines = @(
    '[0:v]split=2[left][right];',
    "[right]crop=$($E.W):$($E.H):$clipX:$clipY,scale=-2:1080:flags=lanczos,setsar=1[right_portrait];",
    '[left][right_portrait]hstack=inputs=2,format=yuv420p'
  )
  $vf = [string]::Join([Environment]::NewLine, $vfLines)
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
