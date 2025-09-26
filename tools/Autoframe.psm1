function Convert-ToFrameExpr {
    param(
        [string]$poly,
        [int]$fps
    )

    if ([string]::IsNullOrWhiteSpace($poly)) {
        return $poly
    }

    $pattern = '(?<![A-Za-z0-9_])n(?![A-Za-z0-9_])'
    return [System.Text.RegularExpressions.Regex]::Replace($poly, $pattern, "(t*$fps)")
}

function Remove-ClipCalls {
    param(
        [string]$expr
    )

    if ([string]::IsNullOrWhiteSpace($expr)) {
        return $expr
    }

    $pattern = 'clip\(([^,]+),\s*[-+0-9\.]+\s*,\s*[-+0-9\.]+\)'
    $result = $expr
    while ([System.Text.RegularExpressions.Regex]::IsMatch($result, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
        $result = [System.Text.RegularExpressions.Regex]::Replace($result, $pattern, '$1', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    }

    return $result
}

function Get-SafeZoomExpr {
    param(
        [string]$zExpr,
        [int]$fps,
        [double]$minZ = 1.08,
        [double]$maxZ = 2.40
    )

    $inner = Convert-ToFrameExpr -poly $zExpr -fps $fps
    $inner = Remove-ClipCalls -expr $inner

    if ([string]::IsNullOrWhiteSpace($inner)) {
        $inner = "1.0"
    }

    return "if(lte(($inner),$minZ),$minZ, if(gte(($inner),$maxZ),$maxZ, ($inner)))"
}

function Get-EvenWindowExprs {
    param(
        [string]$zz
    )

    return @{
        w = "max(16, floor(((ih*9/16)/($zz))/2)*2)",
        h = "max(16, floor((ih/($zz))/2)*2)"
    }
}

function Build-GoalCorridorChain {
    param(
        [string]$cxExpr,
        [string]$cyExpr,
        [string]$zExpr,
        [int]$fps = 24,
        [double]$goalLeft,
        [double]$goalRight,
        [double]$padPx = 40,
        [double]$startRampSec = 0.0,
        [double]$tGoalSec,
        [double]$celeSec,
        [double]$celeTight,
        [double]$yMin,
        [double]$yMax,
        [switch]$ScaleFirst
    )

    $cxBase = Convert-ToFrameExpr -poly $cxExpr -fps $fps
    $cyBase = Convert-ToFrameExpr -poly $cyExpr -fps $fps
    $zzBase = Get-SafeZoomExpr -zExpr $zExpr -fps $fps
    $win = Get-EvenWindowExprs -zz $zzBase
    $w = $win.w
    $h = $win.h

    $cxCorr = $cxBase
    if ($PSBoundParameters.ContainsKey('goalLeft') -and $PSBoundParameters.ContainsKey('goalRight')) {
        $pad = if ($PSBoundParameters.ContainsKey('padPx')) { $padPx } else { 0 }
        $minXc = "(($goalLeft+$pad)+($w)/2)"
        $maxXc = "(($goalRight-$pad)-($w)/2)"
        $cxCorr = "min(max(($cxCorr), $minXc), $maxXc)"
        if ($startRampSec -gt 0) {
            $ramp = "min((t/$startRampSec),1)"
            $cxCorr = "(($cxBase)*(1-($ramp)) + ($cxCorr)*($ramp))"
        }
    }

    $cyClamp = $cyBase
    if ($PSBoundParameters.ContainsKey('yMin') -and $PSBoundParameters.ContainsKey('yMax')) {
        $yMinC = "($yMin+($h)/2)"
        $yMaxC = "($yMax-($h)/2)"
        $cyClamp = "min(max(($cyClamp), $yMinC), $yMaxC)"
    }

    $cxFinal = $cxCorr
    $zzFinal = $zzBase
    if (
        $PSBoundParameters.ContainsKey('tGoalSec') -and
        $PSBoundParameters.ContainsKey('celeSec') -and
        $PSBoundParameters.ContainsKey('celeTight') -and
        $PSBoundParameters.ContainsKey('goalLeft') -and
        $PSBoundParameters.ContainsKey('goalRight')
    ) {
        $midX = "(($goalLeft+$goalRight)/2)"
        $inCele = "between(t,$tGoalSec,($tGoalSec+$celeSec))"
        $cxFinal = "if($inCele, $midX, $cxFinal)"
        $zzFinal = "if($inCele, min($zzFinal,$celeTight), $zzFinal)"
    }

    $x = "min(max((($cxFinal)-($w)/2),0), iw-($w))"
    $y = "min(max((($cyClamp)-($h)/2),0), ih-($h))"

    if ($ScaleFirst) {
        return "scale=w=-2:h=1080:flags=lanczos,setsar=1,crop=w='$w':h='$h':x='$x':y='$y',format=yuv420p"
    }

    return "crop=w='$w':h='$h':x='$x':y='$y',scale=w=-2:h=1080:flags=lanczos,setsar=1,format=yuv420p"
}

function New-VFChain {
    param(
        [Parameter(Mandatory=$true)][Alias('Vars')][string]$VarsPath,
        [int]$fps = 24,
        [double]$goalLeft,
        [double]$goalRight,
        [double]$padPx,
        [double]$startRampSec,
        [double]$tGoalSec,
        [double]$celeSec,
        [double]$celeTight,
        [double]$yMin,
        [double]$yMax,
        [switch]$ScaleFirst
    )

    if (-not (Test-Path $VarsPath)) {
        throw "Vars file not found: $VarsPath"
    }

    . $VarsPath

    if (-not (Test-Path variable:cxExpr)) {
        throw "Expected `$cxExpr to be defined in vars file."
    }

    if (-not (Test-Path variable:cyExpr)) {
        throw "Expected `$cyExpr to be defined in vars file."
    }

    if (-not (Test-Path variable:zExpr)) {
        throw "Expected `$zExpr to be defined in vars file."
    }

    $callArgs = @{
        cxExpr = $cxExpr
        cyExpr = $cyExpr
        zExpr = $zExpr
        fps = $fps
    }

    if ($PSBoundParameters.ContainsKey('goalLeft')) { $callArgs.goalLeft = $goalLeft }
    if ($PSBoundParameters.ContainsKey('goalRight')) { $callArgs.goalRight = $goalRight }
    if ($PSBoundParameters.ContainsKey('padPx')) { $callArgs.padPx = $padPx }
    if ($PSBoundParameters.ContainsKey('startRampSec')) { $callArgs.startRampSec = $startRampSec }
    if ($PSBoundParameters.ContainsKey('tGoalSec')) { $callArgs.tGoalSec = $tGoalSec }
    if ($PSBoundParameters.ContainsKey('celeSec')) { $callArgs.celeSec = $celeSec }
    if ($PSBoundParameters.ContainsKey('celeTight')) { $callArgs.celeTight = $celeTight }
    if ($PSBoundParameters.ContainsKey('yMin')) { $callArgs.yMin = $yMin }
    if ($PSBoundParameters.ContainsKey('yMax')) { $callArgs.yMax = $yMax }
    if ($ScaleFirst.IsPresent) { $callArgs.ScaleFirst = $true }

    return Build-GoalCorridorChain @callArgs
}

Export-ModuleMember -Function Convert-ToFrameExpr,Remove-ClipCalls,Get-SafeZoomExpr,Get-EvenWindowExprs,Build-GoalCorridorChain,New-VFChain
