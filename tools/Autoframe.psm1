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
        [int]$goalLeft,
        [int]$goalRight,
        [int]$padPx = 40,
        [double]$startRampSec = 3.0,
        [double]$tGoalSec = 10.0,
        [double]$celeSec = 2.0,
        [double]$celeTight = 1.80,
        [double]$yMin = 384,
        [double]$yMax = 768,
        [switch]$ScaleFirst
    )

    $cx = Convert-ToFrameExpr -poly $cxExpr -fps $fps
    $cy = Convert-ToFrameExpr -poly $cyExpr -fps $fps
    $zz = Get-SafeZoomExpr -zExpr $zExpr -fps $fps
    $win = Get-EvenWindowExprs -zz $zz
    $w = $win.w
    $h = $win.h

    $midX  = "(($goalLeft+$goalRight)/2)"
    $minXc = "(($goalLeft+$padPx)+($w)/2)"
    $maxXc = "(($goalRight-$padPx)-($w)/2)"
    $cxSoft = "min(max(($cx), $minXc), $maxXc)"
    $rampR  = "min( (t/$startRampSec), 1 )"
    $cxRamp = "( ($midX)*(1-($rampR)) + ($cxSoft)*($rampR) )"
    $inCele = "between(t,$tGoalSec,($tGoalSec+$celeSec))"
    $cxFinal = "if($inCele, $midX, $cxRamp)"
    $zzFinal = "if($inCele, min($zz,$celeTight), $zz)"
    $yMinC   = "($yMin+($h)/2)"
    $yMaxC   = "($yMax-($h)/2)"
    $cyClamp = "min(max(($cy), $yMinC), $yMaxC)"
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
        [int]$goalLeft,
        [int]$goalRight,
        [int]$padPx = 40,
        [double]$startRampSec = 3.0,
        [double]$tGoalSec = 10.0,
        [double]$celeSec = 2.0,
        [double]$celeTight = 1.80,
        [double]$yMin = 360,
        [double]$yMax = 780,
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

    return Build-GoalCorridorChain -cxExpr $cxExpr -cyExpr $cyExpr -zExpr $zExpr -fps $fps -goalLeft $goalLeft -goalRight $goalRight -padPx $padPx -startRampSec $startRampSec -tGoalSec $tGoalSec -celeSec $celeSec -celeTight $celeTight -yMin $yMin -yMax $yMax -ScaleFirst:$ScaleFirst
}

Export-ModuleMember -Function Convert-ToFrameExpr,Remove-ClipCalls,Get-SafeZoomExpr,Get-EvenWindowExprs,Build-GoalCorridorChain,New-VFChain
