function Convert-ToFrameExpr {
    param([Parameter(Mandatory)][string]$poly, [Parameter(Mandatory)][int]$fps)
    # DO NOT swap n -> (t*fps). Keep 'n' so crop eval doesn't see 't' at init.
    return $poly
}

function Remove-ClipCalls {
    param([Parameter(Mandatory)][string]$expr)
    $res = $expr
    while ($res -match 'clip\(\s*([^,]+?)\s*,\s*[-+0-9.eE]+\s*,\s*[-+0-9.eE]+\s*\)') {
        $res = $res -replace 'clip\(\s*([^,]+?)\s*,\s*[-+0-9.eE]+\s*,\s*[-+0-9.eE]+\s*\)', '$1'
    }
    return $res
}

function Get-SafeZoomExpr {
    param([Parameter(Mandatory)][string]$zExpr,
          [Parameter(Mandatory)][int]$fps,
          [double]$minZ = 1.08,
          [double]$maxZ = 2.40)
    # Leave n-based poly alone; just strip clip() and clamp via if()
    $inner = Remove-ClipCalls -expr (Convert-ToFrameExpr -poly $zExpr -fps $fps)
    return "if(lte(($inner),$minZ),$minZ, if(gte(($inner),$maxZ),$maxZ, ($inner)))"
}

function Get-EvenWindowExprs {
    param([Parameter(Mandatory)][string]$zz)
    $w = "max(16, floor(((ih*9/16)/($zz))/2)*2)"
    $h = "max(16, floor((ih/($zz))/2)*2)"
    return @{
        w = $w
        h = $h
    }
}

function Build-GoalCorridorChain {
    param(
        [Parameter(Mandatory)][string]$cxExpr,
        [Parameter(Mandatory)][string]$cyExpr,
        [Parameter(Mandatory)][string]$zExpr,
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
    # Use n-based polynomials
    $cx = Convert-ToFrameExpr -poly $cxExpr -fps $fps
    $cy = Convert-ToFrameExpr -poly $cyExpr -fps $fps
    $zz = Get-SafeZoomExpr   -zExpr $zExpr -fps $fps

    # Even window from zoom (safe for crop eval)
    $win = Get-EvenWindowExprs -zz $zz
    $w = $win.w
    $h = $win.h

    # We'll still need "seconds" for ramps, but express seconds as n/fps (no bare 't')
    $tExpr  = "(n/$fps)"                   # seconds from n
    $rampR  = "min( ($tExpr/$startRampSec), 1 )"

    # Corridor clamp X (soft min/max based on window size)
    $midX    = "(($goalLeft+$goalRight)/2)"
    $minXc   = "(($goalLeft+$padPx)+($w)/2)"
    $maxXc   = "(($goalRight-$padPx)-($w)/2)"
    $cxSoft  = "min(max(($cx), $minXc), $maxXc)"
    $cxRamp  = "( ($midX)*(1-($rampR)) + ($cxSoft)*($rampR) )"

    # Celebration lock: between(t, tGoal, tGoal+cele) but t := n/fps
    $inCele  = "between($tExpr,$tGoalSec,($tGoalSec+$celeSec))"
    $cxFinal = "if($inCele, $midX, $cxRamp)"
    $zzFinal = "if($inCele, min($zz,$celeTight), $zz)"   # window uses $zz (smooth), not $zzFinal

    # Y band clamp (uses window height)
    $yMinC   = "($yMin+($h)/2)"
    $yMaxC   = "($yMax-($h)/2)"
    $cyClamp = "min(max(($cy), $yMinC), $yMaxC)"

    # center->top-left and bounds
    $x = "min(max((($cxFinal)-($w)/2),0), iw-($w))"
    $y = "min(max((($cyClamp)-($h)/2),0), ih-($h))"

    $chainCropScale = "crop=w='$w':h='$h':x='$x':y='$y',scale=w=-2:h=1080:flags=lanczos,setsar=1,format=yuv420p"
    $chainScaleCrop = "scale=w=-2:h=1080:flags=lanczos,setsar=1,crop=w='$w':h='$h':x='$x':y='$y',format=yuv420p"
    if ($ScaleFirst) { return $chainScaleCrop } else { return $chainCropScale }
}

function New-VFChain {
    param(
        [Parameter(Mandatory)][string]$VarsPath,
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
    if (-not (Test-Path $VarsPath)) { throw "Vars file not found: $VarsPath" }
    . $VarsPath
    if (-not ($cxExpr -and $cyExpr -and $zExpr)) { throw "Expected cxExpr, cyExpr, zExpr in $VarsPath" }
    return (Build-GoalCorridorChain -cxExpr $cxExpr -cyExpr $cyExpr -zExpr $zExpr -fps $fps `
        -goalLeft $goalLeft -goalRight $goalRight -padPx $padPx -startRampSec $startRampSec `
        -tGoalSec $tGoalSec -celeSec $celeSec -celeTight $celeTight -yMin $yMin -yMax $yMax `
        -ScaleFirst:$ScaleFirst)
}

Export-ModuleMember -Function Convert-ToFrameExpr,Remove-ClipCalls,Get-SafeZoomExpr,Get-EvenWindowExprs,Build-GoalCorridorChain,New-VFChain
