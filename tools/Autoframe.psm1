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
        [double]$PreWideSec = 2.0,
        [int]$EarlyLeft = 720,
        [int]$EarlyRight = 1200,
        [int]$EarlyPadPx = 24,
        [double]$ShotLead = 0.6,
        [double]$ShotTrail = 0.9,
        [double]$ShotZoomCap = 1.15,
        [double]$CeleBlend = 0.6,
        [double]$CeleZoomCap = 1.22,
        [switch]$ScaleFirst
    )
    # Use n-based polynomials
    $cx = Convert-ToFrameExpr -poly $cxExpr -fps $fps
    $cy = Convert-ToFrameExpr -poly $cyExpr -fps $fps
    $zz = Get-SafeZoomExpr   -zExpr $zExpr -fps $fps

    # time helpers
    $t      = "(n/$fps)"
    $inPre  = "between($t,0,$PreWideSec)"
    $inShot = "between($t,($tGoalSec-$ShotLead),($tGoalSec+$ShotTrail))"
    $inCele = "between($t,$tGoalSec,($tGoalSec+$celeSec))"

    # goal & window centers
    $midX = "(($goalLeft+$goalRight)/2)"

    # base zoom poly (already converted n->t*fps upstream)
    # $zz is your cleaned/clamped zoom expression
    # For width/height we’ll phase-cap zoom:
    $zPre  = $zz
    $zShot = "min($zz,$ShotZoomCap)"
    $celeWidthCap = "min($CeleZoomCap,$celeTight)"
    $zCele = "min($zz,$celeWidthCap)"

    # even crop window using phase-zoom
    $wPre  = "max(16, floor(((ih*9/16)/($zPre))/2)*2)"
    $hPre  = "max(16, floor((ih/($zPre))/2)*2)"
    $wShot = "max(16, floor(((ih*9/16)/($zShot))/2)*2)"
    $hShot = "max(16, floor((ih/($zShot))/2)*2)"
    $wCele = "max(16, floor(((ih*9/16)/($zCele))/2)*2)"
    $hCele = "max(16, floor((ih/($zCele))/2)*2)"

    # pick w/h by phase
    $w = "if($inShot,$wShot, if($inCele,$wCele,$wPre))"
    $h = "if($inShot,$hShot, if($inCele,$hCele,$hPre))"

    # corridor clamps (early is wider & looser)
    $minXcEarly = "(($EarlyLeft+$EarlyPadPx)+($w)/2)"
    $maxXcEarly = "(($EarlyRight-$EarlyPadPx)-($w)/2)"
    $minXcGoal  = "(($goalLeft+$padPx)+($w)/2)"
    $maxXcGoal  = "(($goalRight-$padPx)-($w)/2)"

    # base track in time (already built as $cx, $cy using n->t*fps)
    $cxEarly = "min(max(($cx), $minXcEarly), $maxXcEarly)"
    $cxGoal  = "min(max(($cx), $minXcGoal),  $maxXcGoal)"

    # start ramp (smoothly move from midX to the clamped track)
    $r = "min(($t/$startRampSec),1)"
    $cxRampEarly = "( ($midX)*(1-($r)) + ($cxEarly)*($r) )"
    $cxRampGoal  = "( ($midX)*(1-($r)) + ($cxGoal)*($r) )"

    # shot: center the goal so we see frame+net
    $cxShot = $midX

    # celebration: blend between tracker and goal centre to follow scorer but keep goal visible
    $cxCeleFollow = "( ($cxGoal)*$CeleBlend + ($midX)*(1-$CeleBlend) )"

    # phase X centre
    $cxPhase = "if($inShot,$cxShot, if($inCele,$cxCeleFollow, $cxRampEarly))"
    # After PreWideSec expires, switch ramp source to goal corridor
    $cxPhase = "if(gt($t,$PreWideSec), if($inShot,$cxShot, if($inCele,$cxCeleFollow, $cxRampGoal)), $cxPhase)"

    # Y clamp (same band; you can loosen a little during shot)
    $yMinC   = "($yMin+24+($h)/2)"
    $yMaxC   = "($yMax-24-($h)/2)"
    $cyClamp = "min(max(($cy), $yMinC), $yMaxC)"

    # convert centre->top-left and stay in-bounds
    $x = "min(max((($cxPhase)-($w)/2),0), iw-($w))"
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
        [double]$PreWideSec = 2.0,
        [int]$EarlyLeft = 720,
        [int]$EarlyRight = 1200,
        [int]$EarlyPadPx = 24,
        [double]$ShotLead = 0.6,
        [double]$ShotTrail = 0.9,
        [double]$ShotZoomCap = 1.15,
        [double]$CeleBlend = 0.6,
        [double]$CeleZoomCap = 1.22,
        [switch]$ScaleFirst
    )
    if (-not (Test-Path $VarsPath)) { throw "Vars file not found: $VarsPath" }
    . $VarsPath
    if (-not ($cxExpr -and $cyExpr -and $zExpr)) { throw "Expected cxExpr, cyExpr, zExpr in $VarsPath" }
    return (Build-GoalCorridorChain -cxExpr $cxExpr -cyExpr $cyExpr -zExpr $zExpr -fps $fps `
        -goalLeft $goalLeft -goalRight $goalRight -padPx $padPx -startRampSec $startRampSec `
        -tGoalSec $tGoalSec -celeSec $celeSec -celeTight $celeTight -yMin $yMin -yMax $yMax `
        -PreWideSec $PreWideSec -EarlyLeft $EarlyLeft -EarlyRight $EarlyRight -EarlyPadPx $EarlyPadPx `
        -ShotLead $ShotLead -ShotTrail $ShotTrail -ShotZoomCap $ShotZoomCap -CeleBlend $CeleBlend `
        -CeleZoomCap $CeleZoomCap `
        -ScaleFirst:$ScaleFirst)
}

Export-ModuleMember -Function Convert-ToFrameExpr,Remove-ClipCalls,Get-SafeZoomExpr,Get-EvenWindowExprs,Build-GoalCorridorChain,New-VFChain
