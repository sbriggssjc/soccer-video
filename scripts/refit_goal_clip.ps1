param(
  [string]$In = '.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4',
  [string]$Vars = '.\out\autoframe_work\004__GOAL__t266.50-t283.10_zoom.ps1vars',
  [string]$Out = '',
  [string]$Csv = '.\out\autoframe_work\004__GOAL__t266.50-t283.10_zoom.csv',
  [string]$Config = '',
  [int]$Fps = 24,
  [int]$GoalLeft = 840,
  [int]$GoalRight = 1080,
  [int]$PadPx = 40,
  [double]$StartRampSec = 3.0,
  [double]$TGoalSec = 10.0,
  [double]$CeleSec = 2.0,
  [double]$CeleTight = 1.80,
  [double]$YMin = 360,
  [double]$YMax = 780,
  [string]$MotionCsv,
  [switch]$ScaleFirst
)

if ([string]::IsNullOrWhiteSpace($In)) {
    throw 'Input clip path is required.'
}

if ([string]::IsNullOrWhiteSpace($Vars)) {
    throw 'Vars path is required.'
}

if ([string]::IsNullOrWhiteSpace($Out)) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($In)
    $dir = Split-Path $In -Parent
    if (-not $dir) { $dir = '.' }
    $Out = Join-Path $dir ($stem + '.refit.mp4')
}

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $PSScriptRoot '..\configs\zoom.yaml'
}

Import-Module "$PSScriptRoot\..\tools\Autoframe.psm1" -Force

Remove-Item $Vars -Force -ErrorAction SilentlyContinue

$pythonArgs = @('.\fit_expr.py')
if (-not [string]::IsNullOrWhiteSpace($MotionCsv)) {
    $pythonArgs += @('--csv', $MotionCsv)
}
if (-not [string]::IsNullOrWhiteSpace($Csv)) {
    $pythonArgs += @('--csv', $Csv)
}
$pythonArgs += @(
    '--out', $Vars,
    '--degree', '3',
    '--profile', 'landscape',
    '--roi', 'goal',
    '--config', $Config,
    '--lead-ms', '16',
    '--alpha-slow', '0.06',
    '--alpha-fast', '0.70',
    '--deadzone', '10',
    '--boot-wide-ms', '4200',
    '--snap-accel-th', '0.16',
    '--snap-widen', '0.50',
    '--snap-decay-ms', '380',
    '--snap-hold-ms', '220',
    '--z-tight', '1.65',
    '--z-wide', '1.04',
    '--zoom-tighten-rate', '0.010',
    '--zoom-widen-rate', '0.160',
    '--v-enabled',
    '--v-gain', '0.85',
    '--v-deadzone', '10',
    '--v-top-margin', '60',
    '--v-bottom-margin', '60',
    '--goal-bias', '0.60',
    '--conf-th', '0.70',
    '--celebration-ms', '2200',
    '--celebration-tight', '2.20'
)

Write-Host ("Running: python {0}" -f ($pythonArgs -join ' '))
python @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "fit_expr.py exited with code $LASTEXITCODE"
}

$vf = New-VFChain `
    -Vars $Vars `
    -fps $Fps `
    -goalLeft $GoalLeft `
    -goalRight $GoalRight `
    -padPx $PadPx `
    -startRampSec $StartRampSec `
    -tGoalSec $TGoalSec `
    -celeSec $CeleSec `
    -celeTight $CeleTight `
    -yMin $YMin `
    -yMax $YMax `
    -ScaleFirst:$ScaleFirst

ffmpeg -y -i $In -vf $vf -c:v libx264 -crf 19 -preset veryfast $Out
