[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]]$Clips,
    [string]$OutDir,
    [switch]$ListOnly
)

$ErrorActionPreference = 'Stop'

function Resolve-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue
    if ($resolved) { return $resolved.Path }
    $combined = Join-Path (Get-Location) $Path
    $resolved = Resolve-Path -LiteralPath $combined -ErrorAction SilentlyContinue
    if ($resolved) { return $resolved.Path }
    return $Path
}

function Get-PythonCommand {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Path }
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Path }
    throw 'Python executable not found on PATH. Install Python 3.10+.'
}

$script:ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$script:RepoRoot = Split-Path -Parent $script:ScriptRoot
if (-not $PSBoundParameters.ContainsKey('OutDir') -or [string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $script:RepoRoot 'out'
}
$script:OutDir = Resolve-FullPath -Path $OutDir
$script:BallPlanner = Join-Path $script:RepoRoot 'tools\ball_path_offline.py'
$script:PythonExe = Get-PythonCommand

function Find-PlanJsonl {
    param([Parameter(Mandatory = $true)][string]$Stem)

    $diagRoot = Join-Path $script:OutDir 'follow_diag'
    $stemDir = Join-Path $diagRoot $Stem
    if (-not (Test-Path -LiteralPath $stemDir)) { return $null }

    $existing = Get-ChildItem -LiteralPath $stemDir -Recurse -File -Filter '*.jsonl' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($existing) { return $existing.FullName }
    return $null
}

function Ensure-PlanJsonl {
  param([string]$Stem, [string]$ClipPath)

  $existing = Find-PlanJsonl -Stem $Stem
  if ($existing) { return $existing }

  # Default plan path using your naming convention
  $plan = Join-Path $script:OutDir ("follow_diag\{0}\auto_hill\plan_dx-48_dy-108_sh-2.jsonl" -f $Stem)
  New-Item -Force -ItemType Directory -Path (Split-Path $plan) | Out-Null

  try {
    & $script:PythonExe $script:BallPlanner `
      --in "$ClipPath" `
      --out "$plan" `
      --dx -48 --dy -108 --sh 2
    if (Test-Path $plan) { return $plan }
  } catch {
    Write-Warning ("[PLAN] Auto-generation failed for {0}: {1}" -f $Stem, $_.Exception.Message)
  }

  return $null
}

$clipList = @()
foreach ($item in $Clips) {
    $resolved = Resolve-FullPath -Path $item
    if (-not (Test-Path -LiteralPath $resolved)) {
        Write-Warning ("[SKIP] Clip not found: {0}" -f $item)
        continue
    }
    $clipList += $resolved
}

if ($clipList.Count -eq 0) {
    Write-Warning 'No valid clips provided.'
    return
}

foreach ($Clip in $clipList) {
    $Stem = [System.IO.Path]::GetFileNameWithoutExtension($Clip)

    $Plan = Ensure-PlanJsonl -Stem $Stem -ClipPath $Clip
    if (-not $Plan) { Write-Host "[SKIP] No plan for $Stem (could not auto-generate)"; continue }

    if ($ListOnly) {
        Write-Host ("[LIST] {0} -> {1}" -f $Clip, $Plan)
        continue
    }

    Write-Host ("[READY] {0} using plan {1}" -f $Clip, $Plan)
}
#requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Proj   = "C:\Users\scott\soccer-video"
$OutDir = Join-Path $Proj "out"
$Report = Join-Path $OutDir "coverage_report.csv"

New-Item -Force -ItemType File -Path $Report | Out-Null
'clip,attempt,coverage,frames_ok,frames_total,first_miss_frame,first_miss_time,ball_x,ball_y,need_dx,need_dy,used_params' |
  Out-File -Encoding utf8 $Report

# Escalation tiers
$TIERS = @(
  '--follow-zeta','1.10','--follow-wn','5.4','--deadzone','2.5','--max-vel','900','--max-acc','4500','--lookahead','10','--pre-smooth','0.18','--zoom-out-max','1.45','--zoom-edge-frac','0.72','--cy-frac','0.46';
  '--follow-zeta','1.10','--follow-wn','5.4','--deadzone','2.5','--max-vel','900','--max-acc','4500','--lookahead','10','--pre-smooth','0.18','--zoom-out-max','1.50','--zoom-edge-frac','0.68','--cy-frac','0.46';
  '--follow-zeta','1.10','--follow-wn','5.6','--deadzone','2.0','--max-vel','1000','--max-acc','5000','--lookahead','12','--pre-smooth','0.16','--zoom-out-max','1.50','--zoom-edge-frac','0.68','--cy-frac','0.45'
)

function Find-PlanJsonl {
  param([string]$Stem)
  $root = Join-Path $OutDir ("follow_diag\{0}" -f $Stem)
  $cand = @()

  if (Test-Path $root) {
    $cand += Get-ChildItem -Recurse $root -Filter *.jsonl -ErrorAction SilentlyContinue
  } else {
    # fallback: any folder starting with the stem (sometimes stems differ by suffix)
    $parent = Join-Path $OutDir 'follow_diag'
    if (Test-Path $parent) {
      $cand += Get-ChildItem $parent -Directory -Filter ($Stem + '*') -ErrorAction SilentlyContinue |
              ForEach-Object { Get-ChildItem -Recurse $_.FullName -Filter *.jsonl -ErrorAction SilentlyContinue }
    }
  }

  $cand | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
}

function Invoke-Coverage {
  param([string]$Telemetry)
  $args = @('tools\coverage_from_telemetry.py','--telemetry',"$Telemetry",'--w','1920','--h','1080','--crop-w','560','--crop-h','936','--margin','60')
  $out  = & python @args 2>&1
  $text = ($out | Out-String)

  $ok=0; $tot=0; $pct=0.0
  if ($text -match 'camera-coverage:\s+(\d+)\/(\d+)\s*=\s*([\d\.]+)%') {
    $ok  = [int]$Matches[1]; $tot=[int]$Matches[2]; $pct=[double]$Matches[3]
  }
  $fm=''; $ft=''; $bx=''; $by=''; $ndx=''; $ndy=''
  if ($text -match 'first miss @ frame\s+(\d+)\s+t=([\d\.]+)s\s+ball=\(([\d\.]+),([\d\.]+)\)\s+need_dx=([-\d\.]+)\s+need_dy=([-\d\.]+)') {
    $fm=$Matches[1]; $ft=$Matches[2]; $bx=$Matches[3]; $by=$Matches[4]; $ndx=$Matches[5]; $ndy=$Matches[6]
  }
  [pscustomobject]@{ Pct=$pct; Ok=$ok; Tot=$tot; MissFrame=$fm; MissTime=$ft; BallX=$bx; BallY=$by; NeedDx=$ndx; NeedDy=$ndy; Raw=$text }
}

Get-ChildItem -Recurse (Join-Path $OutDir 'atomic_clips') -Filter '*.mp4' |
  Where-Object { $_.Name -notmatch '__SMOOTH' -and $_.Name -notmatch '__DEBUG' } |
  ForEach-Object {
    $Clip = $_.FullName
    $Stem = $_.BaseName

    $Plan = Find-PlanJsonl -Stem $Stem
    if (-not $Plan) { Write-Host "[SKIP] No plan for $Stem"; continue }

    $OutClip = Join-Path $_.Directory.FullName ($Stem + '.__SMOOTH_LOSTMOTION.mp4')
    $Tel     = $OutClip -replace '\.mp4$','.__telemetry.jsonl'

    $attempt=0; $best=$null; $usedTier=$null
    foreach ($tier in $TIERS) {
      $attempt++
      $args = @('tools\render_follow_unified.py',
        '--in',"$Clip",'--out',"$OutClip",'--portrait','1080x1920','--preset','cinematic',
        '--ball-path',"$Plan",
        '--lost-hold-ms','600','--lost-pan-ms','1400','--lost-lookahead-s','6',
        '--lost-use-motion','--lost-chase-motion-ms','900','--lost-motion-thresh','1.6',
        '--telemetry-out',"$Tel"
      ) + $tier

      & python @args | Out-Null

      $cov = Invoke-Coverage -Telemetry $Tel
      $best = $cov; $usedTier = ($tier -join ' ')
      "{0},{1},{2:N2}%,{3},{4},{5},{6},{7},{8},{9},{10},""{11}""" -f `
        $Stem,$attempt,$cov.Pct,$cov.Ok,$cov.Tot,$cov.MissFrame,$cov.MissTime,$cov.BallX,$cov.BallY,$cov.NeedDx,$cov.NeedDy,$usedTier |
        Out-File -Append -Encoding utf8 $Report

      if ($cov.Pct -ge 95) { break }
    }
    Write-Host ("[DONE] {0} → {1:N2}% (attempt {2})" -f $Stem, $best.Pct, $attempt)
  }

Write-Host "Coverage report: $Report"
