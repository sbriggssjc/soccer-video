param(
  [switch]$Run,      # actually execute
  [switch]$DryRun    # force preview
)

$ErrorActionPreference = 'Stop'
$Repo = $PWD
$CSV  = Join-Path $Repo 'out\reports\pipeline_status.csv'
if (!(Test-Path $CSV)) { throw "Not found: $CSV" }

# Detect tools you already have
$FollowPY   = Join-Path $Repo 'tools\render_follow_unified.py'
$UpscalePY  = Join-Path $Repo 'tools\upscale.py'
$EnhancePS1 = Join-Path $Repo 'tools\auto_enhance\auto_enhance.ps1'
$BrandPS1   = Join-Path $Repo 'tools\tsc_brand.ps1'

function Use-Tool([string]$Path){ Test-Path -LiteralPath $Path }

$DoRun = $false
if ($Run.IsPresent -and -not $DryRun.IsPresent) { $DoRun = $true }

# Ensure output folders exist
$ensure = @(
  'out\atomic_clips_follow',
  'out\upscaled',
  'out\enhanced',
  'out\portrait_reels\clean',
  'out\logs'
) | % { Join-Path $Repo $_ } | % { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

$rows = Import-Csv $CSV
$work = @()

foreach ($r in $rows) {
  $clipID = $r.ClipID
  $atomic = $r.AtomicPath
  if (!(Test-Path $atomic)) { continue }

  # === FOLLOW/STABILIZE ===
  if ($r.Stage_Follow -ne 'True') {
    if (Use-Tool $FollowPY) {
      $outFollow = Join-Path $Repo ("out\\atomic_clips_follow\\{0}__FOLLOW.mp4" -f $clipID)
      $work += [pscustomobject]@{
        Stage='FOLLOW'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$atomic; Out=$outFollow
        Cmd=@('python', $FollowPY, '--in', $atomic, '--out', $outFollow, '--preset', 'cinematic', '--clean-temp')
      }
    }
  }

  # pick best available input for later stages
  $inFollow = Join-Path $Repo ("out\\atomic_clips_follow\\{0}__FOLLOW.mp4" -f $clipID)
  $bestForUpscale = (Test-Path $inFollow) ? $inFollow : $atomic

  # === UPSCALE ===
  if ($r.Stage_Upscaled -ne 'True') {
    $outUp = Join-Path $Repo ("out\\upscaled\\{0}__UP.mp4" -f $clipID)
    if (Use-Tool $UpscalePY) {
      $work += [pscustomobject]@{
        Stage='UPSCALE'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$bestForUpscale; Out=$outUp
        Cmd=@('python','-c',("import sys; sys.path.insert(0, r'{0}'); from upscale import upscale_video; upscale_video(r'{1}', scale=2)" -f (Split-Path $UpscalePY -Parent), $bestForUpscale))
      }
    } else {
      $work += [pscustomobject]@{
        Stage='UPSCALE'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$bestForUpscale; Out=$outUp
        Cmd=@('ffmpeg','-hide_banner','-y','-i', $bestForUpscale, '-vf','scale=1080:1920:flags=lanczos','-c:v','libx264','-preset','slow','-crf','18', $outUp)
      }
    }
  }

  # choose best input for enhance
  $inUp = Join-Path $Repo ("out\\upscaled\\{0}__UP.mp4" -f $clipID)
  $bestForEnh = (Test-Path $inUp) ? $inUp : $bestForUpscale

  # === ENHANCE ===
  if ($r.Stage_Enhanced -ne 'True') {
    $outEnh = Join-Path $Repo ("out\\enhanced\\{0}__ENH.mp4" -f $clipID)
    if (Use-Tool $EnhancePS1) {
      $work += [pscustomobject]@{
        Stage='ENHANCE'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$bestForEnh; Out=$outEnh
        Cmd=@('powershell','-NoProfile','-ExecutionPolicy','Bypass','-File', $EnhancePS1, '-In', $bestForEnh, '-Out', $outEnh)
      }
    } else {
      $work += [pscustomobject]@{
        Stage='ENHANCE'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$bestForEnh; Out=$outEnh
        Cmd=@('ffmpeg','-hide_banner','-y','-i', $bestForEnh, '-vf','hqdn3d=2:1:3:3,unsharp=5:5:0.5:5:5:0.0','-c:v','libx264','-preset','slow','-crf','18', $outEnh)
      }
    }
  }

  # choose best input for brand
  $bestForBrand = (Test-Path $outEnh) ? $outEnh : ((Test-Path $inUp) ? $inUp : $bestForUpscale)

  # === BRAND ===
  if ($r.Stage_Branded -ne 'True') {
    $outBrand = Join-Path $Repo ("out\\portrait_reels\\clean\\{0}__BRAND.mp4" -f $clipID)
    if (Use-Tool $BrandPS1) {
      $work += [pscustomobject]@{
        Stage='BRAND'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$bestForBrand; Out=$outBrand
        Cmd=@('powershell','-NoProfile','-ExecutionPolicy','Bypass','-File', $BrandPS1, '-In', $bestForBrand, '-Out', $outBrand, '-Aspect','9x16')
      }
    }
    else {
      # if no branding script, skip (or add your ffmpeg overlay here)
    }
  }
}

if (-not $work.Count) { Write-Host 'No pending stages (from CSV).'; return }

# Group & run
$log = Join-Path $Repo ("out\\logs\\advance_from_csv.{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
"Log: $log"

$work | Sort-Object Stage, MatchKey, ClipID |
  Group-Object Stage | ForEach-Object {
    Write-Host ("### Stage: {0}" -f $_.Name) -ForegroundColor Cyan
    $_.Group | Group-Object MatchKey | ForEach-Object {
      Write-Host ("# Match: {0}" -f $_.Name) -ForegroundColor Yellow
      foreach ($w in $_.Group) {
        $cmd = $w.Cmd
        $cmdString = ($cmd | % { '"' + $_ + '"' }) -join ' '
        Add-Content -Path $log -Value $cmdString

        # idempotent: skip if Out newer than In
        $skip = $false
        if ((Test-Path $w.In) -and (Test-Path $w.Out)) {
          $inItem  = Get-Item $w.In
          $outItem = Get-Item $w.Out
          if ($outItem.LastWriteTimeUtc -ge $inItem.LastWriteTimeUtc) { $skip = $true }
        }
        if ($skip) { Write-Host "[SKIP] $cmdString"; Add-Content -Path $log -Value '[SKIP]'; continue }

        if (-not $DoRun) { Write-Host "[RUN?] $cmdString"; continue }

        $dir = Split-Path -Parent $w.Out
        if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $cmd[0]
        if ($cmd.Count -gt 1) {
          $psi.Arguments = [string]::Join(' ', ($cmd[1..($cmd.Count-1)] | % { '"' + $_ + '"' }))
        }
        $psi.WorkingDirectory = $Repo
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()
        if ($stdout) { Add-Content -Path $log -Value $stdout }
        if ($stderr) { Add-Content -Path $log -Value $stderr }
        if ($p.ExitCode -ne 0) { Write-Warning "ExitCode=$($p.ExitCode)" }
      }
    }
  }

Write-Host "Done. Log: $log"
