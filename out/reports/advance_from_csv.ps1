param(
  [switch]$Run,      # actually execute commands
  [switch]$DryRun    # force preview even if -Run is present
)

$ErrorActionPreference = 'Stop'

# repo + inputs
$Repo = (Get-Location).Path
$CSV  = Join-Path $Repo 'out\reports\pipeline_status.csv'
if (!(Test-Path -LiteralPath $CSV)) { throw "Not found: $CSV" }

# tool discovery (adjust to your actual tools if names differ)
$FollowPY   = Join-Path $Repo 'tools\render_follow_unified.py'
$UpscalePY  = Join-Path $Repo 'tools\upscale.py'
$EnhancePS1 = Join-Path $Repo 'tools\auto_enhance\auto_enhance.ps1'
$BrandPS1   = Join-Path $Repo 'tools\tsc_brand.ps1'

function Test-Tool { param([string]$Path) return [bool](Test-Path -LiteralPath $Path) }

# ensure output folders
foreach($rel in @(
  'out\atomic_clips_follow','out\upscaled','out\enhanced','out\portrait_reels\clean','out\logs'
)){
  $p = Join-Path $Repo $rel
  if (!(Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

# exec mode
$DoRun = $false
if ($Run.IsPresent -and -not $DryRun.IsPresent) { $DoRun = $true }

$rows = Import-Csv -LiteralPath $CSV
$work = New-Object System.Collections.Generic.List[object]

foreach ($r in $rows) {
  $clipID = $r.ClipID
  $atomic = $r.AtomicPath
  if (-not $clipID -or -not $atomic -or -not (Test-Path -LiteralPath $atomic)) { continue }

  # Possible intermediates (compute paths)
  $outFollow = Join-Path $Repo ("out\\atomic_clips_follow\\{0}__FOLLOW.mp4" -f $clipID)
  $outUp     = Join-Path $Repo ("out\\upscaled\\{0}__UP.mp4"            -f $clipID)
  $outEnh    = Join-Path $Repo ("out\\enhanced\\{0}__ENH.mp4"           -f $clipID)
  $outBrand  = Join-Path $Repo ("out\\portrait_reels\\clean\\{0}__BRAND.mp4" -f $clipID)

  # FOLLOW (stabilize/zoom/pan/ball follow)
  if ($r.Stage_Follow -ne 'True' -and (Test-Tool $FollowPY)) {
    $work.Add([pscustomobject]@{
      Stage='FOLLOW'; MatchKey=$r.MatchKey; ClipID=$clipID
      In=$atomic; Out=$outFollow
      Cmd=@('python', $FollowPY, '--in', $atomic, '--out', $outFollow, '--preset', 'cinematic', '--clean-temp')
    })
  }

  # Choose best source for UPSCALE: prefer follow if exists
  $bestForUpscale = $atomic
  if (Test-Path -LiteralPath $outFollow) { $bestForUpscale = $outFollow }

  # UPSCALE
  if ($r.Stage_Upscaled -ne 'True') {
    if (Test-Tool $UpscalePY) {
      # Use your repo's Python upscaler if present
      $work.Add([pscustomobject]@{
        Stage='UPSCALE'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$bestForUpscale; Out=$outUp
        Cmd=@('python','-c',("import sys; sys.path.insert(0, r'{0}'); from upscale import upscale_video; upscale_video(r'{1}', scale=2)" -f (Split-Path $UpscalePY -Parent), $bestForUpscale))
      })
    } else {
      # Fallback to ffmpeg
      $work.Add([pscustomobject]@{
        Stage='UPSCALE'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$bestForUpscale; Out=$outUp
        Cmd=@('ffmpeg','-hide_banner','-y','-i', $bestForUpscale,'-vf','scale=1080:1920:flags=lanczos','-c:v','libx264','-preset','slow','-crf','18', $outUp)
      })
    }
  }

  # Choose best source for ENHANCE: prefer upscaled if exists
  $bestForEnh = $bestForUpscale
  if (Test-Path -LiteralPath $outUp) { $bestForEnh = $outUp }

  # ENHANCE
  if ($r.Stage_Enhanced -ne 'True') {
    if (Test-Tool $EnhancePS1) {
      $work.Add([pscustomobject]@{
        Stage='ENHANCE'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$bestForEnh; Out=$outEnh
        Cmd=@('powershell','-NoProfile','-ExecutionPolicy','Bypass','-File', $EnhancePS1, '-In', $bestForEnh, '-Out', $outEnh)
      })
    } else {
      $work.Add([pscustomobject]@{
        Stage='ENHANCE'; MatchKey=$r.MatchKey; ClipID=$clipID
        In=$bestForEnh; Out=$outEnh
        Cmd=@('ffmpeg','-hide_banner','-y','-i', $bestForEnh,'-vf','hqdn3d=2:1:3:3,unsharp=5:5:0.5:5:5:0.0','-c:v','libx264','-preset','slow','-crf','18', $outEnh)
      })
    }
  }

  # Choose best source for BRAND: prefer enhanced, else upscaled, else follow/atomic
  $bestForBrand = $bestForEnh
  if (Test-Path -LiteralPath $outEnh) {
    $bestForBrand = $outEnh
  } elseif (Test-Path -LiteralPath $outUp) {
    $bestForBrand = $outUp
  } elseif (Test-Path -LiteralPath $outFollow) {
    $bestForBrand = $outFollow
  } else {
    $bestForBrand = $atomic
  }

  # BRAND (only if you have a branding script)
  if ($r.Stage_Branded -ne 'True' -and (Test-Tool $BrandPS1)) {
    $work.Add([pscustomobject]@{
      Stage='BRAND'; MatchKey=$r.MatchKey; ClipID=$clipID
      In=$bestForBrand; Out=$outBrand
      Cmd=@('powershell','-NoProfile','-ExecutionPolicy','Bypass','-File', $BrandPS1, '-In', $bestForBrand, '-Out', $outBrand, '-Aspect','9x16')
    })
  }
}

if ($work.Count -eq 0) {
  Write-Host 'No pending stages (from CSV).'
  return
}

$log = Join-Path $Repo ("out\\logs\\advance_from_csv.{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
Write-Host ("Log: {0}" -f $log)

$work |
  Sort-Object Stage, MatchKey, ClipID |
  Group-Object Stage | ForEach-Object {
    Write-Host ("### Stage: {0}" -f $_.Name) -ForegroundColor Cyan
    $_.Group | Group-Object MatchKey | ForEach-Object {
      Write-Host ("# Match: {0}" -f $_.Name) -ForegroundColor Yellow
      foreach ($w in $_.Group) {
        $cmd = $w.Cmd
        $cmdString = ($cmd | ForEach-Object { '"' + $_ + '"' }) -join ' '
        Add-Content -Path $log -Value $cmdString

        # idempotent: skip if Out newer than In
        $skip = $false
        if ((Test-Path -LiteralPath $w.In) -and (Test-Path -LiteralPath $w.Out)) {
          $inItem  = Get-Item -LiteralPath $w.In
          $outItem = Get-Item -LiteralPath $w.Out
          if ($outItem.LastWriteTimeUtc -ge $inItem.LastWriteTimeUtc) { $skip = $true }
        }
        if (-not $DoRun) { Write-Host "[RUN?] $cmdString"; continue }
        if ($skip) { Write-Host "[SKIP] $cmdString"; Add-Content -Path $log -Value '[SKIP]'; continue }

        $dir = Split-Path -Parent $w.Out
        if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $cmd[0]
        if ($cmd.Count -gt 1) {
          $psi.Arguments = [string]::Join(' ', ($cmd[1..($cmd.Count-1)] | ForEach-Object { '"' + $_ + '"' }))
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
        $ec = $p.ExitCode
        if ($ec -ne 0) {
          Write-Warning ("ExitCode={0}" -f $ec)
          Write-Warning ("CMD: {0}" -f ($cmd -join ' '))
          $err = $stderr
          Write-Warning ("STDERR: {0}" -f (($err | Out-String).TrimEnd()))
        }
      }
    }
  }

Write-Host "Done. Log: $log"


