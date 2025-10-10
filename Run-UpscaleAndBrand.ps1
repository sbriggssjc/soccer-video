[CmdletBinding()]
param(
    [int]$Scale = 2,
    [string]$Brand = 'tsc',
    [string]$OutDir = 'C:\Users\scott\soccer-video\out\portrait_reels\clean'
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = 'python'
$catalogPy = Join-Path $root 'tools\catalog.py'
$upscalePy = Join-Path $root 'tools\upscale.py'
$renderPy = Join-Path $root 'tools\render_follow_unified.py'

$atomicDir = Join-Path $root 'out\atomic_clips'
$upscaledDir = Join-Path $root 'out\upscaled'

if (!(Test-Path $atomicDir)) {
    throw "Atomic clip directory not found: $atomicDir"
}

if (!(Test-Path $upscaledDir)) {
    New-Item -ItemType Directory -Force -Path $upscaledDir | Out-Null
}

if (!(Test-Path $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

$clips = Get-ChildItem -Path $atomicDir -Filter '*.mp4' | Sort-Object FullName

if (-not $clips) {
    Write-Warning 'No atomic clips found. Add MP4s under out\atomic_clips\ before running.'
    return
}

foreach ($clip in $clips) {
    Write-Host "Processing $($clip.Name)..." -ForegroundColor Cyan
    $clipPath = $clip.FullName
    $clipStem = [System.IO.Path]::GetFileNameWithoutExtension($clipPath)
    $upscaledPath = Join-Path $upscaledDir ($clipStem + ('__x{0}.mp4' -f $Scale))
    $finalPath = Join-Path $OutDir ($clipStem + '_portrait_FINAL.mp4')
    $clipMTime = $clip.LastWriteTimeUtc

    try {
        $needUpscale = $true
        if (Test-Path $upscaledPath) {
            $upscaledInfo = Get-Item $upscaledPath
            if ($upscaledInfo.LastWriteTimeUtc -ge $clipMTime) {
                $needUpscale = $false
            }
        }

        if ($needUpscale) {
            Write-Host '  Upscaling with Real-ESRGAN...' -ForegroundColor Yellow
            & $python $upscalePy --in $clipPath --scale $Scale --out $upscaledPath
            if ($LASTEXITCODE -ne 0) {
                throw "Upscale failed with exit code $LASTEXITCODE"
            }
            $upscaledAt = (Get-Date).ToUniversalTime().ToString('o')
        } else {
            Write-Host '  Upscale already current; skipping.' -ForegroundColor DarkYellow
            $upscaledAt = (Get-Item $upscaledPath).LastWriteTimeUtc.ToString('o')
        }

        & $python $catalogPy --mark-upscaled --clip $clipPath --out $upscaledPath --scale $Scale --model 'realesrgan-x4plus' --at $upscaledAt
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to update upscale catalog.'
        }
    }
    catch {
        $message = $_.Exception.Message
        Write-Warning "  Upscale step failed: $message"
        & $python $catalogPy --mark-upscaled --clip $clipPath --error $message --at (Get-Date).ToUniversalTime().ToString('o')
        continue
    }

    try {
        $needBrand = $true
        if (Test-Path $finalPath) {
            $finalInfo = Get-Item $finalPath
            $upscaledInfo = Get-Item $upscaledPath
            if ($finalInfo.LastWriteTimeUtc -ge $clipMTime -and $finalInfo.LastWriteTimeUtc -ge $upscaledInfo.LastWriteTimeUtc) {
                $needBrand = $false
            }
        }

        if ($needBrand) {
            Write-Host '  Rendering portrait follow/crop + branding...' -ForegroundColor Yellow
            & $python $renderPy --in $clipPath --portrait 1080x1920 --brand $Brand --apply-polish --upscale --upscale-scale $Scale --outdir $OutDir
            if ($LASTEXITCODE -ne 0) {
                throw "Brand render failed with exit code $LASTEXITCODE"
            }
            if (!(Test-Path $finalPath)) {
                throw "Expected branded output not found: $finalPath"
            }
            $brandAt = (Get-Date).ToUniversalTime().ToString('o')
        } else {
            Write-Host '  Branded output already current; skipping.' -ForegroundColor DarkYellow
            $brandAt = (Get-Item $finalPath).LastWriteTimeUtc.ToString('o')
        }

        $argsList = @('--portrait','1080x1920','--brand',$Brand,'--apply-polish','--upscale','--upscale-scale',$Scale.ToString(),'--outdir',$OutDir)
        & $python $catalogPy --mark-branded --clip $clipPath --out $finalPath --brand $Brand --at $brandAt --args $argsList
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to update branding catalog.'
        }
    }
    catch {
        $message = $_.Exception.Message
        Write-Warning "  Branding step failed: $message"
        & $python $catalogPy --mark-branded --clip $clipPath --error $message --at (Get-Date).ToUniversalTime().ToString('o')
        continue
    }

    Write-Host '  Completed successfully.' -ForegroundColor Green
}

Write-Host 'Batch complete.' -ForegroundColor Green
