[CmdletBinding()]
param(
    [switch]$OnlyUpscale,
    [switch]$OnlyBrand,
    [string]$Since,
    [string]$Clip
)

if ($OnlyUpscale -and $OnlyBrand) {
    throw 'Specify only one of -OnlyUpscale or -OnlyBrand.'
}

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = 'python'
$catalogPy = Join-Path $root 'tools\catalog.py'
$upscalePy = Join-Path $root 'tools\upscale.py'
$renderPy = Join-Path $root 'tools\render_follow_unified.py'

$statusCsv = Join-Path $root 'out\catalog\pipeline_status.csv'
if (!(Test-Path $statusCsv)) {
    Write-Warning 'No pipeline_status.csv found. Run Build-AtomicIndex.ps1 and process clips first.'
    return
}

$rows = Import-Csv $statusCsv
if (-not $rows) {
    Write-Warning 'pipeline_status.csv is empty.'
    return
}

if ($Since) {
    try {
        $sinceDate = [DateTime]::Parse($Since, [System.Globalization.CultureInfo]::InvariantCulture).ToUniversalTime()
        $rows = $rows | Where-Object {
            if ([string]::IsNullOrWhiteSpace($_.last_run_at)) { return $true }
            try {
                return ([DateTime]$_.last_run_at).ToUniversalTime() -ge $sinceDate
            } catch {
                return $true
            }
        }
    } catch {
        throw "Unable to parse -Since value '$Since'. Use YYYY-MM-DD or ISO8601."
    }
}

if ($Clip) {
    $rows = $rows | Where-Object { $_.clip_path -like "*${Clip}*" }
}

if (-not $rows) {
    Write-Warning 'No matching entries after filtering.'
    return
}

foreach ($row in $rows) {
    $clipPath = $row.clip_path
    if (-not (Test-Path $clipPath)) {
        Write-Warning "Clip missing on disk: $clipPath"
        continue
    }

    $clipItem = Get-Item $clipPath
    $clipStem = [System.IO.Path]::GetFileNameWithoutExtension($clipPath)

    $sidecarPath = Join-Path $root ("out\\catalog\\sidecar\\$clipStem.json")
    $sidecar = $null
    if (Test-Path $sidecarPath) {
        $sidecar = Get-Content $sidecarPath -Raw | ConvertFrom-Json
    }

    $scale = 2
    if ($sidecar -and $sidecar.steps -and $sidecar.steps.upscale -and $sidecar.steps.upscale.scale) {
        $scale = [int]$sidecar.steps.upscale.scale
    } elseif ($row.upscaled_path -match '__x(\d+)') {
        $scale = [int]$Matches[1]
    }

    $brand = 'tsc'
    if ($sidecar -and $sidecar.steps -and $sidecar.steps.follow_crop_brand -and $sidecar.steps.follow_crop_brand.brand) {
        $brand = $sidecar.steps.follow_crop_brand.brand
    }

    $outDir = 'C:\Users\scott\soccer-video\out\portrait_reels\clean'
    if ($row.branded_path) {
        $outDir = Split-Path $row.branded_path
    } elseif ($sidecar -and $sidecar.steps -and $sidecar.steps.follow_crop_brand -and $sidecar.steps.follow_crop_brand.out) {
        $outDir = Split-Path $sidecar.steps.follow_crop_brand.out
    }

    $upscaledPath = $row.upscaled_path
    if (-not $upscaledPath) {
        $upscaledPath = Join-Path $root ("out\\upscaled\\$clipStem" + ('__x{0}.mp4' -f $scale))
    }

    $finalPath = $row.branded_path
    if (-not $finalPath) {
        $finalPath = Join-Path $outDir ($clipStem + '_portrait_FINAL.mp4')
    }

    Write-Host "Resuming $clipStem" -ForegroundColor Cyan

    $clipMTime = $clipItem.LastWriteTimeUtc

    if (-not $OnlyBrand) {
        $runUpscale = $false
        if (-not (Test-Path $upscaledPath)) {
            $runUpscale = $true
        } else {
            $upscaledInfo = Get-Item $upscaledPath
            if ($upscaledInfo.LastWriteTimeUtc -lt $clipMTime) {
                $runUpscale = $true
            }
        }
        if ($row.last_error -and ([string]::IsNullOrWhiteSpace($row.upscaled_path) -or $row.last_error -like '*upscale*')) {
            $runUpscale = $true
        }

        if ($runUpscale) {
            try {
                Write-Host '  Re-running upscale...' -ForegroundColor Yellow
                & $python $upscalePy --in $clipPath --scale $scale --out $upscaledPath
                if ($LASTEXITCODE -ne 0) {
                    throw "Upscale failed with exit code $LASTEXITCODE"
                }
                $upscaledAt = (Get-Date).ToUniversalTime().ToString('o')
            } catch {
                $message = $_.Exception.Message
                Write-Warning "  Upscale retry failed: $message"
                & $python $catalogPy --mark-upscaled --clip $clipPath --error $message --at (Get-Date).ToUniversalTime().ToString('o')
                continue
            }

            & $python $catalogPy --mark-upscaled --clip $clipPath --out $upscaledPath --scale $scale --model 'realesrgan-x4plus' --at $upscaledAt
            if ($LASTEXITCODE -ne 0) {
                Write-Warning '  Failed to update upscale catalog after retry.'
                continue
            }
        }
    }

    if (-not $OnlyUpscale) {
        $runBrand = $false
        if (-not (Test-Path $finalPath)) {
            $runBrand = $true
        } elseif (-not (Test-Path $upscaledPath)) {
            $runBrand = $true
        } else {
            $finalInfo = Get-Item $finalPath
            $upscaledInfo = Get-Item $upscaledPath
            if ($finalInfo.LastWriteTimeUtc -lt $clipMTime -or $finalInfo.LastWriteTimeUtc -lt $upscaledInfo.LastWriteTimeUtc) {
                $runBrand = $true
            }
        }
        if ($row.last_error -and -not $OnlyUpscale) {
            $runBrand = $true
        }

        if ($runBrand) {
            try {
                Write-Host '  Re-running portrait render...' -ForegroundColor Yellow
                & $python $renderPy --in $clipPath --portrait 1080x1920 --brand $brand --apply-polish --upscale --upscale-scale $scale --outdir $outDir
                if ($LASTEXITCODE -ne 0) {
                    throw "Brand render failed with exit code $LASTEXITCODE"
                }
                if (!(Test-Path $finalPath)) {
                    throw "Expected branded output not found: $finalPath"
                }
                $brandAt = (Get-Date).ToUniversalTime().ToString('o')
            } catch {
                $message = $_.Exception.Message
                Write-Warning "  Branding retry failed: $message"
                & $python $catalogPy --mark-branded --clip $clipPath --error $message --at (Get-Date).ToUniversalTime().ToString('o')
                continue
            }

            $argsList = @('--portrait','1080x1920','--brand',$brand,'--apply-polish','--upscale','--upscale-scale',$scale.ToString(),'--outdir',$outDir)
            & $python $catalogPy --mark-branded --clip $clipPath --out $finalPath --brand $brand --at $brandAt --args $argsList
            if ($LASTEXITCODE -ne 0) {
                Write-Warning '  Failed to update branding catalog after retry.'
                continue
            }
        }
    }

    Write-Host '  Resume complete.' -ForegroundColor Green
}

Write-Host 'Resume run finished.' -ForegroundColor Green
