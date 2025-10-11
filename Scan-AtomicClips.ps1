[CmdletBinding()]
param(
  [Parameter(Position=0)] [string]$RepoRoot      = ".",
  [Parameter(Position=1)] [string]$CinematicRoot = ".\out\autoframe_work\cinematic",
  [Parameter(Position=2)] [string]$BrandedRoot   = ".\out\portrait_reels\branded",
  [Parameter(Position=3)] [string]$OutCsv        = ".\atomic_clips_index.rebuilt.csv",
  [Parameter(Position=4)] [string]$OutJson       = ".\atomic_clips_index.rebuilt.json"
)

$ErrorActionPreference = 'Stop'

# Determine repository root based on the script location
if (-not $PSBoundParameters.ContainsKey('RepoRoot') -or [string]::IsNullOrWhiteSpace($RepoRoot) -or $RepoRoot -eq '.') {
    $RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$repoRoot = (Resolve-Path -Path $RepoRoot).ProviderPath
Set-Location $repoRoot

$logPath = Join-Path $repoRoot 'atomic_clips_scan.log'
if (Test-Path $logPath) {
    Remove-Item $logPath -ErrorAction SilentlyContinue
}

function Write-LogMessage {
    param(
        [string]$Message
    )

    $timestamp = (Get-Date).ToString('s')
    $entry = "[{0}] {1}" -f $timestamp, $Message
    Add-Content -Path $logPath -Value $entry -Encoding UTF8
}

function Normalize-Token {
    param(
        [string]$Text
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ''
    }

    $lower = $Text.ToLowerInvariant()
    $collapsed = $lower -replace '[\s_\-]+', ''
    return $collapsed
}

function Get-ArtifactScore {
    param($Row)

    $score = 0
    if ($Row.proxy_exists -eq 'true') {
        $score++
    }
    if ($Row.trf_exists -eq 'true') {
        $score++
    }
    if ($Row.stabilized_exists -eq 'true') {
        $score++
    }
    if ($Row.has_branded -eq 'true') {
        $score++
    }
    return $score
}

function Add-Note {
    param(
        [string]$Existing,
        [string]$NewNote
    )

    if ([string]::IsNullOrWhiteSpace($NewNote)) {
        return $Existing
    }

    if ([string]::IsNullOrWhiteSpace($Existing)) {
        return $NewNote
    }

    return ($Existing + '; ' + $NewNote)
}

$culture = [System.Globalization.CultureInfo]::InvariantCulture
if ([System.IO.Path]::IsPathRooted($CinematicRoot)) {
    $cinematicRoot = $CinematicRoot
}
else {
    $cinematicRoot = Join-Path $repoRoot $CinematicRoot
}
if ([System.IO.Path]::IsPathRooted($BrandedRoot)) {
    $brandedRoot = $BrandedRoot
}
else {
    $brandedRoot = Join-Path $repoRoot $BrandedRoot
}

if (-not (Test-Path $cinematicRoot)) {
    throw "Cinematic root not found: $cinematicRoot"
}

$legacyFiles = @(
    'atomic_clips_index.csv',
    'atomic_clips_index.json',
    'atomic_clips_index.rebuilt.csv',
    'atomic_clips_index.rebuilt.json',
    'atomic_clips_index.merged.csv'
)
$legacyClipIds = New-Object System.Collections.Generic.HashSet[string]
foreach ($legacyName in $legacyFiles) {
    $legacyPath = Join-Path $repoRoot $legacyName
    if (-not (Test-Path $legacyPath)) {
        continue
    }

    try {
        if ($legacyName.ToLowerInvariant().EndsWith('.csv')) {
            $legacyRows = Import-Csv -Path $legacyPath -ErrorAction Stop
            foreach ($legacyRow in $legacyRows) {
                if ($legacyRow.PSObject.Properties.Name -contains 'clip_id') {
                    $legacyClipId = $legacyRow.clip_id
                    if (-not [string]::IsNullOrWhiteSpace($legacyClipId)) {
                        $null = $legacyClipIds.Add($legacyClipId)
                    }
                }
            }
        }
        elseif ($legacyName.ToLowerInvariant().EndsWith('.json')) {
            $legacyJson = Get-Content -Path $legacyPath -Raw -Encoding UTF8
            if (-not [string]::IsNullOrWhiteSpace($legacyJson)) {
                $legacyData = $legacyJson | ConvertFrom-Json
                if ($legacyData -is [System.Collections.IEnumerable]) {
                    foreach ($item in $legacyData) {
                        if ($item.PSObject.Properties.Name -contains 'clip_id') {
                            $legacyClipId = $item.clip_id
                            if (-not [string]::IsNullOrWhiteSpace($legacyClipId)) {
                                $null = $legacyClipIds.Add($legacyClipId)
                            }
                        }
                    }
                }
            }
        }
    }
    catch {
        Write-LogMessage ("Failed to read legacy index {0}: {1}" -f $legacyPath, $_)
    }
}

$brandedInfo = @()
if (Test-Path $brandedRoot) {
    $brandedFiles = Get-ChildItem -Path $brandedRoot -Recurse -File -Filter '*.mp4'
    foreach ($brand in $brandedFiles) {
        $brandName = [System.IO.Path]::GetFileNameWithoutExtension($brand.Name)
        $brandDate = ''
        $dateMatch = [regex]::Match($brandName, '\d{4}-\d{2}-\d{2}')
        if ($dateMatch.Success) {
            $brandDate = $dateMatch.Value
        }
        $normalizedName = Normalize-Token $brandName
        $brandedInfo += [pscustomobject]@{
            Path = $brand.FullName
            Date = $brandDate
            NormalizedName = $normalizedName
            Matched = $false
        }
    }
}

$pattern = '^(?<index>\d{1,3})__(?<date>\d{4}-\d{2}-\d{2})__(?<homeTeam>.+?)_vs_(?<awayTeam>.+?)__(?<label>.+?)__t(?<tStart>\d+(?:\.\d+)?)\-t(?<tEnd>\d+(?:\.\d+)?)(?<rest>.*)$'
$rowsById = @{}
$unparsedFolders = @()

$candidateDirs = Get-ChildItem -Path $cinematicRoot -Recurse -Directory | Where-Object { $_.Name -match '^\d{1,3}__' }
foreach ($dir in $candidateDirs) {
    $name = $dir.Name
    $match = [regex]::Match($name, $pattern)
    if (-not $match.Success) {
        $warning = "Unparsed folder name (check pattern): $name"
        Write-Warning $warning
        Write-LogMessage $warning
        $unparsedFolders += $name
        continue
    }

    $clipIndex = [int]$match.Groups['index'].Value
    $date = $match.Groups['date'].Value
    $homeTeam = $match.Groups['homeTeam'].Value
    $awayTeam = $match.Groups['awayTeam'].Value
    $label = $match.Groups['label'].Value
    $tStart = [double]::Parse($match.Groups['tStart'].Value, $culture)
    $tEnd = [double]::Parse($match.Groups['tEnd'].Value, $culture)
    $duration = [math]::Round(($tEnd - $tStart), 2)
    $rest = $match.Groups['rest'].Value
    $fullName = $name

    $hasDebug = $false
    if ($rest -match 'DEBUG' -or $fullName -match 'DEBUG') {
        $hasDebug = $true
    }

    $hasFinal = $false
    if ($rest -match 'FINAL' -or $fullName -match 'FINAL') {
        $hasFinal = $true
    }

    $orientation = 'unknown'
    if ($rest -match 'portrait' -or $fullName -match 'portrait') {
        $orientation = 'portrait'
    }

    $proxyCandidate = Join-Path $dir.FullName 'proxy.mp4'
    if (Test-Path $proxyCandidate) {
        $proxyPath = $proxyCandidate
        $proxyExists = 'true'
    }
    else {
        $proxyPath = ''
        $proxyExists = 'false'
    }

    $followDir = Join-Path $dir.FullName 'follow'
    $trfCandidate = Join-Path $followDir 'transforms.trf'
    if (Test-Path $trfCandidate) {
        $trfPath = $trfCandidate
        $trfExists = 'true'
    }
    else {
        $trfPath = ''
        $trfExists = 'false'
    }

    $stabilizedCandidate = Join-Path $followDir 'stabilized.mp4'
    if (Test-Path $stabilizedCandidate) {
        $stabilizedPath = $stabilizedCandidate
        $stabilizedExists = 'true'
    }
    else {
        $stabilizedPath = ''
        $stabilizedExists = 'false'
    }

    $clipId = '{0}|{1}|{2}|{3}|{4}|{5}|{6}' -f $clipIndex, $date, $homeTeam, $awayTeam, $label, $match.Groups['tStart'].Value, $match.Groups['tEnd'].Value

    $notes = ''
    if ($legacyClipIds.Contains($clipId)) {
        $notes = Add-Note $notes 'seen in legacy index'
    }

    if ($proxyExists -ne 'true') {
        $notes = Add-Note $notes 'missing proxy'
    }
    if ($trfExists -ne 'true') {
        $notes = Add-Note $notes 'missing transforms'
    }
    if ($stabilizedExists -ne 'true') {
        $notes = Add-Note $notes 'missing stabilized'
    }

    $row = [pscustomobject][ordered]@{
        clip_id = $clipId
        index = $clipIndex
        date = $date
        homeTeam = $homeTeam
        awayTeam = $awayTeam
        label = $label
        tStart = [math]::Round($tStart, 2)
        tEnd = [math]::Round($tEnd, 2)
        duration_sec = $duration
        orientation = $orientation
        hasDebug = $hasDebug
        hasFinal = $hasFinal
        root_dir = $dir.FullName
        proxy_path = $proxyPath
        trf_path = $trfPath
        stabilized_path = $stabilizedPath
        branded_matches = ''
        proxy_exists = $proxyExists
        trf_exists = $trfExists
        stabilized_exists = $stabilizedExists
        has_branded = 'false'
        notes = $notes
    }

    if ($rowsById.ContainsKey($clipId)) {
        $existingRow = $rowsById[$clipId]
        $newScore = Get-ArtifactScore $row
        $existingScore = Get-ArtifactScore $existingRow
        if ($newScore -gt $existingScore) {
            $rowsById[$clipId] = $row
        }
    }
    else {
        $rowsById[$clipId] = $row
    }
}

$rows = @()
foreach ($value in $rowsById.Values) {
    $rows += $value
}

foreach ($row in $rows) {
    $homeNorm = Normalize-Token $row.homeTeam
    $awayNorm = Normalize-Token $row.awayTeam
    $labelNorm = Normalize-Token $row.label
    $matchedPaths = @()

    foreach ($brand in $brandedInfo) {
        if ($brand.Date -ne $row.date) {
            continue
        }

        $matchesCount = 0
        if ($homeNorm -ne '' -and $brand.NormalizedName -like ('*' + $homeNorm + '*')) {
            $matchesCount++
        }
        if ($awayNorm -ne '' -and $brand.NormalizedName -like ('*' + $awayNorm + '*')) {
            $matchesCount++
        }
        if ($labelNorm -ne '' -and $brand.NormalizedName -like ('*' + $labelNorm + '*')) {
            $matchesCount++
        }

        if ($matchesCount -ge 2) {
            $matchedPaths += $brand.Path
            $brand.Matched = $true
        }
    }

    if ($matchedPaths.Count -gt 0) {
        $row.branded_matches = [string]::Join(';', $matchedPaths)
        $row.has_branded = 'true'
    }
    else {
        $row.branded_matches = ''
        $row.has_branded = 'false'
    }
}

$needsDetectTransform = @()
$needsTransformOnly = @()
foreach ($row in $rows) {
    if ($row.proxy_exists -eq 'true' -and $row.trf_exists -eq 'false' -and $row.stabilized_exists -eq 'false') {
        $needsDetectTransform += $row
    }
    elseif ($row.proxy_exists -eq 'true' -and $row.trf_exists -eq 'true' -and $row.stabilized_exists -eq 'false') {
        $needsTransformOnly += $row
    }
}

$brandedOnlyRows = @()
foreach ($brand in $brandedInfo) {
    if ($brand.Matched) {
        continue
    }

    $brandRow = [pscustomobject][ordered]@{
        clip_id = ''
        index = ''
        date = $brand.Date
        homeTeam = ''
        awayTeam = ''
        label = ''
        tStart = ''
        tEnd = ''
        duration_sec = ''
        orientation = 'unknown'
        hasDebug = $false
        hasFinal = $false
        root_dir = ''
        proxy_path = ''
        trf_path = ''
        stabilized_path = ''
        branded_matches = $brand.Path
        proxy_exists = 'false'
        trf_exists = 'false'
        stabilized_exists = 'false'
        has_branded = 'true'
        notes = 'branded-only evidence'
    }
    $brandedOnlyRows += $brandRow
}

$csvPath = if ([System.IO.Path]::IsPathRooted($OutCsv)) { $OutCsv } else { Join-Path $repoRoot $OutCsv }
$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

$jsonPath = if ([System.IO.Path]::IsPathRooted($OutJson)) { $OutJson } else { Join-Path $repoRoot $OutJson }
$rows | ConvertTo-Json -Depth 6 | Set-Content -Path $jsonPath -Encoding UTF8

$detectListPath = Join-Path $repoRoot 'buildlist_needs_detect_and_transform.csv'
$needsDetectTransform | Export-Csv -Path $detectListPath -NoTypeInformation -Encoding UTF8

$transformListPath = Join-Path $repoRoot 'buildlist_needs_transform_only.csv'
$needsTransformOnly | Export-Csv -Path $transformListPath -NoTypeInformation -Encoding UTF8

$renderOnlyPath = Join-Path $repoRoot 'buildlist_render_only.csv'
$brandedOnlyRows | Export-Csv -Path $renderOnlyPath -NoTypeInformation -Encoding UTF8

$totalCinematic = $rows.Count
$proxyCount = ($rows | Where-Object { $_.proxy_exists -eq 'true' }).Count
$trfCount = ($rows | Where-Object { $_.trf_exists -eq 'true' }).Count
$stabilizedCount = ($rows | Where-Object { $_.stabilized_exists -eq 'true' }).Count
$brandedCount = ($rows | Where-Object { $_.has_branded -eq 'true' }).Count
$brandedOnlyCount = $brandedOnlyRows.Count

Write-Host "Total cinematic folders parsed: $totalCinematic"
Write-Host "Proxy exists count: $proxyCount"
Write-Host "Transforms exists count: $trfCount"
Write-Host "Stabilized exists count: $stabilizedCount"
Write-Host "Has branded count: $brandedCount"
Write-Host "Branded-only evidence rows: $brandedOnlyCount"

if ($unparsedFolders.Count -gt 0) {
    Write-Host 'Unparsed folder names:'
    foreach ($name in $unparsedFolders | Sort-Object -Unique) {
        Write-Host " - $name"
    }
}
else {
    Write-Host 'All candidate folders parsed successfully.'
}
