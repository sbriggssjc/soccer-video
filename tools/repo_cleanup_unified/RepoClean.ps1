<#
.SYNOPSIS
  Unified repository cleanup, inventory, deduplication, and indexing tool for the soccer-video project.
.DESCRIPTION
  Provides multiple modes that inventory repository assets, plan cleanup operations, execute them with
  safety rails, and build lightweight indices. Designed to supersede scattered one-off scripts and provide
  a single maintained entry point for maintenance tasks.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Inventory','Plan','Execute','Index','Doctor','FindExistingTools')]
    [string]$Mode,

    [string]$Root = (Split-Path -Parent $PSScriptRoot),

    [switch]$Fast,
    [ValidateSet('SHA1','SHA256','MD5')]
    [string]$HashAlgo = 'SHA256',
    [int]$MaxDepth = 0,
    [switch]$IncludeHidden,
    [switch]$FollowJunctions,

    [ValidateSet('Quarantine','Delete')]
    [string]$Strategy = 'Quarantine',
    [switch]$EnableHardlink,
    [switch]$Permanent,

    [switch]$ConfirmRun,
    [switch]$DryRun = $true,
    [int]$Concurrent = 4
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-RepoRoot {
    param([string]$RootPath)
    if ([string]::IsNullOrWhiteSpace($RootPath)) {
        $RootPath = Get-Location
    }
    try {
        $resolved = Resolve-Path -LiteralPath $RootPath -ErrorAction Stop
        return $resolved.ProviderPath
    } catch {
        throw "Unable to resolve root path '$RootPath'."
    }
}

function Initialize-OutputPaths {
    param([string]$BaseRoot)
    $baseOut = Join-Path $BaseRoot 'out'
    $paths = @{
        Inventory = Join-Path $baseOut 'inventory'
        Plans      = Join-Path $baseOut 'plans'
        Logs       = Join-Path $baseOut 'logs'
        Index      = Join-Path $baseOut 'index'
    }
    foreach ($dir in $paths.Values) {
        if (-not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    return $paths
}

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet('INFO','WARN','ERROR','SUCCESS')]
        [string]$Level = 'INFO'
    )
    $prefix = "[$($Level.ToUpper())]"
    $timestamp = (Get-Date).ToString('u')
    Write-Host "$timestamp $prefix $Message"
}

function Import-KeepRules {
    param([string]$ScriptRoot)
    $rulesPath = Join-Path $ScriptRoot 'KeepRules.psd1'
    if (-not (Test-Path -LiteralPath $rulesPath)) {
        throw "Keep rules file not found at $rulesPath"
    }
    return Import-PowerShellDataFile -LiteralPath $rulesPath
}

function ConvertTo-RelativePath {
    param(
        [string]$Root,
        [string]$FullPath
    )
    $rootWithSep = [System.IO.Path]::GetFullPath($Root)
    $full = [System.IO.Path]::GetFullPath($FullPath)
    if ($full.StartsWith($rootWithSep, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relative = $full.Substring($rootWithSep.Length).TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
        if ([string]::IsNullOrEmpty($relative)) { return '.' }
        return $relative
    }
    return $full
}

function Get-WildcardSet {
    param([string[]]$Patterns)
    $set = @()
    foreach ($pattern in $Patterns) {
        if ([string]::IsNullOrWhiteSpace($pattern)) { continue }
        $set += New-Object System.Management.Automation.WildcardPattern($pattern, [System.Management.Automation.WildcardOptions]::IgnoreCase)
    }
    return $set
}

function Test-PatternListMatch {
    param(
        [System.Management.Automation.WildcardPattern[]]$Patterns,
        [string]$Path
    )
    if (-not $Patterns -or $Patterns.Count -eq 0) { return $false }
    foreach ($pattern in $Patterns) {
        if ($pattern.IsMatch($Path)) { return $true }
    }
    return $false
}

function Load-HashCache {
    param([string]$CachePath)
    $cache = @{}
    if (Test-Path -LiteralPath $CachePath) {
        try {
            $items = Import-Csv -LiteralPath $CachePath
            foreach ($item in $items) {
                $key = $item.Path
                $cache[$key] = $item
            }
        } catch {
            Write-Log "Failed to read hash cache: $($_.Exception.Message)" 'WARN'
        }
    }
    return $cache
}

function Save-HashCache {
    param(
        [System.Collections.Hashtable]$Cache,
        [string]$CachePath
    )
    $directory = Split-Path -Parent $CachePath
    if (-not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $Cache.GetEnumerator() | ForEach-Object {
        $_.Value
    } | Export-Csv -LiteralPath $CachePath -NoTypeInformation
}

function Get-FileEnumeration {
    param(
        [string]$RootPath,
        [int]$MaxDepth,
        [switch]$IncludeHidden,
        [switch]$FollowJunctions
    )
    $items = New-Object System.Collections.Generic.List[System.IO.FileInfo]
    $queue = New-Object System.Collections.Generic.Queue[psobject]
    $queue.Enqueue([PSCustomObject]@{ Path = $RootPath; Depth = 0 })
    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        try {
            $children = Get-ChildItem -LiteralPath $current.Path -Force:$IncludeHidden -ErrorAction Stop
        } catch {
            Write-Log "Failed to enumerate $($current.Path): $($_.Exception.Message)" 'WARN'
            continue
        }
        foreach ($child in $children) {
            if ($child.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                if (-not $FollowJunctions) { continue }
            }
            if ($child.PSIsContainer) {
                if ($MaxDepth -gt 0 -and $current.Depth -ge $MaxDepth) { continue }
                $queue.Enqueue([PSCustomObject]@{ Path = $child.FullName; Depth = $current.Depth + 1 })
            } else {
                if (-not $IncludeHidden -and ($child.Attributes -band [System.IO.FileAttributes]::Hidden)) { continue }
                $items.Add($child)
            }
        }
    }
    return $items
}

function Invoke-Ffprobe {
    param(
        [string]$FfprobePath,
        [string]$TargetPath
    )
    if (-not $FfprobePath) { return $null }
    $psi = @('-v','error','-select_streams','v:0','-show_entries','stream=width,height','-show_entries','format=duration','-of','json',$TargetPath)
    try {
        $result = & $FfprobePath @psi 2>$null
        if (-not $result) { return $null }
        $parsed = $result | ConvertFrom-Json -ErrorAction Stop
        $duration = [double]::Parse($parsed.format.duration, [System.Globalization.CultureInfo]::InvariantCulture)
        $stream = $parsed.streams | Where-Object { $_.width -and $_.height } | Select-Object -First 1
        return [PSCustomObject]@{
            Duration = $duration
            Width    = $stream.width
            Height   = $stream.height
        }
    } catch {
        return $null
    }
}

function Get-FirstLine {
    param([string]$Path)
    try {
        return (Get-Content -LiteralPath $Path -TotalCount 1 -ErrorAction Stop)
    } catch { return '' }
}

function Get-PurposeGuess {
    param([string]$Path)
    try {
        $lines = Get-Content -LiteralPath $Path -TotalCount 8 -ErrorAction Stop
    } catch { return '' }
    foreach ($line in $lines) {
        $trim = $line.Trim()
        if ([string]::IsNullOrEmpty($trim)) { continue }
        if ($trim.StartsWith('#') -or $trim.StartsWith('//')) { return $trim }
        if ($trim.StartsWith('<#')) { return $trim }
        if ($trim.StartsWith('"""')) { return $trim }
        if ($trim.StartsWith("'''")) { return $trim }
    }
    return ''
}

function Get-FunctionNames {
    param([string]$Path)
    $extension = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
    $pattern = switch ($extension) {
        '.ps1' { 'function\s+([A-Za-z0-9_-]+)' }
        '.psm1' { 'function\s+([A-Za-z0-9_-]+)' }
        '.py' { 'def\s+([A-Za-z0-9_]+)\s*\(' }
        '.sh' { '^([A-Za-z0-9_]+)\s*\(\)\s*\{' }
        '.bash' { '^([A-Za-z0-9_]+)\s*\(\)\s*\{' }
        default { $null }
    }
    if (-not $pattern) { return '' }
    try {
        $matches = Select-String -LiteralPath $Path -Pattern $pattern -AllMatches -ErrorAction Stop
        $names = @()
        foreach ($match in $matches) {
            foreach ($value in $match.Matches.Value) {
                $name = $value -replace 'function','' -replace 'def','' -replace '\(\)',''
                $name = $name.Trim(' ','{','(','\t')
                if (-not [string]::IsNullOrWhiteSpace($name)) {
                    $names += $name
                }
            }
        }
        $names = $names | Sort-Object -Unique
        return ($names -join '; ')
    } catch { return '' }
}

function Map-ScriptToMode {
    param([string]$Name)
    $lower = $Name.ToLowerInvariant()
    if ($lower -match 'doctor|diagnostic|env') { return 'Doctor' }
    if ($lower -match 'inventory|scan|audit') { return 'Inventory' }
    if ($lower -match 'plan|map') { return 'Plan' }
    if ($lower -match 'execute|cleanup|clean|remove|delete|quarantine') { return 'Execute' }
    if ($lower -match 'index|build|manifest|season') { return 'Index' }
    if ($lower -match 'dedupe|dupe|hardlink') { return 'Plan (Hardlink)' }
    return 'Inventory'
}

function Find-ExistingTools {
    param(
        [string]$RootPath,
        [string]$InventoryDir,
        [string]$DocsDir
    )
    $keywords = @('cleanup','dedupe','index','inventory','scan','atomic','portrait','reel','trash','quarantine','stabilize','upscale','esrgan','vidstab','follow','ball','brand','opener','postable','coach','build')
    $extensions = @('.ps1','.psm1','.py','.sh','.bash')
    $files = Get-FileEnumeration -RootPath $RootPath -MaxDepth 0 -IncludeHidden -FollowJunctions:$false
    $matches = @()
    foreach ($file in $files) {
        $ext = [System.IO.Path]::GetExtension($file.FullName).ToLowerInvariant()
        if (-not ($extensions -contains $ext)) { continue }
        $name = $file.Name.ToLowerInvariant()
        $hit = $false
        foreach ($keyword in $keywords) {
            if ($name -like "*${keyword}*") { $hit = $true; break }
        }
        if (-not $hit) { continue }
        $rel = ConvertTo-RelativePath -Root $RootPath -FullPath $file.FullName
        $matches += [PSCustomObject]@{
            RelativePath  = $rel
            FullPath      = $file.FullName
            SizeBytes     = $file.Length
            LastWriteTime = $file.LastWriteTime
            FirstLine     = Get-FirstLine -Path $file.FullName
            Functions     = Get-FunctionNames -Path $file.FullName
            PurposeGuess  = Get-PurposeGuess -Path $file.FullName
            SuggestedMode = Map-ScriptToMode -Name $file.Name
        }
    }
    $outputPath = Join-Path $InventoryDir 'existing_tools.csv'
    $matches | Sort-Object RelativePath | Export-Csv -LiteralPath $outputPath -NoTypeInformation
    if ($matches.Count -gt 0) {
        $table = $matches | Select-Object RelativePath, SuggestedMode, SizeBytes, LastWriteTime | Format-Table -AutoSize | Out-String
        Write-Host $table
    } else {
        Write-Log 'No existing scripts matched the discovery criteria.' 'INFO'
    }
    $compatPath = Join-Path $DocsDir 'SCRIPT_COMPAT_MATRIX.md'
    $lines = @('# Script Compatibility Matrix', '', '| Old Script | New Subcommand |', '| --- | --- |')
    foreach ($match in ($matches | Sort-Object RelativePath)) {
        $lines += "| $($match.RelativePath) | $($match.SuggestedMode) |"
    }
    Set-Content -LiteralPath $compatPath -Value $lines
    return $matches
}

function Get-InventoryRecords {
    param(
        [System.IO.FileInfo[]]$Files,
        [string]$RootPath,
        [switch]$Fast,
        [string]$HashAlgo,
        [System.Management.Automation.WildcardPattern[]]$KeepPatterns,
        [System.Management.Automation.WildcardPattern[]]$RemovePatterns,
        [string[]]$SidecarExts,
        [hashtable]$HashCache,
        [string]$HashCachePath
    )
    $ffprobeCmd = Get-Command -Name 'ffprobe' -ErrorAction SilentlyContinue
    if ($ffprobeCmd) {
        Write-Log "ffprobe detected at $($ffprobeCmd.Source)" 'INFO'
    } else {
        Write-Log 'ffprobe not available; duration/resolution metadata will be skipped.' 'WARN'
    }
    $records = @()
    $mediaExtensions = @('.mp4','.mov','.mkv','.avi','.m4v','.mpg','.mpeg')
    $durationLookup = @{}
    foreach ($file in $Files) {
        $relative = ConvertTo-RelativePath -Root $RootPath -FullPath $file.FullName
        $extension = $file.Extension.ToLowerInvariant()
        $duration = $null
        $width = $null
        $height = $null
        if ($ffprobeCmd -and $mediaExtensions -contains $extension) {
            $meta = Invoke-Ffprobe -FfprobePath $ffprobeCmd.Source -TargetPath $file.FullName
            if ($meta) {
                $duration = [Math]::Round($meta.Duration, 3)
                $width = $meta.Width
                $height = $meta.Height
            }
        }
        if ($duration) { $durationLookup[$file.FullName] = $duration }
        $hash = $null
        $cacheKey = ConvertTo-RelativePath -Root $RootPath -FullPath $file.FullName
        if ($HashCache.ContainsKey($cacheKey)) {
            $entry = $HashCache[$cacheKey]
            if ([double]$entry.SizeBytes -eq $file.Length -and [datetime]$entry.LastWriteTime -eq $file.LastWriteTime) {
                $hash = $entry.Hash
            }
        }
        $needsHash = $true
        if ($Fast) { $needsHash = $false }
        if ($Fast -and $duration) {
            # Hash later only for probable duplicates
            $needsHash = $false
        }
        if (-not $needsHash) {
            # Keep placeholder; may compute later in dedupe pass
            $hash = $hash
        } else {
            if (-not $hash) {
                try {
                    $hashObj = Get-FileHash -Algorithm $HashAlgo -LiteralPath $file.FullName
                    $hash = $hashObj.Hash
                } catch {
                    Write-Log "Failed to hash $relative: $($_.Exception.Message)" 'WARN'
                }
            }
        }
        $status = 'ORPHAN'
        $relLower = $relative.Replace([System.IO.Path]::AltDirectorySeparatorChar, [System.IO.Path]::DirectorySeparatorChar)
        if (Test-PatternListMatch -Patterns $KeepPatterns -Path $relLower) {
            $status = 'KEEP'
        } elseif (Test-PatternListMatch -Patterns $RemovePatterns -Path $relLower) {
            $status = 'CANDIDATE_REMOVE'
        }
        if ($status -eq 'KEEP' -and $SidecarExts -contains $extension.ToLowerInvariant()) {
            $status = 'KEEP_SIDECAR'
        }
        $records += [PSCustomObject]@{
            RelativePath  = $relative
            FullPath      = $file.FullName
            SizeBytes     = $file.Length
            LastWriteTime = $file.LastWriteTime
            Extension     = $extension
            Duration      = $duration
            Width         = $width
            Height        = $height
            Hash          = $hash
            Status        = $status
        }
        if ($hash) {
            $HashCache[$cacheKey] = [PSCustomObject]@{
                Path          = $cacheKey
                SizeBytes     = $file.Length
                LastWriteTime = $file.LastWriteTime
                Hash          = $hash
            }
        }
    }
    return $records
}

function Enhance-HashesForFastMode {
    param(
        [System.Collections.IEnumerable]$Records,
        [string]$RootPath,
        [string]$HashAlgo,
        [hashtable]$HashCache
    )
    $buckets = @{}
    foreach ($record in $Records) {
        $sizeBucket = [Math]::Round($record.SizeBytes / 16)
        $durationBucket = if ($record.Duration) { [Math]::Round($record.Duration,1) } else { 'na' }
        $key = "S${sizeBucket}-D${durationBucket}"
        if (-not $buckets.ContainsKey($key)) {
            $buckets[$key] = New-Object System.Collections.Generic.List[psobject]
        }
        $buckets[$key].Add($record)
    }
    foreach ($bucket in $buckets.GetEnumerator()) {
        if ($bucket.Value.Count -le 1) { continue }
        foreach ($record in $bucket.Value) {
            if ([string]::IsNullOrWhiteSpace($record.Hash)) {
                $cacheKey = $record.RelativePath
                if ($HashCache.ContainsKey($cacheKey)) {
                    $record.Hash = $HashCache[$cacheKey].Hash
                    continue
                }
                try {
                    $hashObj = Get-FileHash -Algorithm $HashAlgo -LiteralPath $record.FullPath
                    $record.Hash = $hashObj.Hash
                    $HashCache[$cacheKey] = [PSCustomObject]@{
                        Path          = $cacheKey
                        SizeBytes     = $record.SizeBytes
                        LastWriteTime = $record.LastWriteTime
                        Hash          = $record.Hash
                    }
                } catch {
                    Write-Log "Fast hash upgrade failed for $($record.RelativePath): $($_.Exception.Message)" 'WARN'
                }
            }
        }
    }
}

function Export-InventoryOutputs {
    param(
        [System.Collections.Generic.List[object]]$Records,
        [string]$InventoryDir
    )
    $jsonPath = Join-Path $InventoryDir 'repo_inventory.json'
    $csvPath = Join-Path $InventoryDir 'repo_inventory.csv'
    $Records | Export-Csv -LiteralPath $csvPath -NoTypeInformation
    $Records | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $jsonPath

    $hashGroups = $Records | Where-Object { $_.Hash } | Group-Object Hash | Where-Object { $_.Count -gt 1 }
    $dupeExactPath = Join-Path $InventoryDir 'duplicates_exact.csv'
    $exactRows = @()
    foreach ($group in $hashGroups) {
        foreach ($item in $group.Group) {
            $exactRows += [PSCustomObject]@{
                Hash         = $group.Name
                RelativePath = $item.RelativePath
                SizeBytes    = $item.SizeBytes
                Duration     = $item.Duration
            }
        }
    }
    $exactRows | Export-Csv -LiteralPath $dupeExactPath -NoTypeInformation

    $probablePath = Join-Path $InventoryDir 'duplicates_probable.csv'
    $probableRows = @()
    $Records | Where-Object { $_.Duration -and $_.Width -and $_.Height } |
        Group-Object -Property { "{0}-{1}-{2}" -f [Math]::Round($_.Duration,1), [Math]::Round($_.Width/10)*10, [Math]::Round($_.Height/10)*10 } |
        Where-Object { $_.Count -gt 1 } |
        ForEach-Object {
            $keyParts = $_.Name -split '-'
            foreach ($item in $_.Group) {
                $probableRows += [PSCustomObject]@{
                    DurationApprox = $keyParts[0]
                    WidthApprox    = $keyParts[1]
                    HeightApprox   = $keyParts[2]
                    RelativePath   = $item.RelativePath
                    Hash           = $item.Hash
                    SizeBytes      = $item.SizeBytes
                }
            }
        }
    $probableRows | Export-Csv -LiteralPath $probablePath -NoTypeInformation

    $keepPath = Join-Path $InventoryDir 'keep_candidates.csv'
    $removePath = Join-Path $InventoryDir 'remove_candidates.csv'
    $orphansPath = Join-Path $InventoryDir 'orphans.csv'
    ($Records | Where-Object { $_.Status -like 'KEEP*' }) | Export-Csv -LiteralPath $keepPath -NoTypeInformation
    ($Records | Where-Object { $_.Status -eq 'CANDIDATE_REMOVE' }) | Export-Csv -LiteralPath $removePath -NoTypeInformation
    ($Records | Where-Object { $_.Status -eq 'ORPHAN' }) | Export-Csv -LiteralPath $orphansPath -NoTypeInformation

    $summaryPath = Join-Path $InventoryDir 'summary_sizes_by_type.csv'
    $Records |
        Group-Object -Property Extension |
        ForEach-Object {
            [PSCustomObject]@{
                Extension = $_.Name
                Count     = $_.Count
                TotalBytes = ($_.Group | Measure-Object -Property SizeBytes -Sum).Sum
            }
        } |
        Sort-Object -Property TotalBytes -Descending |
        Export-Csv -LiteralPath $summaryPath -NoTypeInformation
}

function Build-MappingFiles {
    param(
        [System.Collections.Generic.List[object]]$Records,
        [string]$InventoryDir
    )
    $masters = $Records | Where-Object { $_.RelativePath -match 'master' -or $_.RelativePath -match 'Game ' }
    $atomics = $Records | Where-Object { $_.RelativePath -match 'atomic' }
    $reels   = $Records | Where-Object { $_.RelativePath -match 'reel' -or $_.RelativePath -match 'postable' }

    $atomicMap = @()
    foreach ($atomic in $atomics) {
        $base = [System.IO.Path]::GetFileNameWithoutExtension($atomic.RelativePath)
        $candidates = $masters | Where-Object { $_.RelativePath -like "*$base*" } | Select-Object -First 5
        $atomicMap += [PSCustomObject]@{
            AtomicClip    = $atomic.RelativePath
            CandidateMasters = ($candidates.RelativePath -join '; ')
        }
    }
    $atomicPath = Join-Path $InventoryDir 'map_atomic_to_master.csv'
    $atomicMap | Export-Csv -LiteralPath $atomicPath -NoTypeInformation

    $reelMap = @()
    foreach ($reel in $reels) {
        $base = [System.IO.Path]::GetFileNameWithoutExtension($reel.RelativePath)
        $candidates = $atomics | Where-Object { $_.RelativePath -like "*$base*" } | Select-Object -First 10
        $reelMap += [PSCustomObject]@{
            ReelPath      = $reel.RelativePath
            CandidateAtomics = ($candidates.RelativePath -join '; ')
        }
    }
    $reelPath = Join-Path $InventoryDir 'map_reel_to_atomic.csv'
    $reelMap | Export-Csv -LiteralPath $reelPath -NoTypeInformation
}

function New-CleanupPlan {
    param(
        [string]$RootPath,
        [string]$InventoryDir,
        [string]$PlansDir,
        [string]$Strategy,
        [switch]$EnableHardlink,
        [switch]$Permanent
    )
    $inventoryFile = Join-Path $InventoryDir 'repo_inventory.csv'
    if (-not (Test-Path -LiteralPath $inventoryFile)) {
        throw "Inventory file not found. Run -Mode Inventory first."
    }
    $records = Import-Csv -LiteralPath $inventoryFile
    $plan = @()
    $hardlinkPlan = @()
    $quarantineRoot = Join-Path $RootPath '_quarantine'
    $planIndex = 0
    foreach ($item in $records) {
        if ($item.Status -like 'KEEP*') { continue }
        $action = 'Skip'
        $reason = ''
        if ($item.Status -eq 'CANDIDATE_REMOVE') {
            $action = if ($Strategy -eq 'Delete') { if ($Permanent) { 'DeletePermanent' } else { 'DeleteRecycle' } } else { 'Quarantine' }
            $reason = 'Matches remove glob'
        } elseif ($item.Status -eq 'ORPHAN') {
            $action = if ($Strategy -eq 'Delete') { if ($Permanent) { 'DeletePermanent' } else { 'DeleteRecycle' } } else { 'Quarantine' }
            $reason = 'Not covered by keep rules'
        }
        if ($item.Hash) {
            $duplicates = $records | Where-Object { $_.Hash -eq $item.Hash }
            if ($duplicates.Count -gt 1) {
                $canonical = $duplicates | Sort-Object -Property RelativePath | Select-Object -First 1
                if ($canonical.RelativePath -ne $item.RelativePath) {
                    if ($EnableHardlink) {
                        $hardlinkPlan += [PSCustomObject]@{
                            Source = $item.FullPath
                            Target = $canonical.FullPath
                            Action = 'ReplaceWithHardlink'
                            BytesSaved = [long]$item.SizeBytes
                        }
                    }
                    if ($action -eq 'Skip') {
                        $action = 'Quarantine'
                        $reason = 'Duplicate of ' + $canonical.RelativePath
                    }
                }
            }
        }
        if ($action -eq 'Skip') { continue }
        $targetPath = $null
        if ($action -eq 'Quarantine') {
            $rel = $item.RelativePath
            $targetPath = Join-Path $quarantineRoot $rel
        }
        $plan += [PSCustomObject]@{
            PlanIndex    = $planIndex
            RelativePath = $item.RelativePath
            FullPath     = $item.FullPath
            Action       = $action
            Target       = $targetPath
            Reason       = $reason
            BytesSaved   = [long]$item.SizeBytes
            LastWriteTime = $item.LastWriteTime
            PlanRoot      = $RootPath
        }
        $planIndex++
    }
    $planPath = Join-Path $PlansDir 'cleanup_plan.csv'
    $plan | Export-Csv -LiteralPath $planPath -NoTypeInformation
    if ($EnableHardlink -and $hardlinkPlan.Count -gt 0) {
        $hardlinkPath = Join-Path $PlansDir 'hardlink_plan.csv'
        $hardlinkPlan | Export-Csv -LiteralPath $hardlinkPath -NoTypeInformation
    }
    $summaryText = @()
    $summaryText += "Cleanup Plan Summary"
    $summaryText += "Strategy: $Strategy"
    $summaryText += "Items planned: $($plan.Count)"
    $summaryText += ''
    if ($plan.Count -gt 0) {
        $summaryText += 'Top space savings by folder:'
        $plan |
            Group-Object -Property { Split-Path $_.RelativePath -Parent } |
            ForEach-Object {
                [PSCustomObject]@{
                    Folder = $_.Name
                    Bytes  = ($_.Group | Measure-Object -Property BytesSaved -Sum).Sum
                }
            } |
            Sort-Object -Property Bytes -Descending |
            Select-Object -First 10 |
            ForEach-Object {
                $summaryText += " - $($_.Folder): {0:N2} GB" -f ($_.Bytes / 1GB)
            }
    } else {
        $summaryText += 'No filesystem actions proposed.'
    }
    $summaryPath = Join-Path $PlansDir 'human_readable_summary.txt'
    Set-Content -LiteralPath $summaryPath -Value $summaryText
    Write-Host ($summaryText -join [Environment]::NewLine)
    return $planPath
}

function Invoke-CleanupExecution {
    param(
        [string]$PlanPath,
        [string]$RootPath,
        [string]$LogsDir,
        [switch]$DryRun,
        [switch]$ConfirmRun,
        [int]$Concurrent
    )
    if ($DryRun -or -not $ConfirmRun) {
        throw 'Execution refused: pass both -ConfirmRun and -DryRun:$false to proceed.'
    }
    if (-not (Test-Path -LiteralPath $PlanPath)) {
        throw "Plan file not found at $PlanPath"
    }
    $planInfo = Get-Item -LiteralPath $PlanPath
    if ((Get-Date) - $planInfo.LastWriteTime -gt [TimeSpan]::FromHours(48)) {
        throw 'Plan is older than 48 hours. Generate a fresh plan before executing.'
    }
    $planRaw = Import-Csv -LiteralPath $PlanPath
    if ($planRaw.Count -eq 0) {
        Write-Log 'Plan is empty; nothing to execute.' 'INFO'
        return
    }
    $resolvedRoot = [System.IO.Path]::GetFullPath($RootPath)
    $planRoot = $planRaw[0].PlanRoot
    if ($planRoot) {
        $resolvedPlanRoot = [System.IO.Path]::GetFullPath($planRoot)
        if ($resolvedRoot -ne $resolvedPlanRoot) {
            throw "Plan root '$resolvedPlanRoot' does not match execution root '$resolvedRoot'."
        }
    }
    $plan = @()
    foreach ($row in $planRaw) {
        $lastWrite = $null
        if ($row.LastWriteTime) {
            try { $lastWrite = [datetime]::Parse($row.LastWriteTime) } catch { $lastWrite = $null }
        }
        $plan += [PSCustomObject]@{
            PlanIndex    = [int]$row.PlanIndex
            RelativePath = $row.RelativePath
            FullPath     = $row.FullPath
            Action       = $row.Action
            Target       = $row.Target
            Reason       = $row.Reason
            BytesSaved   = [long]$row.BytesSaved
            LastWriteTime = $lastWrite
        }
    }
    $logPath = Join-Path $LogsDir ("repo_cleanup_run_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
    $failPath = Join-Path $LogsDir 'failed_actions.csv'
    $failed = New-Object System.Collections.Generic.List[psobject]
    $totalBytes = [long]0
    $isWindows = $env:OS -like '*Windows*' -or $PSVersionTable.Platform -eq 'Win32NT'
    $executeBlock = {
        param($item,$isWindows)
        try {
            switch ($item.Action) {
                'Quarantine' {
                    $targetDir = Split-Path -Parent $item.Target
                    if (-not (Test-Path -LiteralPath $targetDir)) {
                        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
                    }
                    Move-Item -LiteralPath $item.FullPath -Destination $item.Target -Force
                    if ($item.LastWriteTime) {
                        [System.IO.File]::SetLastWriteTime($item.Target, $item.LastWriteTime)
                    }
                }
                'DeleteRecycle' {
                    if ($isWindows) {
                        try {
                            $shell = New-Object -ComObject Shell.Application
                            $shell.Namespace(10).MoveHere($item.FullPath)
                        } catch {
                            Remove-Item -LiteralPath $item.FullPath -Force
                        }
                    } else {
                        Remove-Item -LiteralPath $item.FullPath -Force
                    }
                }
                'DeletePermanent' {
                    Remove-Item -LiteralPath $item.FullPath -Force
                }
            }
            return [PSCustomObject]@{ Success = $true; Bytes = [long]$item.BytesSaved; PlanIndex = $item.PlanIndex }
        } catch {
            return [PSCustomObject]@{ Success = $false; Error = $_.Exception.Message; PlanIndex = $item.PlanIndex }
        }
    }
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $results = $plan | ForEach-Object -Parallel {
            param($entry,$execute,$isWin)
            & $execute $entry $isWin
        } -ArgumentList $executeBlock,$isWindows -ThrottleLimit $Concurrent
    } else {
        $results = @()
        foreach ($entry in $plan) {
            $results += & $executeBlock $entry $isWindows
        }
    }
    $resultLookup = @{}
    foreach ($res in $results) {
        $resultLookup[[int]$res.PlanIndex] = $res
    }
    foreach ($entry in $plan) {
        $res = $resultLookup[[int]$entry.PlanIndex]
        if ($res -and $res.Success) {
            $totalBytes += [long]$entry.BytesSaved
        } elseif ($res) {
            $failed.Add([PSCustomObject]@{
                RelativePath = $entry.RelativePath
                Action       = $entry.Action
                Error        = $res.Error
            })
        } else {
            $failed.Add([PSCustomObject]@{
                RelativePath = $entry.RelativePath
                Action       = $entry.Action
                Error        = 'Execution result missing.'
            })
        }
    }
    $summary = "Reclaimed space: {0:N2} GB" -f ($totalBytes / 1GB)
    $summary | Tee-Object -FilePath $logPath
    if ($failed.Count -gt 0) {
        $failed | Export-Csv -LiteralPath $failPath -NoTypeInformation
    }
}

function Build-SeasonIndex {
    param(
        [string]$InventoryDir,
        [string]$IndexDir
    )
    $inventoryFile = Join-Path $InventoryDir 'repo_inventory.csv'
    if (-not (Test-Path -LiteralPath $inventoryFile)) {
        throw 'Inventory not found. Run -Mode Inventory first.'
    }
    $records = Import-Csv -LiteralPath $inventoryFile
    $kept = $records | Where-Object { $_.Status -like 'KEEP*' }
    $index = @()
    $groups = $kept | Group-Object -Property { ($_.RelativePath -split [System.IO.Path]::DirectorySeparatorChar)[0] }
    foreach ($group in $groups) {
        $atomics = $group.Group | Where-Object { $_.RelativePath -match 'atomic' }
        $reels = $group.Group | Where-Object { $_.RelativePath -match 'reel' -or $_.RelativePath -match 'postable' }
        $masters = $group.Group | Where-Object { $_.RelativePath -match 'master' -or $_.RelativePath -match 'Game ' }
        $durationTotal = ($atomics | Measure-Object -Property Duration -Sum).Sum
        $index += [PSCustomObject]@{
            Bucket            = $group.Name
            AtomicClipCount   = $atomics.Count
            AtomicDurationSec = [Math]::Round(($durationTotal ? $durationTotal : 0),2)
            MasterCount       = $masters.Count
            ReelCount         = $reels.Count
            MissingReel       = if ($atomics.Count -gt 0 -and $reels.Count -eq 0) { $true } else { $false }
        }
    }
    $csvPath = Join-Path $IndexDir 'season_index.csv'
    $jsonPath = Join-Path $IndexDir 'season_index.json'
    $index | Export-Csv -LiteralPath $csvPath -NoTypeInformation
    $index | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $jsonPath
}

function Run-Doctor {
    param(
        [string]$RootPath,
        [string]$InventoryDir
    )
    $report = @()
    $report += "RepoClean Doctor Report"
    $report += "Generated: $(Get-Date -Format 'u')"
    $report += ''
    $ffprobe = Get-Command -Name 'ffprobe' -ErrorAction SilentlyContinue
    $report += if ($ffprobe) { "ffprobe detected at $($ffprobe.Source)" } else { 'ffprobe not detected (media metadata will be skipped).' }
    $report += "PowerShell version: $($PSVersionTable.PSVersion)"
    $report += if ($PSVersionTable.PSVersion.Major -ge 7) { 'Parallel features enabled.' } else { 'Running on Windows PowerShell; parallel inventory will be sequential.' }
    $isNtfs = ([System.IO.DriveInfo]([System.IO.Path]::GetPathRoot($RootPath))).DriveFormat -eq 'NTFS'
    $report += if ($isNtfs) { 'NTFS detected; hardlink dedupe available.' } else { 'Non-NTFS filesystem; hardlink dedupe unavailable.' }
    try {
        $longPathEnabled = Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -ErrorAction Stop
        if ($longPathEnabled.LongPathsEnabled -eq 1) {
            $report += 'Long path support is enabled.'
        } else {
            $report += 'Long path support disabled. Enable Windows 10+ long paths for best results.'
        }
    } catch {
        $report += 'Unable to determine long path support. Run as administrator to confirm settings.'
    }
    $reportPath = Join-Path $InventoryDir 'environment_report.txt'
    Set-Content -LiteralPath $reportPath -Value $report
    $report | ForEach-Object { Write-Host $_ }
}

$repoRoot = Get-RepoRoot -RootPath $Root
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputPaths = Initialize-OutputPaths -BaseRoot $repoRoot
$docsDir = Join-Path $scriptRoot 'docs'
$rules = Import-KeepRules -ScriptRoot $scriptRoot
$keepPatterns = Get-WildcardSet -Patterns $rules.KeepGlobs
$removePatterns = Get-WildcardSet -Patterns $rules.RemoveGlobs
$hashCachePath = Join-Path $outputPaths.Inventory 'hash_cache.csv'
$hashCache = Load-HashCache -CachePath $hashCachePath

switch ($Mode) {
    'FindExistingTools' {
        Find-ExistingTools -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -DocsDir $docsDir | Out-Null
    }
    'Inventory' {
        $files = Get-FileEnumeration -RootPath $repoRoot -MaxDepth:$MaxDepth -IncludeHidden:$IncludeHidden -FollowJunctions:$FollowJunctions
        Write-Log "Enumerated $($files.Count) files" 'INFO'
        $records = Get-InventoryRecords -Files $files -RootPath $repoRoot -Fast:$Fast -HashAlgo $HashAlgo -KeepPatterns $keepPatterns -RemovePatterns $removePatterns -SidecarExts $rules.SidecarExts -HashCache $hashCache -HashCachePath $hashCachePath
        if ($Fast) {
            Enhance-HashesForFastMode -Records $records -RootPath $repoRoot -HashAlgo $HashAlgo -HashCache $hashCache
        }
        Export-InventoryOutputs -Records $records -InventoryDir $outputPaths.Inventory
        Build-MappingFiles -Records $records -InventoryDir $outputPaths.Inventory
        Save-HashCache -Cache $hashCache -CachePath $hashCachePath
        Find-ExistingTools -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -DocsDir $docsDir | Out-Null
        Write-Log 'Inventory complete.' 'SUCCESS'
    }
    'Plan' {
        $planPath = New-CleanupPlan -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -PlansDir $outputPaths.Plans -Strategy $Strategy -EnableHardlink:$EnableHardlink -Permanent:$Permanent
        Write-Log "Plan written to $planPath" 'SUCCESS'
    }
    'Execute' {
        $planPath = Join-Path $outputPaths.Plans 'cleanup_plan.csv'
        Invoke-CleanupExecution -PlanPath $planPath -RootPath $repoRoot -LogsDir $outputPaths.Logs -DryRun:$DryRun -ConfirmRun:$ConfirmRun -Concurrent $Concurrent
        Write-Log 'Execution complete.' 'SUCCESS'
    }
    'Index' {
        Build-SeasonIndex -InventoryDir $outputPaths.Inventory -IndexDir $outputPaths.Index
        Write-Log 'Index refreshed.' 'SUCCESS'
    }
    'Doctor' {
        Run-Doctor -RootPath $repoRoot -InventoryDir $outputPaths.Inventory
        Find-ExistingTools -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -DocsDir $docsDir | Out-Null
    }
}
