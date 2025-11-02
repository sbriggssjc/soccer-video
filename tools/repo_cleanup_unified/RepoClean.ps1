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
    [ValidateSet('Inventory','Plan','Execute','Index','Doctor','FindExistingTools')]
    [string]$Mode = 'Inventory',

    [string]$Root = (Split-Path -Parent $PSScriptRoot),

    [switch]$ConfirmRun,
    [switch]$DryRun,
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

    [int]$Concurrent = 4
)

function New-RecordList {
    return [System.Collections.Generic.List[psobject]]::new()
}

function To-RecordList {
    param([System.Collections.IEnumerable]$Items)
    $list = New-RecordList
    foreach ($it in $Items) { [void]$list.Add($it) }
    return $list
}

function Set-Prop {
    param(
        [Parameter(Mandatory)]$Obj,
        [Parameter(Mandatory)][string]$Name,
        $Value
    )
    $p = $Obj.PSObject.Properties[$Name]
    if ($null -eq $p) {
        $Obj | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    } else {
        $Obj.$Name = $Value
    }
}

function New-InventoryRecord {
    param(
        [string]$RelativePath = '',
        [string]$FullPath     = '',
        [nullable[int64]]$SizeBytes = $null,
        [datetime]$LastWriteTime = [datetime]::MinValue
    )
    return [PSCustomObject]@{
        RelativePath  = $RelativePath
        FullPath      = $FullPath
        SizeBytes     = $SizeBytes
        LastWriteTime = $LastWriteTime
        Extension     = ''
        Codec         = $null
        Width         = $null
        Height        = $null
        Hash          = $null
        Status        = $null
        Notes         = $null
    }
}

function Get-RelPath([string]$Base,[string]$Path) {
    try {
        $b = (Resolve-Path -LiteralPath $Base).Path
        $p = (Resolve-Path -LiteralPath $Path).Path
    } catch { return $Path }
    if ($p.StartsWith($b,[System.StringComparison]::OrdinalIgnoreCase)) {
        return $p.Substring($b.Length).TrimStart('\\','/')
    }
    return $p
}

function Run-Inventory {
    param([Parameter(Mandatory)][string]$Root, [switch]$Fast)

    $files = Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue
    Write-Host ("{0} [INFO] Inventory seeded with {1} rows" -f (Get-Date -Format u), $files.Count)

    # Use a growable list; avoid .Add() on fixed-size arrays
    $rows = New-RecordList

    foreach ($f in $files) {
        $ext = [System.IO.Path]::GetExtension($f.FullName)
        if ([string]::IsNullOrWhiteSpace($ext)) { $ext = '' }
        $ext = $ext.ToLowerInvariant()

        $row = [pscustomobject]@{
            FullPath     = $f.FullName
            RelativePath = Get-RelPath $Root $f.FullName
            SizeBytes    = [int64]$f.Length
            Ext          = $ext
        }
        # Maintain compatibility with any downstream code expecting 'Extension'
        Set-Prop $row 'Extension' $ext

        [void]$rows.Add($row)
    }

    $outDir = Join-Path $Root 'out\inventory'
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $csv = Join-Path $outDir 'inventory.csv'
    $rows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $csv

    return $csv
}

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $PSBoundParameters.ContainsKey('DryRun')) {
    $DryRun = [System.Management.Automation.SwitchParameter]::new($true)
}

if ($Mode -eq 'Execute') {
    if (-not $ConfirmRun) {
        throw "Refusing to act: pass -ConfirmRun"
    }
    if ($DryRun) {
        throw "Refusing to act: you left -DryRun on (default). Use -DryRun:`$false."
    }
}

function Get-RelativePath {
    param(
        [string]$Root,
        [string]$FullPath
    )
    if (-not $FullPath) { return $null }
    $resolvedRoot = if ([string]::IsNullOrWhiteSpace($Root)) { $null } else { [System.IO.Path]::GetFullPath($Root) }
    $resolvedFull = [System.IO.Path]::GetFullPath($FullPath)
    if ($resolvedRoot) {
        $trimmedRoot = $resolvedRoot.TrimEnd('\','/')
        $candidates = @(
            $trimmedRoot + [System.IO.Path]::DirectorySeparatorChar,
            $trimmedRoot + [System.IO.Path]::AltDirectorySeparatorChar
        ) | Where-Object { -not [string]::IsNullOrEmpty($_) }
        foreach ($rootNorm in ($candidates | Select-Object -Unique)) {
            if ($resolvedFull.StartsWith($rootNorm, [System.StringComparison]::OrdinalIgnoreCase)) {
                $relative = $resolvedFull.Substring($rootNorm.Length)
                $relative = $relative.TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
                if ([string]::IsNullOrEmpty($relative)) { return '.' }
                return $relative
            }
        }
    }
    return $resolvedFull
}

function New-FileRow {
    param(
        [Parameter(Mandatory)] [string]$Root,
        [Parameter(Mandatory)] [System.IO.FileInfo]$File
    )
    $row = [pscustomobject]@{
        FullPath     = $File.FullName
        RelativePath = Get-RelPath $Root $File.FullName
        SizeBytes    = [int64]$File.Length
    }
    $ext = [System.IO.Path]::GetExtension($File.FullName)
    if ([string]::IsNullOrWhiteSpace($ext)) { $ext = '' }
    $ext = $ext.ToLowerInvariant()
    Set-Prop $row 'Ext' $ext
    Set-Prop $row 'Extension' $ext
    Set-Prop $row 'LastWriteTime' $File.LastWriteTime
    Set-Prop $row 'Status' 'UNKNOWN'
    Set-Prop $row 'Kind' $null
    Set-Prop $row 'Width' $null
    Set-Prop $row 'Height' $null
    Set-Prop $row 'DurationSec' $null
    Set-Prop $row 'Duration' $null
    Set-Prop $row 'Codec' $null
    Set-Prop $row 'Hash' $null
    return $row
}

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
    $relative = Get-RelativePath -Root $Root -FullPath $FullPath
    if ([string]::IsNullOrEmpty($relative)) { return $relative }
    $relative = $relative.TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    if ([string]::IsNullOrEmpty($relative)) { return '.' }
    return $relative
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
            Write-Log ("Failed to read hash cache: {0}" -f $_.Exception.Message) 'WARN'
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
            Write-Log ("Failed to enumerate {0}: {1}" -f $current.Path, $_.Exception.Message) 'WARN'
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
    $psi = @('-v','error','-select_streams','v:0','-show_entries','stream=width,height,codec_name:format=duration','-of','json',$TargetPath)
    try {
        $result = & $FfprobePath @psi 2>$null
        if (-not $result) { return $null }
        $parsed = $result | ConvertFrom-Json -ErrorAction Stop
        $duration = $null
        if ($parsed.format.duration) {
            $duration = [double]::Parse($parsed.format.duration, [System.Globalization.CultureInfo]::InvariantCulture)
        }
        $stream = $parsed.streams | Where-Object { $_.width -and $_.height } | Select-Object -First 1
        $codec = if ($stream -and $stream.codec_name) { $stream.codec_name } else { $null }
        return [PSCustomObject]@{
            DurationSec = $duration
            Width       = if ($stream) { $stream.width } else { $null }
            Height      = if ($stream) { $stream.height } else { $null }
            Codec       = $codec
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
    $extension = [System.IO.Path]::GetExtension($Path)
    if ([string]::IsNullOrWhiteSpace($extension)) { $extension = '' }
    $extension = $extension.ToLowerInvariant()
    $pattern = switch ($extension) {
        '.ps1' { 'function\s+([A-Za-z0-9_-]+)' }
        '.psm1' { 'function\s+([A-Za-z0-9_-]+)' }
        '.py' { 'def\s+([A-Za-z0-9_]+)\s*\(' }
        '.sh' { '^([A-Za-z0-9_]+)\s*\(\)\s*\x7b' }
        '.bash' { '^([A-Za-z0-9_]+)\s*\(\)\s*\x7b' }
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
        $ext = [System.IO.Path]::GetExtension($file.FullName)
        if ([string]::IsNullOrWhiteSpace($ext)) { $ext = '' }
        $ext = $ext.ToLowerInvariant()
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
        if (-not $match.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($match.RelativePath)) {
            if ($match.FullPath) { Set-Prop $match 'RelativePath' (Get-RelPath $RootPath $match.FullPath) }
        }
        $lines += "| $($match.RelativePath) | $($match.SuggestedMode) |"
    }
    Set-Content -LiteralPath $compatPath -Value $lines
    return $matches
}

function Get-InventoryRecords {
    param(
        [System.Collections.IEnumerable]$Files,
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
    $records = New-RecordList
    $mediaExtensions = @('.mp4','.mov','.mkv','.avi','.m4v','.mpg','.mpeg')
    $durationLookup = @{}
    foreach ($file in $Files) {
        $fullPath = $file.FullPath
        if (-not $fullPath) { continue }
        if (-not $file.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($file.RelativePath)) {
            if ($file.FullPath) { Set-Prop $file 'RelativePath' (Get-RelPath $RootPath $file.FullPath) }
        }
        $relative = if ($file.PSObject.Properties.Match('RelativePath')) { $file.RelativePath } else { $null }
        if (-not $relative) { $relative = ConvertTo-RelativePath -Root $RootPath -FullPath $fullPath }
        $extension = $null
        if ($file.PSObject.Properties.Match('Ext') -and -not [string]::IsNullOrWhiteSpace($file.Ext)) {
            $extension = $file.Ext
        } elseif ($file.PSObject.Properties.Match('Extension') -and -not [string]::IsNullOrWhiteSpace($file.Extension)) {
            $extension = $file.Extension
        }
        if ([string]::IsNullOrWhiteSpace($extension)) {
            $extension = [System.IO.Path]::GetExtension($fullPath)
        }
        if ([string]::IsNullOrWhiteSpace($extension)) { $extension = '' }
        $extension = $extension.ToLowerInvariant()
        Set-Prop -Obj $file -Name 'Ext' -Value $extension
        Set-Prop -Obj $file -Name 'Extension' -Value $extension
        $sizeBytes = 0
        if ($file.PSObject.Properties.Match('SizeBytes') -and $null -ne $file.SizeBytes) {
            $sizeBytes = [int64]$file.SizeBytes
        }
        if ($sizeBytes -le 0) {
            try { $sizeBytes = [int64](Get-Item -LiteralPath $fullPath).Length } catch { $sizeBytes = 0 }
            Set-Prop -Obj $file -Name 'SizeBytes' -Value $sizeBytes
        }
        $lastWrite = $null
        if ($file.PSObject.Properties.Match('LastWriteTime') -and $file.LastWriteTime) {
            $lastWrite = $file.LastWriteTime
        }
        if (-not $lastWrite) {
            try { $lastWrite = (Get-Item -LiteralPath $fullPath).LastWriteTime } catch { $lastWrite = $null }
            Set-Prop -Obj $file -Name 'LastWriteTime' -Value $lastWrite
        }
        $durationSec = $null
        $width = $null
        $height = $null
        $codec = $null
        if ($ffprobeCmd -and $mediaExtensions -contains $extension) {
            $meta = Invoke-Ffprobe -FfprobePath $ffprobeCmd.Source -TargetPath $fullPath
            if ($meta) {
                if ($meta.PSObject.Properties.Match('DurationSec') -and $meta.DurationSec) {
                    $durationSec = [Math]::Round([double]$meta.DurationSec, 3)
                }
                if ($meta.PSObject.Properties.Match('Width') -and $meta.Width) {
                    $width = [int]$meta.Width
                }
                if ($meta.PSObject.Properties.Match('Height') -and $meta.Height) {
                    $height = [int]$meta.Height
                }
                if ($meta.PSObject.Properties.Match('Codec') -and $meta.Codec) {
                    $codec = $meta.Codec
                }
            }
        }
        if ($durationSec) { $durationLookup[$fullPath] = $durationSec }
        $hash = if ($file.PSObject.Properties.Match('Hash')) { $file.Hash } else { $null }
        $cacheKey = ConvertTo-RelativePath -Root $RootPath -FullPath $fullPath
        if ($HashCache.ContainsKey($cacheKey)) {
            $entry = $HashCache[$cacheKey]
            if ([double]$entry.SizeBytes -eq $sizeBytes -and [datetime]$entry.LastWriteTime -eq $lastWrite) {
                $hash = $entry.Hash
            }
        }
        $needsHash = -not $Fast
        if ($Fast -and $durationSec) {
            # Hash later only for probable duplicates
            $needsHash = $false
        }
        if ($needsHash -and -not $hash) {
            try {
                $hashObj = Get-FileHash -Algorithm $HashAlgo -LiteralPath $fullPath
                $hash = $hashObj.Hash
            } catch {
                Write-Log ("Failed to hash {0}: {1}" -f $relative, $_.Exception.Message) 'WARN'
            }
        }
        $kind = if ($mediaExtensions -contains $extension) { 'MEDIA' } else { 'FILE' }
        Set-Prop -Obj $file -Name 'Hash' -Value $hash
        $status = if ($file.PSObject.Properties.Match('Status') -and $file.Status -and $file.Status -ne 'UNKNOWN') { $file.Status } else { 'ORPHAN' }
        $relLower = $relative.Replace([System.IO.Path]::AltDirectorySeparatorChar, [System.IO.Path]::DirectorySeparatorChar)
        if (Test-PatternListMatch -Patterns $KeepPatterns -Path $relLower) {
            $status = 'KEEP'
        } elseif (Test-PatternListMatch -Patterns $RemovePatterns -Path $relLower) {
            $status = 'CANDIDATE_REMOVE'
        }
        if ($status -eq 'KEEP' -and $SidecarExts -contains $extension.ToLowerInvariant()) {
            $status = 'KEEP_SIDECAR'
        }
        Set-Prop -Obj $file -Name 'RelativePath' -Value $relative
        Set-Prop -Obj $file -Name 'SizeBytes' -Value $sizeBytes
        Set-Prop -Obj $file -Name 'LastWriteTime' -Value $lastWrite
        Set-Prop -Obj $file -Name 'Extension' -Value $extension
        Set-Prop -Obj $file -Name 'Hash' -Value $hash
        Set-Prop -Obj $file -Name 'Status' -Value $status
        Set-Prop -Obj $file -Name 'Kind' -Value $kind
        Set-Prop -Obj $file -Name 'DurationSec' -Value $durationSec
        Set-Prop -Obj $file -Name 'Duration' -Value $durationSec
        Set-Prop -Obj $file -Name 'Width' -Value $width
        Set-Prop -Obj $file -Name 'Height' -Value $height
        Set-Prop -Obj $file -Name 'Codec' -Value $codec
        [void]$records.Add($file)
        if ($hash) {
            $HashCache[$cacheKey] = [PSCustomObject]@{
                Path          = $cacheKey
                SizeBytes     = $sizeBytes
                LastWriteTime = $lastWrite
                Hash          = $hash
            }
        }
    }
    return $records
function Get-InventoryRecords {
    param(
        [Parameter(Mandatory)][System.Collections.IEnumerable]$Files,
        [Parameter(Mandatory)][string]$RootPath,
        [hashtable]$HashCache = $null,
        [string]$HashAlgo = 'MD5'
    )

    # Always return a growable List
    $records = [System.Collections.Generic.List[psobject]]::new()

    foreach ($file in $Files) {
        $fullPath = $file.FullName
        $relPath  = [IO.Path]::GetRelativePath($RootPath, $fullPath)
        $size     = $file.Length
        $mtime    = $file.LastWriteTimeUtc

        # Pre-seed record with expected fields
        $rec = [PSCustomObject]@{
            RelativePath  = $relPath
            FullPath      = $fullPath
            SizeBytes     = $size
            LastWriteTime = $mtime
            Extension     = ''
            Codec         = $null
            Width         = $null
            Height        = $null
            Hash          = $null
            Status        = $null
            Notes         = $null
        }

        # Extension (no ?? operators – PS 5.1 compatible)
        $ext = [IO.Path]::GetExtension($fullPath)
        if ([string]::IsNullOrEmpty($ext)) { $ext = '' }
        $rec.Extension = $ext.ToLowerInvariant()

        # Optional: hash with cache
        if ($HashCache) {
            $cacheKey = "$relPath|$size|$mtime"
            if ($HashCache.ContainsKey($cacheKey)) {
                $rec.Hash = $HashCache[$cacheKey].Hash
            } else {
                try {
                    $hasher = [System.Security.Cryptography.HashAlgorithm]::Create($HashAlgo)
                    $fs = [IO.File]::OpenRead($fullPath)
                    try {
                        $hash = ($hasher.ComputeHash($fs) | ForEach-Object { <#
.SYNOPSIS
  Unified repository cleanup, inventory, deduplication, and indexing tool for the soccer-video project.
.DESCRIPTION
  Provides multiple modes that inventory repository assets, plan cleanup operations, execute them with
  safety rails, and build lightweight indices. Designed to supersede scattered one-off scripts and provide
  a single maintained entry point for maintenance tasks.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inventory','Plan','Execute','Index','Doctor','FindExistingTools')]
    [string]$Mode = 'Inventory',

    [string]$Root = (Split-Path -Parent $PSScriptRoot),

    [switch]$ConfirmRun,
    [switch]$DryRun,
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

    [int]$Concurrent = 4
)

function New-RecordList {
    return [System.Collections.Generic.List[psobject]]::new()
}

function To-RecordList {
    param([System.Collections.IEnumerable]$Items)
    $list = New-RecordList
    foreach ($it in $Items) { [void]$list.Add($it) }
    return $list
}

function Set-Prop {
    param(
        [Parameter(Mandatory)]$Obj,
        [Parameter(Mandatory)][string]$Name,
        $Value
    )
    $p = $Obj.PSObject.Properties[$Name]
    if ($null -eq $p) {
        $Obj | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    } else {
        $Obj.$Name = $Value
    }
}

function New-InventoryRecord {
    param(
        [string]$RelativePath = '',
        [string]$FullPath     = '',
        [nullable[int64]]$SizeBytes = $null,
        [datetime]$LastWriteTime = [datetime]::MinValue
    )
    return [PSCustomObject]@{
        RelativePath  = $RelativePath
        FullPath      = $FullPath
        SizeBytes     = $SizeBytes
        LastWriteTime = $LastWriteTime
        Extension     = ''
        Codec         = $null
        Width         = $null
        Height        = $null
        Hash          = $null
        Status        = $null
        Notes         = $null
    }
}

function Get-RelPath([string]$Base,[string]$Path) {
    try {
        $b = (Resolve-Path -LiteralPath $Base).Path
        $p = (Resolve-Path -LiteralPath $Path).Path
    } catch { return $Path }
    if ($p.StartsWith($b,[System.StringComparison]::OrdinalIgnoreCase)) {
        return $p.Substring($b.Length).TrimStart('\\','/')
    }
    return $p
}

function Run-Inventory {
    param([Parameter(Mandatory)][string]$Root, [switch]$Fast)

    $files = Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue
    Write-Host ("{0} [INFO] Inventory seeded with {1} rows" -f (Get-Date -Format u), $files.Count)

    # Use a growable list; avoid .Add() on fixed-size arrays
    $rows = New-RecordList

    foreach ($f in $files) {
        $ext = [System.IO.Path]::GetExtension($f.FullName)
        if ([string]::IsNullOrWhiteSpace($ext)) { $ext = '' }
        $ext = $ext.ToLowerInvariant()

        $row = [pscustomobject]@{
            FullPath     = $f.FullName
            RelativePath = Get-RelPath $Root $f.FullName
            SizeBytes    = [int64]$f.Length
            Ext          = $ext
        }
        # Maintain compatibility with any downstream code expecting 'Extension'
        Set-Prop $row 'Extension' $ext

        [void]$rows.Add($row)
    }

    $outDir = Join-Path $Root 'out\inventory'
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $csv = Join-Path $outDir 'inventory.csv'
    $rows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $csv

    return $csv
}

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $PSBoundParameters.ContainsKey('DryRun')) {
    $DryRun = [System.Management.Automation.SwitchParameter]::new($true)
}

if ($Mode -eq 'Execute') {
    if (-not $ConfirmRun) {
        throw "Refusing to act: pass -ConfirmRun"
    }
    if ($DryRun) {
        throw "Refusing to act: you left -DryRun on (default). Use -DryRun:`$false."
    }
}

function Get-RelativePath {
    param(
        [string]$Root,
        [string]$FullPath
    )
    if (-not $FullPath) { return $null }
    $resolvedRoot = if ([string]::IsNullOrWhiteSpace($Root)) { $null } else { [System.IO.Path]::GetFullPath($Root) }
    $resolvedFull = [System.IO.Path]::GetFullPath($FullPath)
    if ($resolvedRoot) {
        $trimmedRoot = $resolvedRoot.TrimEnd('\','/')
        $candidates = @(
            $trimmedRoot + [System.IO.Path]::DirectorySeparatorChar,
            $trimmedRoot + [System.IO.Path]::AltDirectorySeparatorChar
        ) | Where-Object { -not [string]::IsNullOrEmpty($_) }
        foreach ($rootNorm in ($candidates | Select-Object -Unique)) {
            if ($resolvedFull.StartsWith($rootNorm, [System.StringComparison]::OrdinalIgnoreCase)) {
                $relative = $resolvedFull.Substring($rootNorm.Length)
                $relative = $relative.TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
                if ([string]::IsNullOrEmpty($relative)) { return '.' }
                return $relative
            }
        }
    }
    return $resolvedFull
}

function New-FileRow {
    param(
        [Parameter(Mandatory)] [string]$Root,
        [Parameter(Mandatory)] [System.IO.FileInfo]$File
    )
    $row = [pscustomobject]@{
        FullPath     = $File.FullName
        RelativePath = Get-RelPath $Root $File.FullName
        SizeBytes    = [int64]$File.Length
    }
    $ext = [System.IO.Path]::GetExtension($File.FullName)
    if ([string]::IsNullOrWhiteSpace($ext)) { $ext = '' }
    $ext = $ext.ToLowerInvariant()
    Set-Prop $row 'Ext' $ext
    Set-Prop $row 'Extension' $ext
    Set-Prop $row 'LastWriteTime' $File.LastWriteTime
    Set-Prop $row 'Status' 'UNKNOWN'
    Set-Prop $row 'Kind' $null
    Set-Prop $row 'Width' $null
    Set-Prop $row 'Height' $null
    Set-Prop $row 'DurationSec' $null
    Set-Prop $row 'Duration' $null
    Set-Prop $row 'Codec' $null
    Set-Prop $row 'Hash' $null
    return $row
}

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
    $relative = Get-RelativePath -Root $Root -FullPath $FullPath
    if ([string]::IsNullOrEmpty($relative)) { return $relative }
    $relative = $relative.TrimStart([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    if ([string]::IsNullOrEmpty($relative)) { return '.' }
    return $relative
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
            Write-Log ("Failed to read hash cache: {0}" -f $_.Exception.Message) 'WARN'
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
            Write-Log ("Failed to enumerate {0}: {1}" -f $current.Path, $_.Exception.Message) 'WARN'
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
    $psi = @('-v','error','-select_streams','v:0','-show_entries','stream=width,height,codec_name:format=duration','-of','json',$TargetPath)
    try {
        $result = & $FfprobePath @psi 2>$null
        if (-not $result) { return $null }
        $parsed = $result | ConvertFrom-Json -ErrorAction Stop
        $duration = $null
        if ($parsed.format.duration) {
            $duration = [double]::Parse($parsed.format.duration, [System.Globalization.CultureInfo]::InvariantCulture)
        }
        $stream = $parsed.streams | Where-Object { $_.width -and $_.height } | Select-Object -First 1
        $codec = if ($stream -and $stream.codec_name) { $stream.codec_name } else { $null }
        return [PSCustomObject]@{
            DurationSec = $duration
            Width       = if ($stream) { $stream.width } else { $null }
            Height      = if ($stream) { $stream.height } else { $null }
            Codec       = $codec
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
    $extension = [System.IO.Path]::GetExtension($Path)
    if ([string]::IsNullOrWhiteSpace($extension)) { $extension = '' }
    $extension = $extension.ToLowerInvariant()
    $pattern = switch ($extension) {
        '.ps1' { 'function\s+([A-Za-z0-9_-]+)' }
        '.psm1' { 'function\s+([A-Za-z0-9_-]+)' }
        '.py' { 'def\s+([A-Za-z0-9_]+)\s*\(' }
        '.sh' { '^([A-Za-z0-9_]+)\s*\(\)\s*\x7b' }
        '.bash' { '^([A-Za-z0-9_]+)\s*\(\)\s*\x7b' }
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
        $ext = [System.IO.Path]::GetExtension($file.FullName)
        if ([string]::IsNullOrWhiteSpace($ext)) { $ext = '' }
        $ext = $ext.ToLowerInvariant()
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
        if (-not $match.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($match.RelativePath)) {
            if ($match.FullPath) { Set-Prop $match 'RelativePath' (Get-RelPath $RootPath $match.FullPath) }
        }
        $lines += "| $($match.RelativePath) | $($match.SuggestedMode) |"
    }
    Set-Content -LiteralPath $compatPath -Value $lines
    return $matches
}

function Get-InventoryRecords {
    param(
        [System.Collections.IEnumerable]$Files,
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
    $records = New-RecordList
    $mediaExtensions = @('.mp4','.mov','.mkv','.avi','.m4v','.mpg','.mpeg')
    $durationLookup = @{}
    foreach ($file in $Files) {
        $fullPath = $file.FullPath
        if (-not $fullPath) { continue }
        if (-not $file.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($file.RelativePath)) {
            if ($file.FullPath) { Set-Prop $file 'RelativePath' (Get-RelPath $RootPath $file.FullPath) }
        }
        $relative = if ($file.PSObject.Properties.Match('RelativePath')) { $file.RelativePath } else { $null }
        if (-not $relative) { $relative = ConvertTo-RelativePath -Root $RootPath -FullPath $fullPath }
        $extension = $null
        if ($file.PSObject.Properties.Match('Ext') -and -not [string]::IsNullOrWhiteSpace($file.Ext)) {
            $extension = $file.Ext
        } elseif ($file.PSObject.Properties.Match('Extension') -and -not [string]::IsNullOrWhiteSpace($file.Extension)) {
            $extension = $file.Extension
        }
        if ([string]::IsNullOrWhiteSpace($extension)) {
            $extension = [System.IO.Path]::GetExtension($fullPath)
        }
        if ([string]::IsNullOrWhiteSpace($extension)) { $extension = '' }
        $extension = $extension.ToLowerInvariant()
        Set-Prop -Obj $file -Name 'Ext' -Value $extension
        Set-Prop -Obj $file -Name 'Extension' -Value $extension
        $sizeBytes = 0
        if ($file.PSObject.Properties.Match('SizeBytes') -and $null -ne $file.SizeBytes) {
            $sizeBytes = [int64]$file.SizeBytes
        }
        if ($sizeBytes -le 0) {
            try { $sizeBytes = [int64](Get-Item -LiteralPath $fullPath).Length } catch { $sizeBytes = 0 }
            Set-Prop -Obj $file -Name 'SizeBytes' -Value $sizeBytes
        }
        $lastWrite = $null
        if ($file.PSObject.Properties.Match('LastWriteTime') -and $file.LastWriteTime) {
            $lastWrite = $file.LastWriteTime
        }
        if (-not $lastWrite) {
            try { $lastWrite = (Get-Item -LiteralPath $fullPath).LastWriteTime } catch { $lastWrite = $null }
            Set-Prop -Obj $file -Name 'LastWriteTime' -Value $lastWrite
        }
        $durationSec = $null
        $width = $null
        $height = $null
        $codec = $null
        if ($ffprobeCmd -and $mediaExtensions -contains $extension) {
            $meta = Invoke-Ffprobe -FfprobePath $ffprobeCmd.Source -TargetPath $fullPath
            if ($meta) {
                if ($meta.PSObject.Properties.Match('DurationSec') -and $meta.DurationSec) {
                    $durationSec = [Math]::Round([double]$meta.DurationSec, 3)
                }
                if ($meta.PSObject.Properties.Match('Width') -and $meta.Width) {
                    $width = [int]$meta.Width
                }
                if ($meta.PSObject.Properties.Match('Height') -and $meta.Height) {
                    $height = [int]$meta.Height
                }
                if ($meta.PSObject.Properties.Match('Codec') -and $meta.Codec) {
                    $codec = $meta.Codec
                }
            }
        }
        if ($durationSec) { $durationLookup[$fullPath] = $durationSec }
        $hash = if ($file.PSObject.Properties.Match('Hash')) { $file.Hash } else { $null }
        $cacheKey = ConvertTo-RelativePath -Root $RootPath -FullPath $fullPath
        if ($HashCache.ContainsKey($cacheKey)) {
            $entry = $HashCache[$cacheKey]
            if ([double]$entry.SizeBytes -eq $sizeBytes -and [datetime]$entry.LastWriteTime -eq $lastWrite) {
                $hash = $entry.Hash
            }
        }
        $needsHash = -not $Fast
        if ($Fast -and $durationSec) {
            # Hash later only for probable duplicates
            $needsHash = $false
        }
        if ($needsHash -and -not $hash) {
            try {
                $hashObj = Get-FileHash -Algorithm $HashAlgo -LiteralPath $fullPath
                $hash = $hashObj.Hash
            } catch {
                Write-Log ("Failed to hash {0}: {1}" -f $relative, $_.Exception.Message) 'WARN'
            }
        }
        $kind = if ($mediaExtensions -contains $extension) { 'MEDIA' } else { 'FILE' }
        Set-Prop -Obj $file -Name 'Hash' -Value $hash
        $status = if ($file.PSObject.Properties.Match('Status') -and $file.Status -and $file.Status -ne 'UNKNOWN') { $file.Status } else { 'ORPHAN' }
        $relLower = $relative.Replace([System.IO.Path]::AltDirectorySeparatorChar, [System.IO.Path]::DirectorySeparatorChar)
        if (Test-PatternListMatch -Patterns $KeepPatterns -Path $relLower) {
            $status = 'KEEP'
        } elseif (Test-PatternListMatch -Patterns $RemovePatterns -Path $relLower) {
            $status = 'CANDIDATE_REMOVE'
        }
        if ($status -eq 'KEEP' -and $SidecarExts -contains $extension.ToLowerInvariant()) {
            $status = 'KEEP_SIDECAR'
        }
        Set-Prop -Obj $file -Name 'RelativePath' -Value $relative
        Set-Prop -Obj $file -Name 'SizeBytes' -Value $sizeBytes
        Set-Prop -Obj $file -Name 'LastWriteTime' -Value $lastWrite
        Set-Prop -Obj $file -Name 'Extension' -Value $extension
        Set-Prop -Obj $file -Name 'Hash' -Value $hash
        Set-Prop -Obj $file -Name 'Status' -Value $status
        Set-Prop -Obj $file -Name 'Kind' -Value $kind
        Set-Prop -Obj $file -Name 'DurationSec' -Value $durationSec
        Set-Prop -Obj $file -Name 'Duration' -Value $durationSec
        Set-Prop -Obj $file -Name 'Width' -Value $width
        Set-Prop -Obj $file -Name 'Height' -Value $height
        Set-Prop -Obj $file -Name 'Codec' -Value $codec
        [void]$records.Add($file)
        if ($hash) {
            $HashCache[$cacheKey] = [PSCustomObject]@{
                Path          = $cacheKey
                SizeBytes     = $sizeBytes
                LastWriteTime = $lastWrite
                Hash          = $hash
            }
        }
    }
    return $records

function Enhance-HashesForFastMode {
    param(
        [System.Collections.IEnumerable]$Records,
        [string]$RootPath,
        [string]$HashAlgo,
        [hashtable]$HashCache
    )
    $buckets = @{}
    foreach ($record in $Records) {
        $sizeValue = $record.SizeBytes
        if ($null -eq $sizeValue -or -not ($sizeValue -as [double])) {
            $sizeBucket = 'NA'
        } else {
            $sizeBucket = [Math]::Round(([double]$sizeValue) / 16)
        }

        $durationValue = if ($record.PSObject.Properties.Match('DurationSec')) { $record.DurationSec } else { $record.Duration }
        if ($null -eq $durationValue -or -not ($durationValue -as [double])) {
            $durationBucket = 'NA'
        } else {
            $durationBucket = [Math]::Round([double]$durationValue, 1)
        }

        $key = "S${sizeBucket}-D${durationBucket}"
        if (-not $buckets.ContainsKey($key)) {
            $buckets[$key] = New-RecordList
        }
        $buckets[$key].Add($record)
    }
    foreach ($bucket in $buckets.GetEnumerator()) {
        if ($bucket.Value.Count -le 1) { continue }
        foreach ($record in $bucket.Value) {
            if ([string]::IsNullOrWhiteSpace($record.Hash)) {
                if (-not $record.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($record.RelativePath)) {
                    if ($record.FullPath) { Set-Prop $record 'RelativePath' (Get-RelPath $RootPath $record.FullPath) }
                }
                $cacheKey = $record.RelativePath
                if ($HashCache.ContainsKey($cacheKey)) {
                    Set-Prop -Obj $record -Name 'Hash' -Value $HashCache[$cacheKey].Hash
                    continue
                }
                try {
                    $hashObj = Get-FileHash -Algorithm $HashAlgo -LiteralPath $record.FullPath
                    Set-Prop -Obj $record -Name 'Hash' -Value $hashObj.Hash
                    $HashCache[$cacheKey] = [PSCustomObject]@{
                        Path          = $cacheKey
                        SizeBytes     = $record.SizeBytes
                        LastWriteTime = $record.LastWriteTime
                        Hash          = $record.Hash
                    }
                } catch {
                    Write-Log ("Fast hash upgrade failed for {0}: {1}" -f $record.RelativePath, $_.Exception.Message) 'WARN'
                }
            }
        }
    }
}

function Export-InventoryOutputs {
    param(
        [System.Collections.IEnumerable]$Records,
        [string]$InventoryDir,
        [string]$RootPath = $null
    )
    $recordList = To-RecordList $Records
    foreach ($row in $recordList) {
        if (-not $row.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($row.RelativePath)) {
            if ($RootPath -and $row.FullPath) { Set-Prop $row 'RelativePath' (Get-RelPath $RootPath $row.FullPath) }
        }
    }
    $jsonPath = Join-Path $InventoryDir 'repo_inventory.json'
    $csvPath = Join-Path $InventoryDir 'repo_inventory.csv'
    $recordList | Export-Csv -LiteralPath $csvPath -NoTypeInformation
    $recordList | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $jsonPath

    $hashGroups = $recordList | Where-Object { $_.Hash } | Group-Object Hash | Where-Object { $_.Count -gt 1 }
    $dupeExactPath = Join-Path $InventoryDir 'duplicates_exact.csv'
    $exactRows = @()
    foreach ($group in $hashGroups) {
        foreach ($item in $group.Group) {
            $durationValue = if ($item.PSObject.Properties.Match('DurationSec')) { $item.DurationSec } else { $item.Duration }
            $exactRows += [PSCustomObject]@{
                Hash         = $group.Name
                RelativePath = $item.RelativePath
                SizeBytes    = $item.SizeBytes
                Duration     = $durationValue
            }
        }
    }
    $exactRows | Export-Csv -LiteralPath $dupeExactPath -NoTypeInformation

    $probablePath = Join-Path $InventoryDir 'duplicates_probable.csv'
    $probableRows = @()
    $recordList |
        Group-Object -Property {
            $dur = if ($_.PSObject.Properties.Match('DurationSec')) { $_.DurationSec } else { $_.Duration }
            if ($null -eq $dur -or -not ($dur -as [double])) {
                $durKey = 'NA'
            } else {
                $durKey = [Math]::Round([double]$dur, 1)
            }

            $widthVal = $_.Width
            if ($null -eq $widthVal -or -not ($widthVal -as [double])) {
                $widthKey = 'NA'
            } else {
                $widthKey = [Math]::Round(([double]$widthVal) / 10) * 10
            }

            $heightVal = $_.Height
            if ($null -eq $heightVal -or -not ($heightVal -as [double])) {
                $heightKey = 'NA'
            } else {
                $heightKey = [Math]::Round(([double]$heightVal) / 10) * 10
            }

            '{0}-{1}-{2}' -f $durKey, $widthKey, $heightKey
        } |
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
    ($recordList | Where-Object { $_.Status -like 'KEEP*' }) | Export-Csv -LiteralPath $keepPath -NoTypeInformation
    ($recordList | Where-Object { $_.Status -eq 'CANDIDATE_REMOVE' }) | Export-Csv -LiteralPath $removePath -NoTypeInformation
    ($recordList | Where-Object { $_.Status -eq 'ORPHAN' }) | Export-Csv -LiteralPath $orphansPath -NoTypeInformation

    $summaryPath = Join-Path $InventoryDir 'summary_sizes_by_type.csv'
    $recordList |
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
        [System.Collections.IEnumerable]$Records,
        [string]$InventoryDir,
        [string]$RootPath = $null
    )
    $recordList = @($Records)
    foreach ($row in $recordList) {
        if (-not $row.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($row.RelativePath)) {
            if ($RootPath -and $row.FullPath) { Set-Prop $row 'RelativePath' (Get-RelPath $RootPath $row.FullPath) }
        }
    }
    $masters = $recordList | Where-Object { $_.RelativePath -match 'master' -or $_.RelativePath -match 'Game ' }
    $atomics = $recordList | Where-Object { $_.RelativePath -match 'atomic' }
    $reels   = $recordList | Where-Object { $_.RelativePath -match 'reel' -or $_.RelativePath -match 'postable' }

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
    if ($null -eq $records) { $records = @() } else { $records = @($records) }
    $plan = @()
    $hardlinkPlan = @()
    $quarantineRoot = Join-Path $RootPath '_quarantine'
    $planIndex = 0
    foreach ($item in $records) {
        if (-not $item.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($item.RelativePath)) {
            if ($item.FullPath) { Set-Prop $item 'RelativePath' (Get-RelPath $RootPath $item.FullPath) }
        }
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
            $duplicates = @($records | Where-Object { $_.Hash -eq $item.Hash })
            if ($duplicates.Count -gt 1) {
                $canonical = $duplicates | Sort-Object -Property RelativePath | Select-Object -First 1
                if ($canonical -and (-not $canonical.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($canonical.RelativePath))) {
                    if ($canonical.FullPath) { Set-Prop $canonical 'RelativePath' (Get-RelPath $RootPath $canonical.FullPath) }
                }
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
    if (-not $ConfirmRun) {
        throw 'Refusing to act: you must pass -ConfirmRun.'
    }
    if ($DryRun) {
        throw 'Refusing to act: you passed -DryRun (default). Use -DryRun:$false to operate.'
    }
    if (-not (Test-Path -LiteralPath $PlanPath)) {
        throw "Plan file not found at $PlanPath"
    }
    $planInfo = Get-Item -LiteralPath $PlanPath
    if ((Get-Date) - $planInfo.LastWriteTime -gt [TimeSpan]::FromHours(48)) {
        throw 'Plan is older than 48 hours. Generate a fresh plan before executing.'
    }
    $planRaw = Import-Csv -LiteralPath $PlanPath
    if ($null -eq $planRaw) { $planRaw = @() } else { $planRaw = @($planRaw) }
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
        if (-not $row.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($row.RelativePath)) {
            if ($row.FullPath) { Set-Prop $row 'RelativePath' (Get-RelPath $RootPath $row.FullPath) }
        }
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
    $failed = New-RecordList
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
        [string]$IndexDir,
        [string]$RootPath = $null
    )
    $inventoryFile = Join-Path $InventoryDir 'repo_inventory.csv'
    if (-not (Test-Path -LiteralPath $inventoryFile)) {
        throw 'Inventory not found. Run -Mode Inventory first.'
    }
    $records = Import-Csv -LiteralPath $inventoryFile
    if ($null -eq $records) { $records = @() } else { $records = @($records) }
    foreach ($record in $records) {
        if (-not $record.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($record.RelativePath)) {
            if ($RootPath -and $record.FullPath) { Set-Prop $record 'RelativePath' (Get-RelPath $RootPath $record.FullPath) }
        }
    }
    $kept = @($records | Where-Object { $_.Status -like 'KEEP*' })
    $index = @()
    $groups = @($kept | Group-Object -Property { ($_.RelativePath -split [System.IO.Path]::DirectorySeparatorChar)[0] })
    foreach ($group in $groups) {
        $atomics = @($group.Group | Where-Object { $_.RelativePath -match 'atomic' })
        $reels = @($group.Group | Where-Object { $_.RelativePath -match 'reel' -or $_.RelativePath -match 'postable' })
        $masters = @($group.Group | Where-Object { $_.RelativePath -match 'master' -or $_.RelativePath -match 'Game ' })
        $durationTotal = ($atomics | Measure-Object -Property Duration -Sum).Sum
        $durationValue = if ($durationTotal) { $durationTotal } else { 0 }
        $index += [PSCustomObject]@{
            Bucket            = $group.Name
            AtomicClipCount   = $atomics.Count
            AtomicDurationSec = [Math]::Round([double]$durationValue, 2)
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

switch ($Mode.ToLowerInvariant()) {
    'findexistingtools' {
        Find-ExistingTools -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -DocsDir $docsDir | Out-Null
        break
    }
    'inventory' {
        $excludeDirs = @('.git', '.venv', 'out\esrgan', 'out\render_logs\scratch', '_trash', '_quarantine')
        $excludes = $excludeDirs | ForEach-Object { ('*' + $_ + '*') }

        $seedPath = Run-Inventory -Root $repoRoot -Fast:$Fast
        $seedRows = $null
        if ($seedPath -and (Test-Path -LiteralPath $seedPath)) {
            $seedRows = Import-Csv -LiteralPath $seedPath
        }
        if ($null -eq $seedRows) { $seedRows = New-RecordList } else { $seedRows = To-RecordList $seedRows }

        $filesList = New-RecordList
        foreach ($row in $seedRows) {
            if (-not $row) { continue }
            $full = $row.FullPath
            if (-not $full) { continue }
            $skip = $false
            foreach ($pattern in $excludes) {
                if ($full -like $pattern) { $skip = $true; break }
                if ($row.RelativePath -and $row.RelativePath -like $pattern) { $skip = $true; break }
            }
            if ($skip) { continue }
            try {
                $fileInfo = Get-Item -LiteralPath $full -ErrorAction Stop
            } catch {
                continue
            }
            if (-not $FollowJunctions -and ($fileInfo.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { continue }
            if (-not $IncludeHidden -and ($fileInfo.Attributes -band [System.IO.FileAttributes]::Hidden)) { continue }
            if ($MaxDepth -gt 0) {
                $relative = $row.RelativePath
                if (-not $relative) { $relative = Get-RelPath $repoRoot $full }
                if (-not [string]::IsNullOrEmpty($relative) -and $relative -ne '.') {
                    $segments = ($relative -split '[\\/]+') | Where-Object { $_ }
                    if ($segments.Count -gt $MaxDepth) { continue }
                }
            }
            $fileRow = New-FileRow -Root $repoRoot -File $fileInfo
            if (-not $fileRow.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($fileRow.RelativePath)) {
                if ($fileRow.FullPath) { Set-Prop $fileRow 'RelativePath' (Get-RelPath $repoRoot $fileRow.FullPath) }
            }
            [void]$filesList.Add($fileRow)
        }

        $files = $filesList
        Write-Log "Inventory seeded with $($files.Count) rows" 'INFO'
        $Inventory = Get-InventoryRecords -Files $files -RootPath $repoRoot -Fast:$Fast -HashAlgo $HashAlgo -KeepPatterns $keepPatterns -RemovePatterns $removePatterns -SidecarExts $rules.SidecarExts -HashCache $hashCache -HashCachePath $hashCachePath
        $Inventory = To-RecordList $Inventory
        if ($Fast) {
            Enhance-HashesForFastMode -Records $Inventory -RootPath $repoRoot -HashAlgo $HashAlgo -HashCache $hashCache
        }
        Export-InventoryOutputs -Records $Inventory -InventoryDir $outputPaths.Inventory -RootPath $repoRoot
        Build-MappingFiles -Records $Inventory -InventoryDir $outputPaths.Inventory -RootPath $repoRoot
        Save-HashCache -Cache $hashCache -CachePath $hashCachePath
        Find-ExistingTools -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -DocsDir $docsDir | Out-Null
        Write-Log 'Inventory complete.' 'SUCCESS'
        break
    }
    'plan' {
        $inventoryFile = Join-Path $outputPaths.Inventory 'repo_inventory.csv'
        if (-not (Test-Path -LiteralPath $inventoryFile)) {
            throw 'No inventory found. Run -Mode Inventory first.'
        }
        $planPath = New-CleanupPlan -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -PlansDir $outputPaths.Plans -Strategy $Strategy -EnableHardlink:$EnableHardlink -Permanent:$Permanent
        Write-Log "Plan written to $planPath" 'SUCCESS'
        break
    }
    'execute' {
        $planPath = Join-Path $outputPaths.Plans 'cleanup_plan.csv'
        Invoke-CleanupExecution -PlanPath $planPath -RootPath $repoRoot -LogsDir $outputPaths.Logs -DryRun:$DryRun -ConfirmRun:$ConfirmRun -Concurrent $Concurrent
        Write-Log 'Execution complete.' 'SUCCESS'
        break
    }
    'index' {
        Build-SeasonIndex -InventoryDir $outputPaths.Inventory -IndexDir $outputPaths.Index -RootPath $repoRoot
        Write-Log 'Index refreshed.' 'SUCCESS'
        break
    }
    'doctor' {
        Run-Doctor -RootPath $repoRoot -InventoryDir $outputPaths.Inventory
        Find-ExistingTools -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -DocsDir $docsDir | Out-Null
        break
    }
    default {
        throw "Unknown -Mode '$Mode'."
    }
}

.ToString('x2') }) -join ''
                        $rec.Hash = $hash
                        $HashCache[$cacheKey] = [PSCustomObject]@{
                            Path          = $cacheKey
                            SizeBytes     = $size
                            LastWriteTime = $mtime
                            Hash          = $hash
                        }
                    } finally { $fs.Dispose() }
                } catch {
                    # non-fatal
                }
            }
        }

        [void]$records.Add($rec)
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
        $sizeValue = $record.SizeBytes
        if ($null -eq $sizeValue -or -not ($sizeValue -as [double])) {
            $sizeBucket = 'NA'
        } else {
            $sizeBucket = [Math]::Round(([double]$sizeValue) / 16)
        }

        $durationValue = if ($record.PSObject.Properties.Match('DurationSec')) { $record.DurationSec } else { $record.Duration }
        if ($null -eq $durationValue -or -not ($durationValue -as [double])) {
            $durationBucket = 'NA'
        } else {
            $durationBucket = [Math]::Round([double]$durationValue, 1)
        }

        $key = "S${sizeBucket}-D${durationBucket}"
        if (-not $buckets.ContainsKey($key)) {
            $buckets[$key] = New-RecordList
        }
        $buckets[$key].Add($record)
    }
    foreach ($bucket in $buckets.GetEnumerator()) {
        if ($bucket.Value.Count -le 1) { continue }
        foreach ($record in $bucket.Value) {
            if ([string]::IsNullOrWhiteSpace($record.Hash)) {
                if (-not $record.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($record.RelativePath)) {
                    if ($record.FullPath) { Set-Prop $record 'RelativePath' (Get-RelPath $RootPath $record.FullPath) }
                }
                $cacheKey = $record.RelativePath
                if ($HashCache.ContainsKey($cacheKey)) {
                    Set-Prop -Obj $record -Name 'Hash' -Value $HashCache[$cacheKey].Hash
                    continue
                }
                try {
                    $hashObj = Get-FileHash -Algorithm $HashAlgo -LiteralPath $record.FullPath
                    Set-Prop -Obj $record -Name 'Hash' -Value $hashObj.Hash
                    $HashCache[$cacheKey] = [PSCustomObject]@{
                        Path          = $cacheKey
                        SizeBytes     = $record.SizeBytes
                        LastWriteTime = $record.LastWriteTime
                        Hash          = $record.Hash
                    }
                } catch {
                    Write-Log ("Fast hash upgrade failed for {0}: {1}" -f $record.RelativePath, $_.Exception.Message) 'WARN'
                }
            }
        }
    }
}

function Export-InventoryOutputs {
    param(
        [System.Collections.IEnumerable]$Records,
        [string]$InventoryDir,
        [string]$RootPath = $null
    )
    $recordList = To-RecordList $Records
    foreach ($row in $recordList) {
        if (-not $row.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($row.RelativePath)) {
            if ($RootPath -and $row.FullPath) { Set-Prop $row 'RelativePath' (Get-RelPath $RootPath $row.FullPath) }
        }
    }
    $jsonPath = Join-Path $InventoryDir 'repo_inventory.json'
    $csvPath = Join-Path $InventoryDir 'repo_inventory.csv'
    $recordList | Export-Csv -LiteralPath $csvPath -NoTypeInformation
    $recordList | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $jsonPath

    $hashGroups = $recordList | Where-Object { $_.Hash } | Group-Object Hash | Where-Object { $_.Count -gt 1 }
    $dupeExactPath = Join-Path $InventoryDir 'duplicates_exact.csv'
    $exactRows = @()
    foreach ($group in $hashGroups) {
        foreach ($item in $group.Group) {
            $durationValue = if ($item.PSObject.Properties.Match('DurationSec')) { $item.DurationSec } else { $item.Duration }
            $exactRows += [PSCustomObject]@{
                Hash         = $group.Name
                RelativePath = $item.RelativePath
                SizeBytes    = $item.SizeBytes
                Duration     = $durationValue
            }
        }
    }
    $exactRows | Export-Csv -LiteralPath $dupeExactPath -NoTypeInformation

    $probablePath = Join-Path $InventoryDir 'duplicates_probable.csv'
    $probableRows = @()
    $recordList |
        Group-Object -Property {
            $dur = if ($_.PSObject.Properties.Match('DurationSec')) { $_.DurationSec } else { $_.Duration }
            if ($null -eq $dur -or -not ($dur -as [double])) {
                $durKey = 'NA'
            } else {
                $durKey = [Math]::Round([double]$dur, 1)
            }

            $widthVal = $_.Width
            if ($null -eq $widthVal -or -not ($widthVal -as [double])) {
                $widthKey = 'NA'
            } else {
                $widthKey = [Math]::Round(([double]$widthVal) / 10) * 10
            }

            $heightVal = $_.Height
            if ($null -eq $heightVal -or -not ($heightVal -as [double])) {
                $heightKey = 'NA'
            } else {
                $heightKey = [Math]::Round(([double]$heightVal) / 10) * 10
            }

            '{0}-{1}-{2}' -f $durKey, $widthKey, $heightKey
        } |
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
    ($recordList | Where-Object { $_.Status -like 'KEEP*' }) | Export-Csv -LiteralPath $keepPath -NoTypeInformation
    ($recordList | Where-Object { $_.Status -eq 'CANDIDATE_REMOVE' }) | Export-Csv -LiteralPath $removePath -NoTypeInformation
    ($recordList | Where-Object { $_.Status -eq 'ORPHAN' }) | Export-Csv -LiteralPath $orphansPath -NoTypeInformation

    $summaryPath = Join-Path $InventoryDir 'summary_sizes_by_type.csv'
    $recordList |
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
        [System.Collections.IEnumerable]$Records,
        [string]$InventoryDir,
        [string]$RootPath = $null
    )
    $recordList = @($Records)
    foreach ($row in $recordList) {
        if (-not $row.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($row.RelativePath)) {
            if ($RootPath -and $row.FullPath) { Set-Prop $row 'RelativePath' (Get-RelPath $RootPath $row.FullPath) }
        }
    }
    $masters = $recordList | Where-Object { $_.RelativePath -match 'master' -or $_.RelativePath -match 'Game ' }
    $atomics = $recordList | Where-Object { $_.RelativePath -match 'atomic' }
    $reels   = $recordList | Where-Object { $_.RelativePath -match 'reel' -or $_.RelativePath -match 'postable' }

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
    if ($null -eq $records) { $records = @() } else { $records = @($records) }
    $plan = @()
    $hardlinkPlan = @()
    $quarantineRoot = Join-Path $RootPath '_quarantine'
    $planIndex = 0
    foreach ($item in $records) {
        if (-not $item.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($item.RelativePath)) {
            if ($item.FullPath) { Set-Prop $item 'RelativePath' (Get-RelPath $RootPath $item.FullPath) }
        }
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
            $duplicates = @($records | Where-Object { $_.Hash -eq $item.Hash })
            if ($duplicates.Count -gt 1) {
                $canonical = $duplicates | Sort-Object -Property RelativePath | Select-Object -First 1
                if ($canonical -and (-not $canonical.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($canonical.RelativePath))) {
                    if ($canonical.FullPath) { Set-Prop $canonical 'RelativePath' (Get-RelPath $RootPath $canonical.FullPath) }
                }
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
    if (-not $ConfirmRun) {
        throw 'Refusing to act: you must pass -ConfirmRun.'
    }
    if ($DryRun) {
        throw 'Refusing to act: you passed -DryRun (default). Use -DryRun:$false to operate.'
    }
    if (-not (Test-Path -LiteralPath $PlanPath)) {
        throw "Plan file not found at $PlanPath"
    }
    $planInfo = Get-Item -LiteralPath $PlanPath
    if ((Get-Date) - $planInfo.LastWriteTime -gt [TimeSpan]::FromHours(48)) {
        throw 'Plan is older than 48 hours. Generate a fresh plan before executing.'
    }
    $planRaw = Import-Csv -LiteralPath $PlanPath
    if ($null -eq $planRaw) { $planRaw = @() } else { $planRaw = @($planRaw) }
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
        if (-not $row.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($row.RelativePath)) {
            if ($row.FullPath) { Set-Prop $row 'RelativePath' (Get-RelPath $RootPath $row.FullPath) }
        }
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
    $failed = New-RecordList
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
        [string]$IndexDir,
        [string]$RootPath = $null
    )
    $inventoryFile = Join-Path $InventoryDir 'repo_inventory.csv'
    if (-not (Test-Path -LiteralPath $inventoryFile)) {
        throw 'Inventory not found. Run -Mode Inventory first.'
    }
    $records = Import-Csv -LiteralPath $inventoryFile
    if ($null -eq $records) { $records = @() } else { $records = @($records) }
    foreach ($record in $records) {
        if (-not $record.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($record.RelativePath)) {
            if ($RootPath -and $record.FullPath) { Set-Prop $record 'RelativePath' (Get-RelPath $RootPath $record.FullPath) }
        }
    }
    $kept = @($records | Where-Object { $_.Status -like 'KEEP*' })
    $index = @()
    $groups = @($kept | Group-Object -Property { ($_.RelativePath -split [System.IO.Path]::DirectorySeparatorChar)[0] })
    foreach ($group in $groups) {
        $atomics = @($group.Group | Where-Object { $_.RelativePath -match 'atomic' })
        $reels = @($group.Group | Where-Object { $_.RelativePath -match 'reel' -or $_.RelativePath -match 'postable' })
        $masters = @($group.Group | Where-Object { $_.RelativePath -match 'master' -or $_.RelativePath -match 'Game ' })
        $durationTotal = ($atomics | Measure-Object -Property Duration -Sum).Sum
        $durationValue = if ($durationTotal) { $durationTotal } else { 0 }
        $index += [PSCustomObject]@{
            Bucket            = $group.Name
            AtomicClipCount   = $atomics.Count
            AtomicDurationSec = [Math]::Round([double]$durationValue, 2)
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

switch ($Mode.ToLowerInvariant()) {
    'findexistingtools' {
        Find-ExistingTools -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -DocsDir $docsDir | Out-Null
        break
    }
    'inventory' {
        $excludeDirs = @('.git', '.venv', 'out\esrgan', 'out\render_logs\scratch', '_trash', '_quarantine')
        $excludes = $excludeDirs | ForEach-Object { ('*' + $_ + '*') }

        $seedPath = Run-Inventory -Root $repoRoot -Fast:$Fast
        $seedRows = $null
        if ($seedPath -and (Test-Path -LiteralPath $seedPath)) {
            $seedRows = Import-Csv -LiteralPath $seedPath
        }
        if ($null -eq $seedRows) { $seedRows = New-RecordList } else { $seedRows = To-RecordList $seedRows }

        $filesList = New-RecordList
        foreach ($row in $seedRows) {
            if (-not $row) { continue }
            $full = $row.FullPath
            if (-not $full) { continue }
            $skip = $false
            foreach ($pattern in $excludes) {
                if ($full -like $pattern) { $skip = $true; break }
                if ($row.RelativePath -and $row.RelativePath -like $pattern) { $skip = $true; break }
            }
            if ($skip) { continue }
            try {
                $fileInfo = Get-Item -LiteralPath $full -ErrorAction Stop
            } catch {
                continue
            }
            if (-not $FollowJunctions -and ($fileInfo.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { continue }
            if (-not $IncludeHidden -and ($fileInfo.Attributes -band [System.IO.FileAttributes]::Hidden)) { continue }
            if ($MaxDepth -gt 0) {
                $relative = $row.RelativePath
                if (-not $relative) { $relative = Get-RelPath $repoRoot $full }
                if (-not [string]::IsNullOrEmpty($relative) -and $relative -ne '.') {
                    $segments = ($relative -split '[\\/]+') | Where-Object { $_ }
                    if ($segments.Count -gt $MaxDepth) { continue }
                }
            }
            $fileRow = New-FileRow -Root $repoRoot -File $fileInfo
            if (-not $fileRow.PSObject.Properties['RelativePath'] -or [string]::IsNullOrWhiteSpace($fileRow.RelativePath)) {
                if ($fileRow.FullPath) { Set-Prop $fileRow 'RelativePath' (Get-RelPath $repoRoot $fileRow.FullPath) }
            }
            [void]$filesList.Add($fileRow)
        }

        $files = $filesList
        Write-Log "Inventory seeded with $($files.Count) rows" 'INFO'
        $Inventory = Get-InventoryRecords -Files $files -RootPath $repoRoot -Fast:$Fast -HashAlgo $HashAlgo -KeepPatterns $keepPatterns -RemovePatterns $removePatterns -SidecarExts $rules.SidecarExts -HashCache $hashCache -HashCachePath $hashCachePath
        $Inventory = To-RecordList $Inventory
        if ($Fast) {
            Enhance-HashesForFastMode -Records $Inventory -RootPath $repoRoot -HashAlgo $HashAlgo -HashCache $hashCache
        }
        Export-InventoryOutputs -Records $Inventory -InventoryDir $outputPaths.Inventory -RootPath $repoRoot
        Build-MappingFiles -Records $Inventory -InventoryDir $outputPaths.Inventory -RootPath $repoRoot
        Save-HashCache -Cache $hashCache -CachePath $hashCachePath
        Find-ExistingTools -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -DocsDir $docsDir | Out-Null
        Write-Log 'Inventory complete.' 'SUCCESS'
        break
    }
    'plan' {
        $inventoryFile = Join-Path $outputPaths.Inventory 'repo_inventory.csv'
        if (-not (Test-Path -LiteralPath $inventoryFile)) {
            throw 'No inventory found. Run -Mode Inventory first.'
        }
        $planPath = New-CleanupPlan -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -PlansDir $outputPaths.Plans -Strategy $Strategy -EnableHardlink:$EnableHardlink -Permanent:$Permanent
        Write-Log "Plan written to $planPath" 'SUCCESS'
        break
    }
    'execute' {
        $planPath = Join-Path $outputPaths.Plans 'cleanup_plan.csv'
        Invoke-CleanupExecution -PlanPath $planPath -RootPath $repoRoot -LogsDir $outputPaths.Logs -DryRun:$DryRun -ConfirmRun:$ConfirmRun -Concurrent $Concurrent
        Write-Log 'Execution complete.' 'SUCCESS'
        break
    }
    'index' {
        Build-SeasonIndex -InventoryDir $outputPaths.Inventory -IndexDir $outputPaths.Index -RootPath $repoRoot
        Write-Log 'Index refreshed.' 'SUCCESS'
        break
    }
    'doctor' {
        Run-Doctor -RootPath $repoRoot -InventoryDir $outputPaths.Inventory
        Find-ExistingTools -RootPath $repoRoot -InventoryDir $outputPaths.Inventory -DocsDir $docsDir | Out-Null
        break
    }
    default {
        throw "Unknown -Mode '$Mode'."
    }
}


