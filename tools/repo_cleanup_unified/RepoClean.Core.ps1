# --- RepoClean.Core.ps1 ---

function Get-RelativePath {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Child
    )
    $root  = ($Root  -replace '/','\').TrimEnd('\')
    $child =  $Child -replace '/','\'
    try {
        $uRoot  = [Uri]($root + '\')
        $uChild = [Uri]$child
        if ($uRoot.IsBaseOf($uChild)) {
            return ($uRoot.MakeRelativeUri($uChild).ToString() -replace '/','\')
        }
    } catch { }
    if ($child.StartsWith($root, [StringComparison]::InvariantCultureIgnoreCase)) {
        return $child.Substring($root.Length).TrimStart('\')
    }
    return $child
}

function Get-ContentHash {
    param(
        [string]$Path,
        [string]$HashMode = 'sha256'
    )
    try {
        switch ($HashMode) {
            'sha256' { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash }
            'size'   { return (Get-Item -LiteralPath $Path).Length.ToString() }
            default  { return '' }
        }
    } catch { return '' }
}

function Get-InventoryRecords {
    param(
        [Parameter(Mandatory)][System.Collections.IEnumerable]$Files,
        [Parameter(Mandatory)][string]$RootPath,
        [string]$HashMode = 'sha256',
        [string[]]$Extensions = $script:targetExtensions
    )
    $records = [System.Collections.Generic.List[psobject]]::new()
    foreach ($file in $Files) {
        $fullPath = $file.FullName
        $relPath  = Get-RelativePath -Root $RootPath -Child $fullPath
        $size     = $file.Length
        $mtime    = $file.LastWriteTimeUtc
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
        $ext = [IO.Path]::GetExtension($fullPath)
        if ([string]::IsNullOrEmpty($ext)) { $ext = '' }
        $rec.Extension = $ext.ToLowerInvariant()

        # Skip zero-byte files and non-target extensions when hashing/dupe checking
        $shouldHash = $true
        if ($size -le 0 -or ($Extensions -and ($Extensions -notcontains $rec.Extension))) {
            $shouldHash = $false
        }

        $hash = ''
        if ($shouldHash) {
            $hash = Get-ContentHash -Path $fullPath -HashMode $HashMode
        }
        $rec.Hash = $hash

        [void]$records.Add($rec)
    }
    return $records
}
