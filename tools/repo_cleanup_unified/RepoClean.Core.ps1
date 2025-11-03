# --- RepoClean.Core.ps1 ---
# PS 5.1-safe relative path
function Get-RelativePath {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Child
    )
    $root  = ($Root  -replace '/','\').TrimEnd('\\')
    $child =  $Child -replace '/','\'

    try {
        $uRoot  = [Uri]($root + '\\')
        $uChild = [Uri]$child
        if ($uRoot.IsBaseOf($uChild)) {
            return ($uRoot.MakeRelativeUri($uChild).ToString() -replace '/','\')
        }
    } catch { }

    if ($child.StartsWith($root, [StringComparison]::InvariantCultureIgnoreCase)) {
        return $child.Substring($root.Length).TrimStart('\\')
    }
    return $child
}

# Always returns a growable List[psobject]
function Get-InventoryRecords {
    param(
        [Parameter(Mandatory)][System.Collections.IEnumerable]$Files,
        [Parameter(Mandatory)][string]$RootPath,
        [hashtable]$HashCache = $null,
        [string]$HashAlgo = 'MD5'
    )

    $records = [System.Collections.Generic.List[psobject]]::new()
    $targetExtensions = @('.mp4','.mov','.mkv','.avi','.m4v','.jpg','.jpeg','.png','.gif','.webp')

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
        $extension = [IO.Path]::GetExtension($fullPath)
        if ([string]::IsNullOrEmpty($extension)) { $extension = '' }
        else { $extension = $extension.ToLowerInvariant() }
        $rec.Extension = $extension

        # Skip zero-byte files and non-target extensions when hashing/dupe checking
        $shouldHash = $true
        if ($size -le 0 -or ($targetExtensions -notcontains $rec.Extension)) {
            $shouldHash = $false
        }

        if ($HashCache -and $shouldHash) {
            $cacheKey = "$relPath|$size|$mtime"
            if ($HashCache.ContainsKey($cacheKey)) {
                $rec.Hash = $HashCache[$cacheKey].Hash
            } else {
                try {
                    $hasher = [System.Security.Cryptography.HashAlgorithm]::Create($HashAlgo)
                    $fs = [IO.File]::OpenRead($fullPath)
                    try {
                        $bytes = $hasher.ComputeHash($fs)
                        $hash  = ($bytes | ForEach-Object { $_.ToString('x2') }) -join ''
                        $rec.Hash = $hash
                        $HashCache[$cacheKey] = [PSCustomObject]@{
                            Path          = $cacheKey
                            SizeBytes     = $size
                            LastWriteTime = $mtime
                            Hash          = $hash
                        }
                    } finally { $fs.Dispose() }
                } catch { }
            }
        }

        [void]$records.Add($rec)
    }

    return $records
}
