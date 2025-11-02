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
            Extension     = ([IO.Path]::GetExtension($fullPath) ?? '' | ForEach-Object { $_ })  # will be '' on PS5
            Codec         = $null
            Width         = $null
            Height        = $null
            Hash          = $null
            Status        = $null
            Notes         = $null
        }
        if ([string]::IsNullOrEmpty($rec.Extension)) { $rec.Extension = '' }
        else { $rec.Extension = $rec.Extension.ToLowerInvariant() }

        if ($HashCache) {
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
