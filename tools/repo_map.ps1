<# 
  repo_map.ps1 — inventory your toolkit and capture CLI help (PowerShell 5.1 safe)
  Outputs:
    - out\repo_map.md
    - out\repo_map_commands.csv
#>

param(
  [string]$Root = ".",
  [string]$OutDir = ".\out"
)

$ErrorActionPreference = "Stop"

# --- setup ---
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$md  = Join-Path $OutDir "repo_map.md"
$csv = Join-Path $OutDir "repo_map_commands.csv"

# --- collect files ---
$files = Get-ChildItem $Root -Recurse -File -ErrorAction SilentlyContinue
$sizeGB = [math]::Round(($files | Measure-Object Length -Sum).Sum / 1GB, 3)

$py    = $files | Where-Object { $_.Extension -eq ".py" } | Sort-Object FullName
$ps    = $files | Where-Object { $_.Extension -in @(".ps1",".psm1") } | Sort-Object FullName
$yaml  = $files | Where-Object { $_.Extension -in @(".yml",".yaml") } | Sort-Object FullName
$readmes = $files | Where-Object { $_.Name -like "README*" } | Sort-Object FullName

# --- helpers ---
function Try-Run([string]$CmdLine, [int]$TimeoutSec = 25) {
  try {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "powershell.exe"
    $psi.Arguments = "-NoProfile -Command " + $CmdLine
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute = $false
    $p = [System.Diagnostics.Process]::Start($psi)
    if (-not $p.WaitForExit($TimeoutSec * 1000)) { $p.Kill() | Out-Null }
    $out = $p.StandardOutput.ReadToEnd()
    if (-not $out) { $out = $p.StandardError.ReadToEnd() }
    return $out
  } catch {
    return "ERROR running: " + $CmdLine + "`n" + $_.Exception.Message
  }
}

function MDCode([string]$s) {
  # avoid the PowerShell backtick entirely; just return the raw text or wrap with simple quotes
  return "'" + $s + "'"
}

# --- discover console scripts (lightweight) ---
$consoleScripts = @()

# Try to read pyproject.toml for [project.scripts]
$pyproject = Get-ChildItem $Root -Recurse -Filter "pyproject.toml" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pyproject) {
  $txt = Get-Content -Raw $pyproject.FullName
  $lines = $txt -split "`r?`n"
  $inBlock = $false
  foreach ($line in $lines) {
    $t = ($line -replace "#.*","").Trim()
    if ($t -match '^\[project\.scripts\]$') { $inBlock = $true; continue }
    if ($inBlock -and $t -match '^\[') { $inBlock = $false }
    if ($inBlock -and $t -match '^\s*([\w\-]+)\s*=') {
      $name = $Matches[1]
      if ($name) { $consoleScripts += $name }
    }
  }
}

# Your main CLI often named 'soccerhl'; include it as a guess
if ($consoleScripts -notcontains "soccerhl") { $consoleScripts += "soccerhl" }

# de-dupe
$consoleScripts = $consoleScripts | Where-Object { $_ } | Sort-Object -Unique

# --- collect help text ---
$cmdRows = New-Object System.Collections.Generic.List[object]
$helpMap = @{}

foreach ($cmd in $consoleScripts) {
  $h = Try-Run ($cmd + " --help")
  if ($h -and $h -notmatch 'is not recognized') {
    $helpMap[$cmd] = $h

    # parse subcommands from a "Commands:" section (best effort)
    $subs = @()
    $seen = $false
    foreach ($line in ($h -split "`n")) {
      if ($line -match '^\s{0,4}Commands?:\s*$') { $seen = $true; continue }
      if ($seen) {
        if ($line -match '^\s{0,6}([A-Za-z0-9\-_\.]+)\s{2,}(.+)$') {
          $subs += [pscustomobject]@{ sub = $Matches[1]; desc = $Matches[2] }
        } elseif ($line -match '^\s{0,6}([A-Za-z0-9\-_\.]+)\s*$') {
          $subs += [pscustomobject]@{ sub = $Matches[1]; desc = "" }
        } elseif ($line -match '^\s*$') {
          # blank keeps scanning
        }
      }
    }

    if ($subs.Count -eq 0) {
      # fallback common set
      $subs = @(
        [pscustomobject]@{ sub='detect'; desc='detect highlight windows' },
        [pscustomobject]@{ sub='shrink'; desc='tighten windows around peaks' },
        [pscustomobject]@{ sub='clips';  desc='export per-window MP4s' },
        [pscustomobject]@{ sub='topk';   desc='rank and pick best plays' },
        [pscustomobject]@{ sub='reel';   desc='render final reel' }
      )
    }

    foreach ($s in $subs) {
      $subHelp = Try-Run ($cmd + " " + $s.sub + " --help")
      if ($subHelp -and $subHelp -notmatch 'is not recognized') {
        $helpMap[$cmd + " " + $s.sub] = $subHelp
      }
      $cmdRows.Add([pscustomobject]@{
        command    = $cmd
        subcommand = $s.sub
        description= $s.desc
      })
    }
  }
}

# --- scan python files for CLI hints ---
$entryRows = New-Object System.Collections.Generic.List[object]
foreach ($p in $py) {
  $head = ""
  try { $head = (Get-Content $p.FullName -TotalCount 400 -ErrorAction SilentlyContinue) -join "`n" } catch {}
  $hasMain  = ($head -match '__main__')
  $hasArgp  = ($head -match '\bargparse\b')
  $hasClick = ($head -match '\bclick\b')
  $hasTyper = ($head -match '\btyper\b')
  if ($hasMain -or $hasArgp -or $hasClick -or $hasTyper) {
    $entryRows.Add([pscustomobject]@{
      path     = $p.FullName
      argparse = $hasArgp
      click    = $hasClick
      typer    = $hasTyper
      main     = $hasMain
    })
  }
}

# --- write CSV of commands ---
$cmdRows | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8

# --- build markdown ---
$sb = New-Object System.Text.StringBuilder
$null = $sb.AppendLine("# Repository Map")
$null = $sb.AppendLine("")
$null = $sb.AppendLine("*Generated: " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "*")
$null = $sb.AppendLine("")
$null = $sb.AppendLine("Files: " + $files.Count + "  •  Size: " + $sizeGB + " GB")
$null = $sb.AppendLine("")
$null = $sb.AppendLine("## Key Files")
$null = $sb.AppendLine("")

if ($readmes -and $readmes.Count -gt 0) {
  $null = $sb.AppendLine("READMEs")
  foreach ($r in $readmes) {
    $rel = Resolve-Path $r.FullName -Relative
    $null = $sb.AppendLine("- " + (MDCode($rel)))
  }
  $null = $sb.AppendLine("")
}

if ($yaml -and $yaml.Count -gt 0) {
  $null = $sb.AppendLine("YAML / config")
  foreach ($y in ($yaml | Sort-Object FullName)) {
    $rel = Resolve-Path $y.FullName -Relative
    $null = $sb.AppendLine("- " + (MDCode($rel)))
  }
  $null = $sb.AppendLine("")
}

$null = $sb.AppendLine("PowerShell scripts")
foreach ($f in $ps) {
  $rel = Resolve-Path $f.FullName -Relative
  $null = $sb.AppendLine("- " + (MDCode($rel)))
}
$null = $sb.AppendLine("")

$null = $sb.AppendLine("Python files with CLI hints (argparse/click/typer or __main__)")
foreach ($e in $entryRows) {
  $rel = Resolve-Path $e.path -Relative
  $null = $sb.AppendLine("- " + (MDCode($rel)) + "  (argparse=" + $e.argparse + ", click=" + $e.click + ", typer=" + $e.typer + ", main=" + $e.main + ")")
}
$null = $sb.AppendLine("")

# --- Discovered Console Scripts ---
$null = $sb.AppendLine("## Discovered Console Scripts")
foreach ($cmd in $consoleScripts) {
  $null = $sb.AppendLine("")
  $null = $sb.AppendLine("### " + $cmd)
  $null = $sb.AppendLine("")

  if ($helpMap.ContainsKey($cmd)) {
    $null = $sb.AppendLine("~~~~")
    $null = $sb.AppendLine($helpMap[$cmd].Trim())
    $null = $sb.AppendLine("~~~~")
  }

  $subs = $cmdRows | Where-Object { $_.command -eq $cmd }
  if ($subs -and $subs.Count -gt 0) {
    $null = $sb.AppendLine("")
    $null = $sb.AppendLine("#### Subcommands")
    foreach ($s in $subs) {
      $null = $sb.AppendLine("")
      $desc = if ($s.description) { $s.description } else { "" }
      $null = $sb.AppendLine("**" + $s.subcommand + "** - " + $desc)

      $subKey = $cmd + " " + $s.subcommand
      if ($helpMap.ContainsKey($subKey)) {
        $null = $sb.AppendLine("")
        $null = $sb.AppendLine("~~~~")
        $null = $sb.AppendLine($helpMap[$subKey].Trim())
        $null = $sb.AppendLine("~~~~")
      }
    }
  }
}

# --- Recently Modified ---
$null = $sb.AppendLine("")
$null = $sb.AppendLine("## Recently Modified (last 7 days)")
$recent = $files | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) } |
  Sort-Object LastWriteTime -Descending | Select-Object -First 40

foreach ($f in $recent) {
  $rel = Resolve-Path $f.FullName -Relative
  $kb  = [math]::Round($f.Length/1KB, 1)
  $null = $sb.AppendLine("- " + $f.LastWriteTime.ToString('yyyy-MM-dd HH:mm') + "  " + $rel + "  (" + $kb + " KB)")
}

# --- write it ---
[IO.File]::WriteAllText($md, $sb.ToString(), [Text.Encoding]::UTF8)

"OK: wrote " + $md
"OK: wrote " + $csv
