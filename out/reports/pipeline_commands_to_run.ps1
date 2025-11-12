param(
  [switch]$Run,          # when present, actually execute commands
  [switch]$DryRun        # optional; forces dry-run even if -Run is used
)

# Decide execution mode (default = dry-run)
$Execute = $false
if ($Run.IsPresent -and -not $DryRun.IsPresent) { $Execute = $true }
$ErrorActionPreference = 'Stop'
$script:RepoRoot = '/workspace/soccer-video'
$logPath = Join-Path $script:RepoRoot 'out\logs\advance_pipeline.20251112_214230.log'
$null = New-Item -ItemType Directory -Force -Path (Split-Path $logPath)
$logStream = New-Item -ItemType File -Force -Path $logPath
$commands = @()
if ($commands.Count -eq 0) { Write-Host 'No pending stages.'; return }
foreach ($stageGroup in ($commands | Sort-Object Stage, MatchKey, ClipID | Group-Object Stage)) {
  Write-Host ('### Stage: {0}' -f $stageGroup.Name) -ForegroundColor Cyan
  foreach ($matchGroup in ($stageGroup.Group | Group-Object MatchKey)) {
    Write-Host ('# Match: {0}' -f $matchGroup.Name) -ForegroundColor Yellow
    foreach ($entry in $matchGroup.Group) {
      $inPath = $entry.Input
      $outPath = $entry.Output
      $cmd = $entry.Command
      $cmdString = ($cmd | ForEach-Object { '"' + $_ + '"' }) -join ' ' 
      Add-Content -Path $logPath -Value $cmdString
      $outDir = Split-Path -Parent $outPath
      if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
      $shouldSkip = $false
      if ((Test-Path $inPath) -and (Test-Path $outPath)) {
        $inItem = Get-Item $inPath
        $outItem = Get-Item $outPath
        if ($outItem.LastWriteTimeUtc -ge $inItem.LastWriteTimeUtc) { $shouldSkip = $true }
      }
      if ($shouldSkip) {
        Write-Host ('[SKIP] {0} :: up-to-date' -f $cmdString)
        Add-Content -Path $logPath -Value '[SKIP] up-to-date'
        continue
      }
      Write-Host ('[RUN ] {0}' -f $cmdString)
      if (-not $Execute) {
        Write-Host '       DryRun: not executed.' -ForegroundColor Green
        Add-Content -Path $logPath -Value '[DryRun] not executed'
        continue
      }
      $psi = New-Object System.Diagnostics.ProcessStartInfo
      $psi.FileName = $cmd[0]
      if ($cmd.Count -gt 1) { $psi.Arguments = [string]::Join(' ', ($cmd[1..($cmd.Count-1)] | ForEach-Object { '"' + $_ + '"' })) }
      $psi.WorkingDirectory = $script:RepoRoot
      $psi.UseShellExecute = $false
      $psi.RedirectStandardOutput = $true
      $psi.RedirectStandardError = $true
      $proc = [System.Diagnostics.Process]::Start($psi)
      $stdout = $proc.StandardOutput.ReadToEnd()
      $stderr = $proc.StandardError.ReadToEnd()
      $proc.WaitForExit()
      if ($stdout) { Add-Content -Path $logPath -Value $stdout }
      if ($stderr) { Add-Content -Path $logPath -Value $stderr }
      if ($proc.ExitCode -ne 0) {
        Write-Warning ('Command exited with code {0}' -f $proc.ExitCode)
        Add-Content -Path $logPath -Value ('ExitCode={0}' -f $proc.ExitCode)
      }
    }
  }
}
Write-Host ('Log: {0}' -f $logPath)
