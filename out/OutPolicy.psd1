@{
  KeepTopLevel = @(
    'atomic_clips','branded_clips','masters','reels','portrait_1080x1920',
    'WIDE','tiktok','games','inventory','misc','_tmp','_quarantine'
  )

  # Glob patterns to ban from portrait reels
  BannedPortrait = @(
    '*__DEBUG__*',
    '*__x2.mp4'       # keep if you truly need x2; remove this if not
  )

  Prune = @{
    RootBackupsKeep = 2
    EmptyDirs       = $true
    TmpDaysOld      = 2    # delete _tmp files older than 2 days
    QuarantineDays  = 3    # delete quarantine files older than 3 days
  }
}
