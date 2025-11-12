@{
  KeepTopLevel = @(
    'atomic_clips','branded_clips','masters','reels','portrait_1080x1920',
    'WIDE','tiktok','games','inventory','misc','_tmp','_quarantine'
  )

  # Glob patterns to ban from portrait reels
  BannedPortrait = @('*__DEBUG__*','*_x2.mp4','*\unique_from_upscaled\*','*FOLLOW_SMARTAUDIO_TWEAK*')

  Prune = @{
    RootBackupsKeep = 3            # keep last N root_backups_*
    EmptyDirs       = $true
    TmpDaysOld      = 7            # delete files older than this in _tmp
    QuarantineDays  = 14           # auto-delete quarantine leftovers older than this
  }
}
