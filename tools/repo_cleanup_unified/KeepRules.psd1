@{
  KeepGlobs = @(
    '**/Game * - */**/*MASTER*ordered*.mp4',
    '**/Game * - */**/master*/**/*.mp4',
    '**/season_index/**/*.csv',
    '**/index/**/*.csv',
    '**/*_master_index.csv',
    '**/*_master_index.json',
    '**/*game_log*.csv',
    '**/out/atomic_clips/**/*.mp4',
    '**/atomic_clips/**/*.mp4',
    '**/out/portrait_reels/clean/**/*.mp4',
    '**/out/postable/**/ball_follow/**/*.mp4',
    '**/out/**/polished/**/ball_follow/**/*.mp4'
  );

  SidecarExts = @('.csv','.json','.srt','.vtt','.txt');

  RemoveGlobs = @(
    '**/_trash/**','**/_tmp/**','**/*.tmp','**/*.bak','**/*.old','**/*.partial','**/*.crdownload',
    '**/*__tmp.mp4','**/*__work*.mp4','**/stabilized/**','**/vidstab/**','**/esrgan/**','**/upscaled/**',
    '**/debug/**','**/scratch/**','**/drafts/**','**/test_outputs/**','**/portrait_reels/raw/**',
    '**/portrait_reels/wip/**','**/portrait_reels/alt/**'
  );
}
