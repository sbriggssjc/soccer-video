How to use:

1) Open the CSV and set 'master_start'/'master_end' (in seconds) for each row you want.
   - Keep 'id' as a 3-digit sequence (001, 002, …).
   - Fill 'playtag' / 'phase' / 'side' / etc. as needed for overlays & inventory.
   - Add or remove rows as needed — the 20 pre-filled rows are just a starting point.

2) Generate atomic clips (existing pipeline examples):
   powershell -ExecutionPolicy Bypass -File tools\Build-AtomicClipsFromCsv.ps1
     -Csv 'D:\Projects\soccer-video\out\catalog\2026-03-01__TSC_vs_North_OKC\events_selected.csv' -OutRoot 'D:\Projects\soccer-video\out\atomic_clips' -Brand 'tsc'

3) Add the newly created atomic events to inventory:
   powershell -ExecutionPolicy Bypass -File tools\Append-AtomicCsv-To-Inventory.ps1
     -Csv 'D:\Projects\soccer-video\out\catalog\2026-03-01__TSC_vs_North_OKC\events_selected.csv' -Inventory 'D:\Projects\soccer-video\out\inventory\atomic_events.csv'

Master: C:\Users\scott\Downloads\03.01.2026 - North OKC 2016 Black (Championship).mp4

4) Extract sideline-angle clips (same timestamps, second camera):
   First determine the sync offset by finding the same event (e.g. kickoff)
   in both the XBot Go master and the sideline stabilizer video, then compute:
     offset = sideline_timestamp - master_timestamp

   python tools\extract_sideline_angles.py ^
     --game 2026-03-01__TSC_vs_North_OKC ^
     --sideline "PATH_TO_SIDELINE_VIDEO.mp4" ^
     --offset <OFFSET_SECONDS> ^
     --dry-run

   Remove --dry-run to extract.  Clips land in:
     out\atomic_clips\2026-03-01__TSC_vs_North_OKC\sideline\

   Or batch via sideline_sources.csv:
     python tools\extract_sideline_angles.py --from-config --game North_OKC
