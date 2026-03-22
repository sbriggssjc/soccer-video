How to use:

1) Open the CSV and set 'master_start'/'master_end' (in seconds) for each row you want.
   - Keep 'id' as a 3-digit sequence (001, 002, …).
   - Fill 'playtag' / 'phase' / 'side' / etc. as needed for overlays & inventory.

2) Generate atomic clips (existing pipeline examples):
   powershell -ExecutionPolicy Bypass -File tools\Build-AtomicClipsFromCsv.ps1 
     -Csv 'C:\Users\scott\soccer-video\out\catalog\2025-11-04__TSC_-_Navy_Oct_2025\events_selected.csv' -OutRoot 'C:\Users\scott\soccer-video\out\atomic_clips' -Brand 'tsc'

3) Add the newly created atomic events to inventory:
   powershell -ExecutionPolicy Bypass -File tools\Append-AtomicCsv-To-Inventory.ps1 
     -Csv 'C:\Users\scott\soccer-video\out\catalog\2025-11-04__TSC_-_Navy_Oct_2025\events_selected.csv' -Inventory 'C:\Users\scott\soccer-video\out\inventory\atomic_events.csv'

Master: C:\Users\scott\soccer-video\out\masters\2025-11-04__TSC_-_Navy_Oct_2025\2025-11-04__TSC_-_Navy_Oct_2025__master.mp4
Duration: ~3990.22s
