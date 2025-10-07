# Repository Indexing Utilities

To generate the repository inventory and analysis artifacts from Windows, run:

```powershell
cd C:\Users\scott\soccer-video
pwsh -File tools\index_repo.ps1 -Root .
```

Add `-IncludeOutputs` if you want to include large media/output folders in the
scan. All artifacts will be written to the `out\` folder alongside
`out\index_repo.log`.

