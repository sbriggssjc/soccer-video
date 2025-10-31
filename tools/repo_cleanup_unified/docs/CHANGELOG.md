# RepoClean Changelog

## v1 — Unified RepoClean introduced
- Initial consolidation of legacy cleanup/indexing scripts into `RepoClean.ps1` with Inventory, Plan, Execute, Index, and Doctor modes.
- Established `KeepRules.psd1` for configurable glob management without touching code.
- Added automatic discovery of historical scripts with exports to `out/inventory/existing_tools.csv` and the generated `docs/SCRIPT_COMPAT_MATRIX.md` compatibility matrix.
- Documented workflows in `README.md` and `USAGE_GUIDE.md`, including the safety model and recovery steps.
