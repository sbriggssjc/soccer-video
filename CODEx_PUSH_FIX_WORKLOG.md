# Git Push Large File Fix Worklog

## Scope
- Topic: GitHub push rejection on `main` due to oversized CSV report artifacts.
- Objective: Remove the oversized report files from the local-only commit history being pushed and prevent them from being recommitted.

## Current State
- Repository branch: `main`
- Ahead of remote: 2 commits ahead of `origin/main`
- Rejected paths confirmed in local `HEAD`:
  - `shared/reports/inventory/repo_inventory.FSRESET.csv` (`57,320,091` bytes)
  - `shared/reports/logs/dup_exact_by_hash.csv` (`60,301,002` bytes)
  - `shared/reports/inventory/orphans.csv` (`61,586,003` bytes)
  - `shared/reports/logs/dup_candidates_by_size.csv` (`137,416,164` bytes)

## Findings
- The large files were introduced in local-only commit `fefcab53` (`GPT changes.`).
- Commit `48285764` does not contain the blocked large report files.
- Because the large blobs are present in the outgoing history, a normal delete commit will not fix the push.

## Plan
- Add ignore rules for generated report artifacts under `shared/reports/`.
- Rewrite the two local-only commits so the oversized CSVs are removed from history before pushing.
- Verify no outgoing object still exceeds GitHub's hard limit.

## Progress Log
- 2026-03-22: Inspected `git status`, outgoing commits, and the tree sizes for the reported CSVs.
- 2026-03-22: Confirmed the oversized files are limited to the local-only commit range and identified the introducing commit.
- 2026-03-22: Created backup branch `codex/backup-main-largefile-fix-20260322` before rewriting history.
