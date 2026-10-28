# data

Shared local payloads, reused across the projects in `projects/` and never tracked
(only this README is committed). Rebuilt by the data-prep scripts in
`projects/pkps/helm/data/` (HELM dumps, EEE store, embedding parquets); override the
location with the `DKPS_DATA` environment variable.

- `helm/`        HELM response dumps + score TSVs
- `eee/`         Every Eval Ever datastore download (manifest, raw runs, parquets)
- `exports/`     embedding parquets for both suites (Gemini)
- `cache/`       dkps.embed disk cache (linked from projects/pkps/helm/.cache)
- `multi-embed/` alternative-embedding parquets (ablations)
