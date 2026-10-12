# artifacts

Committed reports and evidence retained for review. Routine experiment outputs are
git-ignored under `experiments/*/results*` and `projects/pkps/scripts/results*`.

- `synthetic-report/` -- the original "Double-Kernel DKPS" synthetic study: report.md
  with its CSVs and figures (superseded by `projects/pkps/scripts/synthetic/` and the paper's
  Figure 2).
- `figures/` -- one-off analysis figures (EEE datastore paired-fraction).
- `stress-test/` -- evidence for the [PKPS stress-test revisions](../STRESS_TEST_REVISIONS.md):
  mathematical checks, manuscript excerpts, 16-seed EEE follow-ups, uncertainty
  audits, and a SHA256 inventory. Raw responses, embeddings, and model weights are
  excluded; the revisions document gives reproduction commands and limitations.
