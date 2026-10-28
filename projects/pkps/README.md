# pkps — Benchmark Prediction with Sparse and Unpaired Cached Responses

The PKPS paper project. Everything the paper needs lives here:

```
projects/pkps/
  scripts/     pipeline + experiments + figure scripts (run from scripts/);
               scripts/data/ holds the data-prep code, scripts/synthetic/ the
               synthetic study
  artifacts/   generated figures (canonical names, PDF + PNG preview;
               git history = versioning)
  writing/     main.tex mirror of the Overleaf source of truth, references,
               style files; writing/figures/ is the latex-facing copy of the
               paper figures
  data/        gzipped canonical copies of the result CSVs the figures and
               Table 1 render from (pandas reads .csv.gz directly); heavy
               inputs (embeddings, downloads) stay in the repo-level data/
               (gitignored, rebuilt by scripts/data/)
```

Each figure script writes into the local `scripts/results-*` directory it reads
from; the canonical CSV behind every published number is cached under `data/`.

| Artifact | Script (from `scripts/`) | Reads |
|---|---|---|
| Table 1 (hero table) | `validate_package.py` verifies every cell | `results-{pkps,eee}-rd1/rd1_suite_budget.csv`, `results-*-unified/completion_suite_coverage.csv` |
| Fig: HELM query efficiency | `figures/query_efficiency.py --suite helm` | `results-pkps-rd1/rd1_suite_{budget,n_models,coverage}.csv` |
| Fig: HELM QE conditional | `figures/qe_conditional.py` | `results-pkps-rd1/qe_cond_cells.csv` (built by `experiments/dump_qe_conditional.py`) |
| Fig: HELM completion | `figures/completion.py --suite helm` | `results-pkps-unified/completion_suite_*.csv` |
| Fig: HELM completion conditional | `figures/completion_conditional.py` | `results-pkps-unified/comp_cond_cells.csv` (built by `experiments/dump_completion_conditional.py`) |
| Fig: EEE query efficiency | `figures/query_efficiency.py --suite eee` | `results-eee-rd1/rd1_suite_*.csv` |
| Fig: EEE completion | `figures/completion.py --suite eee` | `results-eee-unified/completion_suite_*.csv` |
| Fig: sensitivity (both suites) | `figures/sensitivity.py` | `results-sens/{helm,eee}-{d4,d24,sig0..sig5}/` + main dirs |
| Fig: synthetic study | `scripts/synthetic/` (own README) | its own results/ |
| Fig: concept diagram | `figures/concept.py` | none (drawn) |

The result CSVs regenerate from `experiments/query_efficiency.py` and
`experiments/completion.py` (both suites; 16 seeds; see `scripts/README.md` for
the full sweep list). The base protocol is the joint response space: standardized
answer text, one shared PCA per suite whose dimension matches the per-dataset
total (HELM 92, EEE 83), unit-normalized, query kernel gating. The published
per-dataset protocols remain reachable
(`--resp_mode native` / `--response_space blocked`) and are validated bit-exact
by `validate_package.py --response_space blocked`.
