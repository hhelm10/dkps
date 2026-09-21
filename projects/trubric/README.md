# trubric — Embedding Agentic Traces for Efficient Evaluation

The paper project ("trubric" = *trace rubric*; formerly "qubric").
Everything the paper needs lives here:

```
projects/trubric/
  scripts/     one script per artifact (below) + trubric_common.py
  artifacts/   generated figures/tables (canonical names; git = versioning)
  writing/     paper.tex, references, style files, one-pagers
  data/        cached result JSONs the scripts render from
```

Each script writes its artifact under a canonical name (PNG preview +
vector PDF together; use the PDFs in the paper). Git history is the
version control.

| Artifact | Script | Compute stages |
|---|---|---|
| Table 1: hero MAE table | `trubric_table1_hero.py` | render-only (reads data/) |
| Fig 1: linear-probe radars | `trubric_figure1_radar.py` | render-only |
| Fig 2: PKPS geometry (n, m) | `trubric_figure2_pkps.py` | computes from embeddings (~1 min) |
| Fig 3: protocol grid | `trubric_figure3_protocols.py` | `compute\|raw\|render\|adaptive` |
| Fig 4: ablations | `trubric_figure4_ablations.py` | `nsweep\|regime20\|embedders\|render` |
| Fig 5: cost-to-achieve | `trubric_figure5_cost.py` | render-only |
| Fig 6: rubric sensitivity | `trubric_figure6_sensitivity.py` | full compute in `main()`; render from cache |

Scripts run from anywhere (`trubric_common.py` pins the cwd to the repo
root); heavy inputs (trace embeddings, panels) stay in the repo-level
`data/` (gitignored), and the estimator machinery lives in `scripts/`
and `dkps/`.

Protocol everywhere: leave-one-family-out, consensus centering,
per-draw pooled CV over sigma x {kNN, LOO-honest ridge-on-MDS},
B=50 shared draws, +/-1 SEM. Style canon: `scripts/hv_style.py`
(color = method; dashed = score-only, solid = uses embeddings;
marker fill = probe regime; solid/dashed = reference-library size
where swept).

## Building the paper

`writing/trubric.tex` is the paper; `trubric.pdf` is committed alongside
it. A pre-commit hook (`.githooks/pre-commit`) rebuilds the PDF whenever
the tex/bib/style sources are staged — enable it once per clone with
`git config core.hooksPath .githooks`.
