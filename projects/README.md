# projects/

One self-contained directory per paper or deliverable. Repo-level code
(`scripts/`, `dkps/`) and heavy raw data (repo-level `data/`, gitignored)
are shared infrastructure; a project directory holds everything that is
*specific to the deliverable* — the per-artifact scripts, the rendered
artifacts, the writing, and the small cached result files the artifacts
are rendered from.

Current projects:

| Project | What it is |
|---|---|
| `trubric/` | Paper: "Task-efficient agent evaluation using cached traces" (ICLR 2027 submission). See `trubric/README.md`. |
| `vantablack/` | Detecting colluding subnetworks from an outside observer's view. See `vantablack/README.md`. |

## Standard project layout

```
projects/<name>/
  scripts/     one script per artifact + <name>_common.py
  artifacts/   generated figures/tables under canonical names
  writing/     paper .tex, references.bib, style files, committed PDF
  data/        small cached result JSONs that the scripts render from
```

Rules that make this work:

- **One script per artifact.** Naming:
  `<name>_figure{N}_{subtitle}.py`, `<name>_table{N}_{subtitle}.py`,
  `<name>_appendix{X}_{subtitle}.py`. The script is the single source of
  truth for its artifact; anyone should be able to regenerate any figure
  by running its script with no arguments.
- **`<name>_common.py`** pins the working directory to the repo root (so
  repo-relative data paths resolve no matter where the script is invoked
  from) and provides `save_artifact(fig, name)` — which writes both
  `artifacts/<name>.png` (200 dpi raster preview) and
  `artifacts/<name>.pdf` (vector, the one the paper includes) — plus
  `save_text_artifact` for generated `.tex`.
- **Canonical artifact names, no version suffixes.** Never `_v2`,
  `_final`, `_new`. Git history is the version control; an updated
  artifact overwrites the old file in place.
- **Compute/render separation.** Expensive stages write a JSON cache into
  the project's `data/`; the figure script's render path reads only that
  cache and is cheap to re-run (style tweaks never require recompute).
  The project `data/` caches are committed; raw inputs (trace archives,
  embedding matrices, panels) stay in the repo-level gitignored `data/`.
- **The paper PDF is committed and built by hook.** `.githooks/pre-commit`
  (enable once per clone: `git config core.hooksPath .githooks`) syncs
  `artifacts/*.pdf` and generated `.tex` into `writing/figures/`, rebuilds
  the paper (pdflatex → bibtex → pdflatex ×2), stages the result, and
  blocks the commit if LaTeX fails.
- **Compute etiquette.** One CPU-heavy job at a time; cap BLAS threads
  (`OPENBLAS_NUM_THREADS=...`) and run under `nice`. API keys live in the
  repo-level `.env` (never committed, never printed).

## Figure conventions ("Helivan Blues")

The style canon lives in `scripts/hv_style.py`. Every plotting script
starts with:

```python
import hv_style
hv_style.apply()          # rc params: white bg, muted spines/grid, ink colors
SZ = hv_style.SIZES       # all font sizes come from here — never hardcode
```

### Channel grammar

The core idea: **each visual channel encodes exactly one factor, and the
assignment is fixed across every figure in the paper.**

1. **Color = method.** One blue at many lightnesses plus two grays; the
   mapping never changes between figures:

   | Method (display name) | Role in `hv_style.ROLES` | Hex | Linestyle |
   |---|---|---|---|
   | sample score | `baseline_gray` | `#6b7280` | dashed |
   | IRT | `baseline_pale` | `#afbec6` | dashed |
   | raw trace | `comparator` | `#93aacc` | solid |
   | generic | `generic` | `#486884` | solid |
   | trubric | `focus` | `#3596ff` | solid |
   | IRT + trubric | `anchor` | `#114471`, 3.6 pt | solid |
   | *(reserve)* | `reserve` | `#6d5bd0` violet | — |

   Violet is reserved for one concept per paper — in trubric it colors
   the benchmark score $y$ (scatter colorbars), never a method. Color
   follows the method, never its rank in a particular panel, and default
   matplotlib palettes are never cycled.

2. **Linestyle = information source.** Dashed = score-only methods,
   solid = methods that use trace embeddings. Declared once, in the
   first figure's caption, and honored everywhere.

3. **A second swept factor gets its own channel and its own legend.**
   Marker fill encodes the task-selection regime (filled = adaptive,
   open = random). Where markers are unavailable, a figure-wide
   solid/dashed pair encodes the swept pair (e.g. n = 107 vs n = 20 in
   the sensitivity figure) and is declared in that figure's legend.

4. **Orthogonal factors → separate legends.** Never a single legend
   enumerating the cross product (method × regime × n); instead one
   compact legend per factor, placed together below the panels.

### Method display names

Exactly: `sample score`, `IRT`, `raw trace`, `generic`, `trubric`,
`IRT + trubric`. No parenthetical qualifiers ("(2PL)") and no "geometry"
suffix in legends — prose may say "trace geometry" after defining it.

### Layout and annotation

- Panels that invite comparison share fixed y-limits and, where one
  y-axis serves a row, interior tick labels are dropped.
- Uncertainty is a ±1 SEM shaded band (`alpha≈.13`), not error bars.
- Reference values are dotted `hv_style.REFLINE` lines with a small
  rotated inline label ("chance"); the full-benchmark point is a gold
  star (`#f59e0b`) placed off-axis with a cost annotation, not a line.
- Suptitles/labels/ticks/annotations take their sizes from
  `hv_style.SIZES` (suptitle 21.2 … annot 14.0). These were tuned for
  print legibility at `\textwidth`; bump the whole dict, not one label.
- Colormaps: `hv_style.CMAP_BLUE` (sequential, brand blue),
  `hv_style.CMAP_RESERVE` (violet ramp, reserved concept), and the
  registered `'helivan_div'` diverging map (blue = below center,
  Helivan gold = above; center with `TwoSlopeNorm`).

### Output and tables

- Always save through `save_artifact` → canonical `.png` + `.pdf` pair,
  tight bbox, `pad_inches=0.03`. The paper includes the PDFs.
- Tables are plain `table`/`tabular` with booktabs — no custom column
  commands. **Bold** = best method within a (benchmark, selection,
  budget) column; <u>underline</u> = significantly different from IRT
  under a two-sided Wilcoxon signed-rank test on paired per-system
  errors (α = 0.05). The caption defines both conventions.
- Every number printed in the paper must be reproducible from a
  committed cache in the project's `data/` by the artifact's script.
