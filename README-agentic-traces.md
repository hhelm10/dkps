# Transforming semantic embedding functions for agentic traces

*A trace representation should be faithful to content and invariant to
authorship.*

This branch (`agentic-traces`) extends DKPS ("Query-efficient model
evaluation using cached responses", arXiv:2605.07096) to agentic
evaluations, where each (system, task) pair produces a multi-step tool-call
*trace* rather than a short answer. The core method, **qubric**
(query-specific rubrics), re-represents a trace as an LLM judge's
rubric-structured description of it, embedded section-wise and centered
against the cross-system consensus.

Read `notes/summary.pdf` for the current write-up, `FINDINGS.md` for the
complete experimental log (F1–F26), `PAPER.md` for the exhibit plan, and
`RELATED.md` for the literature context.

## Layout

```
dkps/traces/qubric.py   the method as a 3-function API (see below)
dkps/traces/            loaders (leaderboard + Langfuse), canonicalization,
                        channels, local/API embedders
dkps/unpaired_dkps.py   PKPS (product-kernel perspective space)
scripts/data.py         corpus download (SWE-bench leaderboard, Terminal-Bench)
scripts/quench.py       query-efficient benchmarking results via qubric
scripts/pillars.py      the six property metrics (radar/heatmap inputs)
scripts/judge_*.py      judging runners (OpenRouter, escalation, matrices)
scripts/fig_*.py        figure generation (all figures/ regenerable)
notes/summary.tex       the write-up (pdflatex notes/summary.tex)
```

## Setup

```bash
pip install -e . numpy scipy graspologic "setuptools<81" requests \
    python-dotenv tqdm sentence-transformers datasets huggingface_hub \
    awscli matplotlib   # setuptools pin: graspologic still imports pkg_resources
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # any OpenAI-compatible endpoint works
```

LLM calls (rubric writing, extraction) go through OpenRouter by default —
the validated open judge is `deepseek/deepseek-chat-v3.1` (matches the
closed reference within panel resolution; see FINDINGS F22). Embeddings run
locally via sentence-transformers (default `nomic-ai/nomic-embed-text-v1.5`,
GPU recommended); no closed-API dependency anywhere in the pipeline.

## Data

```bash
# SWE-bench Verified leaderboard: official labels (fast, ~1 min)
python scripts/data.py swebench-labels

# trajectories + final patches from the public S3 bucket (anonymous;
# ~40GB total; --limit N for a smoke test)
python scripts/data.py swebench-trajs --limit 5
python scripts/data.py swebench-trajs

# Terminal-Bench 2 leaderboard corpus (~40GB, millions of small files)
python scripts/data.py terminalbench
```

This produces `data/leaderboard/verified_labels.json` (resolved lists +
metadata for every submission) and
`data/leaderboard/verified/<submission>/{trajs,all_preds.jsonl}`. Each
submission's `trajs/` is in that team's own format; `dkps/traces/leaderboard.py`
renders any of them to text (`render_text`, format-agnostic).

## The qubric API

```python
from dkps.traces import write_rubrics, grade_traces, embed_graded, consensus_center

# 1. task description(s) -> instance-specific rubric(s)   [Model 1]
rubrics = write_rubrics({"task1": problem_statement}, api_key,
                        "deepseek/deepseek-chat-v3.1")

# 2. rubric + trace(s) -> graded trace(s): six short factual
#    descriptions per trace                               [Model 2]
graded = grade_traces(rubrics["task1"], {"run_a": trace_text}, api_key,
                      "deepseek/deepseek-chat-v3.1")

# 3. graded trace(s) -> concatenated embedding (n, 6*d)
X = embed_graded(list(graded.values()), None, "nomic-ai/nomic-embed-text-v1.5")

# corpus-level: per-instance consensus centering (median over systems)
Xc = consensus_center(X, instance_ids)
```

`grade_traces` accepts one rubric for many traces, or per-task rubrics with
`task_ids` alignment. `embed_graded` runs sentence-transformers models
locally (`api_key=None`) or any OpenAI-compatible `/embeddings` endpoint.

## Reproducing the results

All numbers derive from cached judge outputs under `data/judge/` (built by
`scripts/judge_openrouter.py` / `scripts/smallcohort_judge.py`; ~$1–12 per
judge sweep via OpenRouter).

```bash
# Query-efficient benchmarking: error vs probe budget m
# (geometry / sample score / honest ensemble, leave-one-LLM-out)
python scripts/quench.py

# The six property metrics for raw vs qubric representations
python scripts/pillars.py

# Judge x construction matrix; property heatmap; figures
python scripts/judge_matrix.py && python scripts/fig_judge_matrix.py
python scripts/fig_heatmap.py
```

Evaluation protocol notes that matter: cross-validation is
**leave-one-LLM-out** (excluding same-`model_display` references; naive CV
inflates accuracy ~25% via model-family leakage), and per-instance
consensus centering is part of the representation. See FINDINGS §2 for the
full protocol and F17/F18 for the documented overfitting traps.

## Figures

Every figure in `notes/summary.pdf` regenerates from committed data:

| figure | script | data |
|---|---|---|
| sensitivity radars | `scripts/fig_radar.py` | `figures/radar_all_data_v2.json`, `figures/family_metric.json` |
| property heatmap | `scripts/fig_heatmap.py` | same + `figures/stability_column.json` |
| QUENCH pair | `scripts/fig_quench.py` | `figures/pipeline_final.json`, `figures/qspec_fig_data_pkps.json`, `figures/quench_n.json` |

Also available: `scripts/fig1_schematic.py` (methods schematic),
`scripts/fig_judge_matrix.py` (judge x construction matrix). To recompute the
underlying numbers rather than redraw: `scripts/pillars.py` (radar/heatmap
metrics), `scripts/quench.py` (QUENCH curves), `scripts/judge_matrix.py`
(matrix cells), `scripts/stability_retest.py` + `scripts/stability_column.py`
(stability measurements).

## Shared data artifacts

Everything expensive or impossible to recreate ships as tarballs (~2.6GB;
ask HH for the transfer location). From the repo root:

```bash
python scripts/data.py unpack-artifacts /path/to/share
```

This restores `data/judge/` (all LLM judge outputs -- rubrics, extractions
for 6+ judges, retests, small-cohort judging, rendered trace texts,
embedding caches), `.dkps_cache_lb/` (OpenAI embedding caches --
irreplaceable), `data/multiembed_*.npz`, and the labels file. With these
unpacked, every script above runs without any API key; the raw 40GB
trajectory corpus is only needed to extend to new instances
(`python scripts/data.py swebench-trajs`).

## Documents

- `notes/summary.pdf` — the working two-pager + worked example appendix
- `FINDINGS.md` — numbered findings F1–F26 with exact numbers and recipes
- `PAPER.md` — figure/table plan and evidence ledger
- `NARRATIVE.md` — the project story
- `RELATED.md` — related-work narrative + annotated bibliography
