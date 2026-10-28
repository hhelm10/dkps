#!/bin/bash
# Reproduce the real-data results: query-efficient evaluation (Result 1) and
# matrix completion (Result 2) on the 18-task, 93-model suite, plus the per-cell
# conditional dumps and the paper figures. Run from projects/pkps/scripts/.
# Outputs: results-pkps-rd1/ (query efficiency) and results-pkps-unified/ (completion).
set -e
cd "$(dirname "$0")"
SECONDS=0
step () { echo; echo "===== $1 (+${SECONDS}s) ====="; }

# ---- Result 1: query-efficient evaluation -----------------------------------
step "query efficiency -- budget / cohort / coverage sweeps"
pixi run python experiments/query_efficiency.py --sweep budget   --n_seeds 16
pixi run python experiments/query_efficiency.py --sweep n_models --n_seeds 16
pixi run python experiments/query_efficiency.py --sweep coverage --n_seeds 16
step "query efficiency -- per-cell conditional dump"
pixi run python experiments/dump_qe_conditional.py

# ---- Result 2: matrix completion --------------------------------------------
step "completion -- coverage / depth / tasks / cohort sweeps"
pixi run python experiments/completion.py --sweep coverage --n_seeds 16
pixi run python experiments/completion.py --sweep p_query  --n_seeds 16
pixi run python experiments/completion.py --sweep n_tasks  --n_seeds 16
pixi run python experiments/completion.py --sweep n_models --n_seeds 16
step "completion -- per-cell conditional dump"
pixi run python experiments/dump_completion_conditional.py

# ---- Figures ----------------------------------------------------------------
step "figures"
pixi run python figures/query_efficiency.py
pixi run python figures/qe_conditional.py
pixi run python figures/completion.py
pixi run python figures/completion_conditional.py
pixi run python figures/concept.py

echo; echo "===== DONE (+${SECONDS}s) ====="
