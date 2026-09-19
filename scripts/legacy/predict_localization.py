"""Query-efficient prediction of gold-file localization from DKPS geometry.

The paper's experiment, on agentic traces: hold out one model; embed its traces
on a random subset of n queries jointly with the reference models' cached
traces; predict its benchmark score (mean gold-patch file localization) by
distance-weighted kNN over the reference models in the perspective space;
report MAE over held-out models and Monte Carlo query subsets.

  - error vs n_queries: all reference models, growing query budget
  - error vs n_models: fixed query budgets, growing reference pool

Baseline: direct evaluation -- the held-out model's localization computed on
the same n sampled queries.

kNN uses the DKPS distance matrix directly; CMDS is distance-preserving, so
kNN on perspective coordinates and kNN on the distance matrix agree, and
skipping CMDS makes the Monte Carlo loop ~instant. Per-query squared-distance
matrices are precomputed once per embedding config, so a subset's model
distance matrix is a sum-and-sqrt away.

Run scripts/embed_traces.py first. Usage:
    python scripts/predict_localization.py [--n-repeats 500] [--figdir figures]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import (TraceEmbedder, file_localization, load_langfuse_corpus,
                         load_swebench_gold_files, rms_scale)

CONFIGS = {
    'action': ['action'],
    'step_text': ['step_text'],
    'outcome': ['outcome'],
    'scalar': ['scalar'],
    'whole trace (naive)': ['whole'],
    'all structured': ['action', 'step_text', 'outcome', 'scalar'],
    'structured + whole': ['action', 'step_text', 'outcome', 'scalar', 'whole'],
}


def gold_files_cached(query_ids, path='data/gold_files.json'):
    if os.path.exists(path):
        with open(path) as f:
            return {k: set(v) for k, v in json.load(f).items()}
    gold = load_swebench_gold_files(query_ids)
    with open(path, 'w') as f:
        json.dump({k: sorted(v) for k, v in gold.items()}, f)
    return gold


def per_query_sq_dists(traces, X, models, queries):
    """PD[q]: (n_models, n_models) squared distances between models' mean
    embeddings on query q. Subset distance: sqrt(PD[S].sum(0)) / sqrt(|S|)."""
    cell = defaultdict(list)
    for tr, x in zip(traces, X):
        cell[(tr.model_id, tr.query_id)].append(x)
    C = np.array([[np.mean(cell[(m, q)], axis=0) for q in queries] for m in models])
    PD = np.zeros((len(queries), len(models), len(models)))
    for qi in range(len(queries)):
        diff = C[:, qi, None, :] - C[None, :, qi, :]
        PD[qi] = (diff ** 2).sum(-1)
    return PD


def knn_predict(dist_row, y_ref, k=3):
    """Distance-weighted kNN regression from one model's distances to references."""
    k = min(k, len(y_ref))
    nn = np.argsort(dist_row)[:k]
    w = 1.0 / (dist_row[nn] + 1e-12)
    return float(np.dot(w, y_ref[nn]) / w.sum())


def subset_dist(PD, S):
    return np.sqrt(PD[S].sum(axis=0) / len(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces-root', default='data/traces')
    ap.add_argument('--cache-dir', default='.dkps_cache')
    ap.add_argument('--figdir', default='figures')
    ap.add_argument('--n-repeats', type=int, default=500)
    ap.add_argument('--knn', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    traces = load_langfuse_corpus(args.traces_root)
    models = sorted({t.model_id for t in traces})
    queries = sorted({t.query_id for t in traces})
    M, Q = len(models), len(queries)

    gold = gold_files_cached(queries)
    loc_cell = defaultdict(list)
    for t in traces:
        loc_cell[(t.model_id, t.query_id)].append(file_localization(t, gold))
    L = np.array([[np.mean(loc_cell[(m, q)]) for q in queries] for m in models])
    y = L.mean(axis=1)
    print('true localization scores:')
    for m, v in sorted(zip(models, y), key=lambda t: -t[1]):
        print(f'  {m:45s} {v:.3f}')

    blocks = TraceEmbedder(cache_dir=args.cache_dir).transform_channels(traces)
    PDs = {name: per_query_sq_dists(
               traces, np.hstack([rms_scale(blocks[c]) for c in chans]),
               models, queries)
           for name, chans in CONFIGS.items()}

    # ---- error vs n_queries (all 13 reference models) -------------------
    ref_mask = ~np.eye(M, dtype=bool)
    n_queries_grid = range(1, Q + 1)
    mae_q = {name: [] for name in PDs}
    mae_q['direct eval'] = []
    for n in n_queries_grid:
        subsets = [rng.choice(Q, n, replace=False) for _ in range(args.n_repeats)] \
            if n < Q else [np.arange(Q)]
        errs = {name: [] for name in mae_q}
        for S in subsets:
            for name, PD in PDs.items():
                D = subset_dist(PD, S)
                errs[name].extend(
                    abs(knn_predict(D[i][ref_mask[i]], y[ref_mask[i]], args.knn) - y[i])
                    for i in range(M))
            errs['direct eval'].extend(abs(L[i, S].mean() - y[i]) for i in range(M))
        for name in mae_q:
            mae_q[name].append(np.mean(errs[name]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, vals in mae_q.items():
        style = dict(lw=2.5, color='k', ls='--') if name == 'direct eval' else \
                dict(lw=2.5) if name in ('all structured', 'whole trace (naive)') else \
                dict(lw=1, alpha=0.6)
        ax.plot(list(n_queries_grid), vals, label=name, **style)
    ax.set_xlabel('number of queries'); ax.set_ylabel('MAE of predicted localization')
    ax.set_title(f'Error vs query budget (13 reference models, kNN k={args.knn}, '
                 f'{args.n_repeats} subsets)')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.figdir, 'error_vs_n_queries.png'), dpi=150)

    print('\nMAE by config (n_queries = 1 / 3 / 6 / 12):')
    for name, vals in mae_q.items():
        print(f'  {name:22s} {vals[0]:.4f} / {vals[2]:.4f} / {vals[5]:.4f} / {vals[-1]:.4f}')

    # ---- error vs n_models (fixed query budgets) ------------------------
    budgets = [3, 6, Q]
    n_models_grid = range(2, M)          # size of the reference pool
    fig, axes = plt.subplots(1, len(budgets), figsize=(15, 5), sharey=True)
    print('\nMAE vs n reference models:')
    for ax, nq in zip(axes, budgets):
        mae_m = {name: [] for name in PDs}
        for n_ref in n_models_grid:
            errs = {name: [] for name in PDs}
            for _ in range(args.n_repeats):
                S = rng.choice(Q, nq, replace=False) if nq < Q else np.arange(Q)
                for name, PD in PDs.items():
                    D = subset_dist(PD, S)
                    i = rng.integers(M)
                    refs = rng.choice([j for j in range(M) if j != i], n_ref,
                                      replace=False)
                    errs[name].append(
                        abs(knn_predict(D[i][refs], y[refs], args.knn) - y[i]))
            for name in mae_m:
                mae_m[name].append(np.mean(errs[name]))
        direct = np.mean([abs(L[i, rng.choice(Q, nq, replace=False)].mean() - y[i])
                          for i in range(M) for _ in range(args.n_repeats)]) \
            if nq < Q else 0.0
        for name, vals in mae_m.items():
            style = dict(lw=2.5) if name in ('all structured', 'whole trace (naive)') \
                else dict(lw=1, alpha=0.6)
            ax.plot(list(n_models_grid), vals, label=name, **style)
        ax.axhline(direct, color='k', ls='--', lw=2.5, label='direct eval')
        ax.set_title(f'query budget = {nq}')
        ax.set_xlabel('number of reference models')
        best = min(mae_m, key=lambda n: mae_m[n][-1])
        print(f'  budget {nq:2d}: best config at 13 refs = {best} '
              f'({mae_m[best][-1]:.4f}); direct eval = {direct:.4f}')
    axes[0].set_ylabel('MAE of predicted localization')
    axes[-1].legend(fontsize=8)
    fig.suptitle(f'Error vs reference-model pool (kNN k={args.knn})')
    fig.tight_layout()
    fig.savefig(os.path.join(args.figdir, 'error_vs_n_models.png'), dpi=150)
    print(f'\nfigures written to {args.figdir}/')


if __name__ == '__main__':
    main()
