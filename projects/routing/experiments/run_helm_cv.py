"""HELM routing with leak-free CV bandwidth selection and a finer grid.

Compares, per seed: the finer fixed-sigma grid, cv-global (one sigma chosen
on anchor pseudo-evals), and cv-per-query (sigma chosen per eval query from
nearby anchors' pseudo-errors), against static / random / oracle.

Run from repo root: pixi run python -m projects.routing.experiments.run_helm_cv
"""

import os

import numpy as np
import pandas as pd

from .bandwidth_cv import (pseudo_error_table, select_cv_global,
                           select_cv_per_query)
from .evaluate import sigma_grid, stratified_split
from .geometry import pairwise_query_dist_tensor, rbf_weights, weighted_dist_matrix
from .router import route_nearest
from .run_helm import RESULTS, load_paired_core

FRACTIONS = (0.06, 0.09, 0.125, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0)


def run_seed(P, query_emb, categories, seed, eval_frac=0.2):
    m, n, _ = P.shape
    rng = np.random.default_rng(seed)
    anchor_idx, eval_idx = stratified_split(categories, eval_frac, rng)
    P_anchor, anchor_emb = P[anchor_idx], query_emb[anchor_idx]

    named = sigma_grid(anchor_emb, FRACTIONS, rng)
    names, sigmas = list(named), list(named.values())
    med = sigmas[FRACTIONS.index(1.0)]

    # pseudo-eval error table on anchors (leak-free selection signal)
    V, table = pseudo_error_table(P_anchor, anchor_emb, sigmas, rng=rng)
    gi = select_cv_global(table)
    pq = select_cv_per_query(table, anchor_emb[V], query_emb[eval_idx],
                             meta_sigma=0.5 * med)

    # geometries for the real eval queries at every sigma
    D_qa = {}
    for name, s in named.items():
        W = rbf_weights(anchor_emb, query_emb[eval_idx], s)
        D_qa[name] = weighted_dist_matrix(P_anchor, W)
    D_static = weighted_dist_matrix(P_anchor, np.ones(len(anchor_idx)))

    E = np.sqrt(P[eval_idx])
    rows = []
    eye = np.eye(n, dtype=bool)
    for qi, q in enumerate(eval_idx):
        Eq = E[qi]
        off = Eq[~eye].reshape(n, n - 1)
        rand_err, oracle_err = off.mean(axis=1), off.min(axis=1)
        for t in range(n):
            picks = {'static': route_nearest(D_static, t),
                     'cv-global': route_nearest(D_qa[names[gi]][qi], t),
                     'cv-per-query': route_nearest(D_qa[names[pq[qi]]][qi], t)}
            for name in names:
                picks[name] = route_nearest(D_qa[name][qi], t)
            for method, r in picks.items():
                rows.append((seed, q, categories[q], t, method, float(Eq[t, r])))
            rows.append((seed, q, categories[q], t, 'random', float(rand_err[t])))
            rows.append((seed, q, categories[q], t, 'oracle', float(oracle_err[t])))
    return pd.DataFrame(rows, columns=['seed', 'query', 'category', 'target',
                                       'method', 'error'])


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X, query_emb, categories, models = load_paired_core()
    P = pairwise_query_dist_tensor(X)
    print(f'P {P.shape}')

    df = pd.concat([run_seed(P, query_emb, categories, s) for s in range(5)],
                   ignore_index=True)
    df.to_parquet(os.path.join(RESULTS, 'helm_cv_errors.parquet'))

    per_seed = df.groupby(['method', 'seed'])['error'].mean().unstack()
    summary = pd.DataFrame({
        'mean_error': per_seed.mean(axis=1),
        'se_over_seeds': per_seed.std(axis=1) / np.sqrt(per_seed.shape[1]),
    }).sort_values('mean_error')
    summary.to_csv(os.path.join(RESULTS, 'helm_cv_summary.csv'))
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
