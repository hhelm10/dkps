"""Held-out-query evaluation of query-aware routing.

Protocol: split queries into an anchor set (the cached history all geometries
are built from) and an evaluation set (the "new" queries). For each evaluation
query q* and each target model t, each method picks a candidate r != t; its
mimicry error is ||x_{r,q*} - x_{t,q*}|| in response-embedding space. The
target's own response is the metric's zero, so baseline (1) is the floor by
construction and is reported implicitly.

Methods:
    qa-sigma-*  : nearest to target in the query-weighted geometry (RBF, sigma)
    static      : nearest to target in the uniform-weight (DKPS) geometry
    random      : expected error of a uniform-random candidate
    oracle      : hindsight-best candidate (lower bound for any router)
"""

import numpy as np
import pandas as pd

from .geometry import rbf_weights, weighted_dist_matrix
from .router import route_nearest


def stratified_split(categories, eval_frac, rng):
    """Return (anchor_idx, eval_idx), stratified by category."""
    categories = np.asarray(categories)
    eval_idx = []
    for c in np.unique(categories):
        idx = np.flatnonzero(categories == c)
        k = max(1, int(round(eval_frac * len(idx))))
        eval_idx.append(rng.choice(idx, size=k, replace=False))
    eval_idx = np.sort(np.concatenate(eval_idx))
    anchor_idx = np.setdiff1d(np.arange(len(categories)), eval_idx)
    return anchor_idx, eval_idx


def sigma_grid(anchor_query_emb, fractions=(0.125, 0.25, 0.5, 1.0, 2.0), rng=None, n_pairs=20000):
    """Sigmas as fractions of the median pairwise anchor-query distance."""
    m = len(anchor_query_emb)
    rng = rng or np.random.default_rng(0)
    i = rng.integers(0, m, size=n_pairs)
    k = rng.integers(0, m, size=n_pairs)
    keep = i != k
    med = np.median(np.linalg.norm(
        anchor_query_emb[i[keep]] - anchor_query_emb[k[keep]], axis=1))
    return {f'qa-{f:g}x': float(f * med) for f in fractions}


def run_seed(P, query_emb, categories, seed, eval_frac=0.2, sigma_fractions=(0.125, 0.25, 0.5, 1.0, 2.0)):
    """One train/eval split. Returns a tidy DataFrame of per-(query, target,
    method) mimicry errors."""
    m, n, _ = P.shape
    rng = np.random.default_rng(seed)
    anchor_idx, eval_idx = stratified_split(categories, eval_frac, rng)
    # LOQO invariant: geometries may only be built from anchor queries; no
    # model's response to an evaluation query may enter any geometry.
    assert len(np.intersect1d(anchor_idx, eval_idx)) == 0

    P_anchor = P[anchor_idx]
    sigmas = sigma_grid(query_emb[anchor_idx], sigma_fractions, rng)

    # Distance matrices: static once, task-localized per category,
    # query-aware per (eval query, sigma).
    D_static = weighted_dist_matrix(P_anchor, np.ones(len(anchor_idx)))
    cats_anchor = np.asarray(categories)[anchor_idx]
    D_task = {c: weighted_dist_matrix(P_anchor, (cats_anchor == c).astype(float))
              for c in np.unique(cats_anchor)}
    D_qa = {}
    for name, s in sigmas.items():
        W = rbf_weights(query_emb[anchor_idx], query_emb[eval_idx], s)   # (q, m_a)
        D_qa[name] = weighted_dist_matrix(P_anchor, W)                   # (q, n, n)

    # Per-eval-query candidate errors: E[q, t, i] = ||x_{i,q*} - x_{t,q*}||.
    E = np.sqrt(P[eval_idx])

    rows = []
    eye = np.eye(n, dtype=bool)
    for qi, q in enumerate(eval_idx):
        Eq = E[qi]                                    # (n_targets, n_candidates)
        off = Eq[~eye].reshape(n, n - 1)
        rand_err = off.mean(axis=1)
        oracle_err = off.min(axis=1)
        for t in range(n):
            picks = {'static': route_nearest(D_static, t),
                     'task': route_nearest(D_task[categories[q]], t)}
            for name in sigmas:
                picks[name] = route_nearest(D_qa[name][qi], t)
            for method, r in picks.items():
                rows.append((seed, q, categories[q], t, method, float(Eq[t, r]), r))
            rows.append((seed, q, categories[q], t, 'random', float(rand_err[t]), -1))
            rows.append((seed, q, categories[q], t, 'oracle', float(oracle_err[t]), -1))

    return pd.DataFrame(rows, columns=['seed', 'query', 'category', 'target', 'method', 'error', 'routed'])


def run_experiment(P, query_emb, categories, seeds=(0, 1, 2, 3, 4), **kw):
    return pd.concat([run_seed(P, query_emb, categories, s, **kw) for s in seeds],
                     ignore_index=True)
