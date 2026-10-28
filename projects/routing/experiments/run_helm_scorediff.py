"""Routing on HELM with the score-difference objective.

Outcome of routing candidate r for target t on query q*:

    err(t, r, q*) = |s_{r,q*} - s_{t,q*}|

(on binary tasks: correctness disagreement). Routers:
    qa-*      : nearest in the query-localized RESPONSE geometry
    task/static: task-indicator / uniform response geometries
    rfS-*     : direct kernel regression of realized |score diff| on the
                query embedding (the score-native router)
    rfS-task  : task-mean historical score disagreement
random/oracle bracket. Also runs the near/far residual-correlation
diagnostic on the score-difference surfaces: unlike mimicry, difficulty
makes |score diff| mechanically query-dependent, so this tests whether the
plateau verdict carries over.

Run from repo root: pixi run python -m projects.routing.experiments.run_helm_scorediff
"""

import os

import numpy as np
import pandas as pd

from .evaluate import sigma_grid, stratified_split
from .geometry import pairwise_query_dist_tensor, rbf_weights, weighted_dist_matrix
from .run_helm import RESULTS, load_paired_core

FRACTIONS = (0.125, 0.25, 0.5, 1.0)


def run_seed(P, P_s, query_emb, categories, seed, eval_frac=0.2):
    m, n, _ = P.shape
    rng = np.random.default_rng(seed)
    anchor_idx, eval_idx = stratified_split(categories, eval_frac, rng)
    assert len(np.intersect1d(anchor_idx, eval_idx)) == 0
    named = sigma_grid(query_emb[anchor_idx], FRACTIONS, rng)
    cats_anchor = categories[anchor_idx]

    # geometries (response space) and direct regressions (score space)
    W = {name: rbf_weights(query_emb[anchor_idx], query_emb[eval_idx], s)
         for name, s in named.items()}
    D_geo = {f'qa-{k.split("-")[1]}': weighted_dist_matrix(P[anchor_idx], w)
             for k, w in W.items()}
    D_static = weighted_dist_matrix(P[anchor_idx], np.ones(len(anchor_idx)))
    D_task = {c: weighted_dist_matrix(P[anchor_idx],
                                      (cats_anchor == c).astype(float))
              for c in np.unique(cats_anchor)}

    R_rf = {f'rfS-{k.split("-")[1]}':
            np.tensordot(w / w.sum(axis=1, keepdims=True), P_s[anchor_idx],
                         axes=([1], [0]))
            for k, w in W.items()}
    T_rf = {c: np.tensordot((cats_anchor == c) / (cats_anchor == c).sum(),
                            P_s[anchor_idx], axes=([0], [0]))
            for c in np.unique(cats_anchor)}

    E = P_s[eval_idx]                                   # realized |score diff|
    eye = np.eye(n, dtype=bool)
    rows = []
    for qi, q in enumerate(eval_idx):
        Eq = E[qi]
        off = Eq[~eye].reshape(n, n - 1)
        rand_err, oracle_err = off.mean(axis=1), off.min(axis=1)
        preds = {name: D[qi] for name, D in D_geo.items()}
        preds['static'] = D_static
        preds['task'] = D_task[categories[q]]
        preds.update({name: R[qi] for name, R in R_rf.items()})
        preds['rfS-task'] = T_rf[categories[q]]
        for t in range(n):
            for method, D in preds.items():
                d = D[t].copy(); d[t] = np.inf
                rows.append((seed, q, categories[q], t, method,
                             float(Eq[t, int(np.argmin(d))])))
            rows.append((seed, q, categories[q], t, 'random', float(rand_err[t])))
            rows.append((seed, q, categories[q], t, 'oracle', float(oracle_err[t])))
    return pd.DataFrame(rows, columns=['seed', 'query', 'category', 'target',
                                       'method', 'error'])


def diagnostic(P_s, query_emb, categories):
    n = P_s.shape[1]
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(0)
    near, far = [], []
    for task in sorted(set(categories)):
        idx = np.flatnonzero(categories == task)
        if len(idx) < 20:
            continue
        V = P_s[idx][:, iu[0], iu[1]]
        Vc = V - V.mean(axis=0, keepdims=True)
        Vn = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-12)
        U = query_emb[idx]
        a = rng.integers(0, len(idx), 4000); b = rng.integers(0, len(idx), 4000)
        keep = a != b; a, b = a[keep], b[keep]
        qd = np.linalg.norm(U[a] - U[b], axis=1)
        corr = (Vn[a] * Vn[b]).sum(axis=1)
        med = np.median(qd)
        near.append(corr[qd < med].mean()); far.append(corr[qd >= med].mean())
    return float(np.mean(near)), float(np.mean(far))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X, query_emb, categories, models, S = load_paired_core(with_scores=True)
    P = pairwise_query_dist_tensor(X)
    P_s = np.abs(S[None, :, :].transpose(2, 1, 0)
                 - S[None, :, :].transpose(2, 0, 1)).astype(np.float32)
    print(f'P {P.shape}, P_s {P_s.shape}, mean |score diff| {P_s.mean():.4f}')

    df = pd.concat([run_seed(P, P_s, query_emb, categories, s)
                    for s in range(5)], ignore_index=True)
    df.to_parquet(os.path.join(RESULTS, 'helm_scorediff.parquet'))

    per_seed = df.groupby(['method', 'seed'])['error'].mean().unstack()
    summary = pd.DataFrame({
        'mean_error': per_seed.mean(axis=1),
        'se': per_seed.std(axis=1) / np.sqrt(per_seed.shape[1]),
    }).sort_values('mean_error')
    summary.to_csv(os.path.join(RESULTS, 'helm_scorediff_summary.csv'))
    print(summary.round(4).to_string())

    nr, fr = diagnostic(P_s, query_emb, categories)
    print(f'\nnear/far residual correlation of |score diff| surfaces: '
          f'near={nr:.3f} far={fr:.3f}')


if __name__ == '__main__':
    main()
