"""Response-free direct-regression router on HELM (equivalence check).

On a paired suite the qa geometry is already the kernel regression of
SQUARED mimicry error onto the query embedding, so the direct router
rf-sq (route by tensordot(w, P)) must pick identically to qa; rf-abs
(regress |error| instead) differs only through the transform. This run
verifies both against the geometry.

Run from repo root: pixi run python -m projects.routing.experiments.run_helm_rf
"""

import os

import numpy as np
import pandas as pd

from .evaluate import sigma_grid, stratified_split
from .geometry import pairwise_query_dist_tensor, rbf_weights
from .run_helm import RESULTS, load_paired_core

FRACTIONS = (0.125, 0.25, 0.5)


def run_seed(P, E_abs, query_emb, categories, seed, eval_frac=0.2):
    m, n, _ = P.shape
    rng = np.random.default_rng(seed)
    anchor_idx, eval_idx = stratified_split(categories, eval_frac, rng)
    named = sigma_grid(query_emb[anchor_idx], FRACTIONS, rng)

    E = np.sqrt(P[eval_idx])
    eye = np.eye(n, dtype=bool)
    rows = []
    for name, s in named.items():
        W = rbf_weights(query_emb[anchor_idx], query_emb[eval_idx], s)
        Wn = W / W.sum(axis=1, keepdims=True)
        pred_sq = np.tensordot(Wn, P[anchor_idx], axes=([1], [0]))     # (q, n, n)
        pred_abs = np.tensordot(Wn, E_abs[anchor_idx], axes=([1], [0]))
        for qi in range(len(eval_idx)):
            for kind, D in (('qa/rf-sq', pred_sq[qi]), ('rf-abs', pred_abs[qi])):
                Dm = D.copy(); Dm[eye] = np.inf
                picks = Dm.argmin(axis=1)
                err = E[qi][np.arange(n), picks]
                rows.append((seed, name, kind, float(err.mean())))
    return pd.DataFrame(rows, columns=['seed', 'sigma', 'router', 'mean_error'])


def main():
    X, query_emb, categories, models = load_paired_core()
    P = pairwise_query_dist_tensor(X)
    E_abs = np.sqrt(P)
    df = pd.concat([run_seed(P, E_abs, query_emb, categories, s)
                    for s in range(5)], ignore_index=True)
    out = df.groupby(['sigma', 'router'])['mean_error'].mean().unstack()
    out.to_csv(os.path.join(RESULTS, 'helm_rf_check.csv'))
    print(out.round(4).to_string())


if __name__ == '__main__':
    main()
