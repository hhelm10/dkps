"""Fine low-dimensional sweep of the QUERY space for routing.

Hypothesis: RBF localization degrades in higher dimensions (distance
concentration), so aggressive PCA of the query embeddings may win if each
dimension is given its own well-tuned bandwidth. Response space fixed at
full dimension (the sweep optimum); metric fixed in the full response
space. Reports, per d_Q: the best error over a WIDE sigma grid and which
sigma fraction won -- if low-d wins, its best fraction should be usable;
if high-d only works at tiny fractions (1-NN limit), that supports the
concentration story.

Run from repo root: pixi run python -m projects.routing.experiments.query_dim_fine
"""

import os

import numpy as np
import pandas as pd

from .dim_sweep import load_full, pca_to
from .evaluate import stratified_split
from .geometry import pairwise_query_dist_tensor, rbf_weights, weighted_dist_matrix
from .run_helm import RESULTS, load_paired_core

D_Q_GRID = (2, 3, 4, 6, 8, 12, 16, 20, 40)
FRACTIONS = (0.03, 0.05, 0.09, 0.125, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    data = load_full()
    X_full, Qu_full, categories, models = load_paired_core(data)
    n = len(models)
    P = pairwise_query_dist_tensor(X_full)      # router AND metric: full space
    E = np.sqrt(P)
    eye = np.eye(n, dtype=bool)
    print(f'P {P.shape}, Qu {Qu_full.shape}')

    splits = [stratified_split(categories, 0.2, np.random.default_rng(s))
              for s in range(3)]
    rng = np.random.default_rng(0)

    rows = []
    for d_q in D_Q_GRID:
        U = pca_to(Qu_full.astype(np.float64), d_q).astype(np.float32)
        for si, (anchor_idx, eval_idx) in enumerate(splits):
            Ua = U[anchor_idx]
            i = rng.integers(0, len(Ua), 20000); k = rng.integers(0, len(Ua), 20000)
            keep = i != k
            med = float(np.median(np.linalg.norm(Ua[i[keep]] - Ua[k[keep]], axis=1)))
            for f in FRACTIONS:
                W = rbf_weights(Ua, U[eval_idx], f * med)
                D = weighted_dist_matrix(P[anchor_idx], W)
                errs = []
                for qi in range(len(eval_idx)):
                    Dm = D[qi].copy(); Dm[eye] = np.inf
                    picks = Dm.argmin(axis=1)
                    errs.append(E[eval_idx[qi]][np.arange(n), picks].mean())
                rows.append((d_q, f, si, float(np.mean(errs))))
        best = (pd.DataFrame([r for r in rows if r[0] == d_q],
                             columns=['d', 'f', 's', 'e'])
                .groupby('f')['e'].mean())
        print(f'd_Q={d_q:3d}: best={best.min():.4f} at {best.idxmin():g}x', flush=True)

    df = pd.DataFrame(rows, columns=['d_Q', 'fraction', 'seed', 'error'])
    df.to_csv(os.path.join(RESULTS, 'query_dim_fine.csv'), index=False)
    pivot = df.groupby(['d_Q', 'fraction'])['error'].mean().unstack()
    print('\nmean error by (d_Q, sigma fraction):')
    print(pivot.round(4).to_string())
    print('\nbest per d_Q:')
    print(pd.DataFrame({'best_error': pivot.min(axis=1),
                        'best_fraction': pivot.idxmin(axis=1)}).to_string())


if __name__ == '__main__':
    main()
