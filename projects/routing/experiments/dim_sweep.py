"""Sweep the router's query- and response-embedding dimensions on HELM.

The evaluation metric is FIXED: mimicry error in the full-dimensional
response space (unreduced Gemini blocks + native one-hots, unit-normalized
per block, as in the loaders). Only the router's inputs vary: the response
dimension d_R used to build its geometry tensor (per-dataset PCA, one-hot
blocks kept native when smaller) and the query dimension d_Q used for the
localization weights (PCA of the raw 3072-d query embeddings).

Each (d_R, d_Q) cell reports the best of three bandwidth fractions
(0.125, 0.25, 0.5 x that query space's median) -- a landscape map, not a
deployment number. Also reruns the near/far residual-correlation
diagnostic at full query dimension.

Run from repo root: pixi run python -m projects.routing.experiments.dim_sweep
"""

import os

import numpy as np
import pandas as pd

from .geometry import pairwise_query_dist_tensor, rbf_weights, weighted_dist_matrix
from .evaluate import stratified_split
from .run_helm import RESULTS, load_paired_core
from pipeline import loaders as H  # noqa: E402  (path set by run_helm import)

D_R_GRID = (8, 16, 32, 48, 96, 'full')
D_Q_GRID = (5, 10, 20, 40, 'full')
FRACTIONS = (0.125, 0.25, 0.5)


def load_full():
    """Load the suite with reductions disabled (identity patch on the PCA)."""
    orig = H.pca_reduce_elbow
    H.pca_reduce_elbow = lambda R, max_components=None: R
    try:
        helm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'pkps', 'helm')
        cwd = os.getcwd()
        os.chdir(helm_dir)
        try:
            data = H.load_suite()
        finally:
            os.chdir(cwd)
    finally:
        H.pca_reduce_elbow = orig
    return data


def block_masks(X, categories):
    """Column support per dataset (blocks are disjoint coordinates)."""
    ds_of = np.array([c.split(':')[0] for c in categories])
    masks = {}
    for ds in np.unique(ds_of):
        act = (np.abs(X[:, ds_of == ds, :]) > 0).any(axis=(0, 1))
        masks[ds] = np.flatnonzero(act)
    return ds_of, masks


def pca_to(M, d):
    """PCA of rows of M to at most d components (centered)."""
    mu = M.mean(axis=0, keepdims=True)
    Z = M - mu
    # economy SVD on (N, D); N can be large -> use covariance trick on min side
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    k = min(d, Vt.shape[0])
    return Z @ Vt[:k].T


def reduce_responses(X, ds_of, masks, d_r):
    """Per-dataset PCA of the response blocks to d_r; unit-normalize rows."""
    if d_r == 'full':
        return X
    n, m, _ = X.shape
    parts, widths = [], []
    for ds, cols in masks.items():
        sel = ds_of == ds
        B = X[:, sel, :][:, :, cols].reshape(n * sel.sum(), len(cols))
        k = min(d_r, len(cols))
        R = pca_to(B.astype(np.float64), k)
        R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
        parts.append((ds, sel, R.reshape(n, sel.sum(), k).astype(np.float32)))
        widths.append(k)
    total = sum(widths)
    out = np.zeros((n, m, total), dtype=np.float32)
    off = 0
    for (ds, sel, R), w in zip(parts, widths):
        out[:, sel, off:off + w] = R
        off += w
    return out


def run():
    os.makedirs(RESULTS, exist_ok=True)
    data = load_full()
    X_full, query_emb_full, categories, models = load_paired_core(data)
    n = len(models)
    print(f'full: X {X_full.shape}, Qu {query_emb_full.shape}')

    ds_of, masks = block_masks(X_full, categories)
    print('block widths:', {k: len(v) for k, v in masks.items()})

    P_eval = pairwise_query_dist_tensor(X_full)          # FIXED metric
    E_eval = np.sqrt(P_eval)

    Q_spaces = {d: (query_emb_full if d == 'full'
                    else pca_to(query_emb_full.astype(np.float64),
                                d).astype(np.float32))
                for d in D_Q_GRID}

    # shared splits across all cells
    splits = [stratified_split(categories, 0.2, np.random.default_rng(s))
              for s in range(3)]
    eye = np.eye(n, dtype=bool)

    rows = []
    for d_r in D_R_GRID:
        Xr = reduce_responses(X_full, ds_of, masks, d_r)
        P_r = pairwise_query_dist_tensor(Xr) if d_r != 'full' else P_eval
        del Xr
        for d_q in D_Q_GRID:
            U = Q_spaces[d_q]
            for si, (anchor_idx, eval_idx) in enumerate(splits):
                med = np.median(np.linalg.norm(
                    U[anchor_idx][np.random.default_rng(0).choice(len(anchor_idx), 400)][:, None]
                    - U[anchor_idx][np.random.default_rng(1).choice(len(anchor_idx), 400)][None, :],
                    axis=-1))
                best = np.inf
                for f in FRACTIONS:
                    W = rbf_weights(U[anchor_idx], U[eval_idx], f * med)
                    D = weighted_dist_matrix(P_r[anchor_idx], W)
                    errs = []
                    for qi in range(len(eval_idx)):
                        Dm = D[qi].copy(); Dm[eye] = np.inf
                        picks = Dm.argmin(axis=1)
                        errs.append(E_eval[eval_idx[qi]][np.arange(n), picks].mean())
                    best = min(best, float(np.mean(errs)))
                rows.append((str(d_r), str(d_q), si, best))
                print(f'd_R={d_r} d_Q={d_q} seed={si}: {best:.4f}', flush=True)
        del P_r

    df = pd.DataFrame(rows, columns=['d_R', 'd_Q', 'seed', 'error'])
    df.to_csv(os.path.join(RESULTS, 'helm_dim_sweep.csv'), index=False)
    pivot = df.groupby(['d_R', 'd_Q'])['error'].mean().unstack()
    print('\nmean mimicry error (full-space metric), best-of-3 sigma:')
    print(pivot.round(4).to_string())

    # brackets under the full-space metric
    st_err, or_err, rd_err = [], [], []
    for anchor_idx, eval_idx in splits:
        D_st = weighted_dist_matrix(P_eval[anchor_idx], np.ones(len(anchor_idx)))
        for qi in eval_idx:
            Eq = E_eval[qi]
            off = Eq[~eye].reshape(n, n - 1)
            or_err.append(off.min(axis=1).mean()); rd_err.append(off.mean())
        Dm = D_st.copy(); Dm[eye] = np.inf
        picks = Dm.argmin(axis=1)
        st_err += [E_eval[qi][np.arange(n), picks].mean() for qi in eval_idx]
    print(f'\nstatic {np.mean(st_err):.4f} | oracle {np.mean(or_err):.4f} '
          f'| random {np.mean(rd_err):.4f}')

    # near/far residual-correlation diagnostic at FULL query dimension
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(0)
    near_all, far_all = [], []
    for task in sorted(set(categories)):
        idx = np.flatnonzero(categories == task)
        if len(idx) < 20:
            continue
        V = E_eval[idx][:, iu[0], iu[1]]
        Vc = V - V.mean(axis=0, keepdims=True)
        Vn = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-12)
        Uq = query_emb_full[idx]
        a = rng.integers(0, len(idx), 4000); b = rng.integers(0, len(idx), 4000)
        keep = a != b; a, b = a[keep], b[keep]
        qd = np.linalg.norm(Uq[a] - Uq[b], axis=1)
        corr = (Vn[a] * Vn[b]).sum(axis=1)
        med = np.median(qd)
        near_all.append(corr[qd < med].mean()); far_all.append(corr[qd >= med].mean())
    print(f'near/far residual correlation at FULL d_Q: '
          f'near={np.mean(near_all):.3f} far={np.mean(far_all):.3f}')


if __name__ == '__main__':
    run()
