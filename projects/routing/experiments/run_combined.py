"""Mimicry routing on the COMBINED HELM + EEE pool as one unlabeled task.

The union is inherently unpaired (the suites share no queries and the model
sets are disjoint up to naming), so everything runs through the factorized
localized geometry. Task and suite labels are hidden from all routers;
`task` (hidden labels) is reported only as a reference ceiling. Query
embeddings from both suites share one raw Gemini space and are jointly
PCA-reduced; response spaces are block-padded (HELM 80-d | EEE 83-d), so
within-suite mimicry errors are unchanged and responders of any query are
always same-suite.

Run from repo root: pixi run python -m projects.routing.experiments.run_combined
"""

import os

import numpy as np
import pandas as pd

from .dim_sweep import pca_to
from .run_eee import RESULTS, batched_localized_stats, localized_stats

FRACTIONS = (0.125, 0.25, 0.5, 1.0, 2.0)
D_Q = 40


def load_combined():
    import sys
    helm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'pkps', 'helm')
    sys.path.insert(0, helm_dir)
    from pipeline import loaders as H

    orig = H.pca_reduce_elbow

    def selective(R, max_components=None, **kw):
        if max_components == 64:          # query reduction call: keep raw
            return np.asarray(R)
        return orig(R, max_components=max_components, **kw)

    cwd = os.getcwd()
    os.chdir(helm_dir)
    H.pca_reduce_elbow = selective
    try:
        helm = H.load_suite()
        eee = H.load_eee()
    finally:
        H.pca_reduce_elbow = orig
        os.chdir(cwd)

    frames, Xs, Qs, all_names = [], [], [], []
    n_off, code_off = 0, 0
    dims = [helm[0].shape[1], eee[0].shape[1]]
    for si, (suite, data) in enumerate((('helm', helm), ('eee', eee))):
        (resp_X, Qu, qid_code, model_id, task_id, query_id,
         score_mat, models, tasks, groups, row_score, qmed) = data
        m2i = {mm: i for i, mm in enumerate(models)}
        frames.append(pd.DataFrame({
            'model': [m2i[mm] + n_off for mm in model_id],
            'task': [f'{suite}:{t}' for t in task_id],
            'query': [f'{suite}:{q}' for q in query_id],
            'code': qid_code + code_off,
            'score': row_score,
            'suite': suite}))
        pad = np.zeros((len(resp_X), sum(dims)), dtype=np.float32)
        off = sum(dims[:si])
        pad[:, off:off + dims[si]] = resp_X
        Xs.append(pad)
        Qs.append(np.asarray(Qu, dtype=np.float64))
        all_names.extend(f'{suite}:{mm}' for mm in models)
        n_off += len(models)
        code_off += len(Qu)

    rows = pd.concat(frames, ignore_index=True)
    X = np.concatenate(Xs)
    Qu_raw = np.concatenate(Qs)                       # joint raw Gemini space
    Qu = pca_to(Qu_raw, D_Q).astype(np.float32)
    return X, Qu, rows, n_off, all_names


def run_seed(X, Qu, rows, n, seed, n_eval=400):
    rng = np.random.default_rng(seed)
    resp = rows.groupby('query')['model'].nunique()
    eligible = resp[resp >= 3].index.to_numpy()
    eval_q = set(rng.choice(eligible, size=min(n_eval, len(eligible)),
                            replace=False))
    is_eval = rows['query'].isin(eval_q).to_numpy()
    anchor = ~is_eval
    assert not rows['query'][anchor].isin(eval_q).any()

    Xa = X[anchor]
    sa = rows['score'].to_numpy()[anchor]
    model_a = rows['model'].to_numpy()[anchor]
    task_a = rows['task'].to_numpy()[anchor]
    ua = Qu[rows['code'].to_numpy()[anchor]]

    # sigma from the joint anchor space median
    i = rng.integers(0, len(ua), 20000); k = rng.integers(0, len(ua), 20000)
    keep = i != k
    med = float(np.median(np.linalg.norm(ua[i[keep]] - ua[k[keep]], axis=1)))
    sigmas = {f'qa-{f:g}x': f * med for f in FRACTIONS}

    model_groups = [np.flatnonzero(model_a == m) for m in range(n)]
    eval_groups = list(rows[is_eval].groupby('query'))
    Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in eval_groups])
    D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
    D2 = D2 - D2.min(axis=1, keepdims=True)
    batched = {name: batched_localized_stats(
                   Xa, sa, model_groups, n,
                   np.exp(-D2 / (2.0 * s ** 2)).astype(np.float32))
               for name, s in sigmas.items()}
    task_stats = {tn: localized_stats(Xa, sa, (task_a == tn).astype(float),
                                      model_a, n)
                  for tn in np.unique(task_a)}
    static_stats = localized_stats(Xa, sa, np.ones(len(Xa)), model_a, n)

    out = []
    for gi, (q, g) in enumerate(eval_groups):
        resp_models = g['model'].to_numpy()
        t_name, suite = g['task'].iloc[0], g['suite'].iloc[0]
        Xq = X[g.index.to_numpy()]
        E = np.sqrt(((Xq[:, None] - Xq[None, :]) ** 2).sum(-1))
        r = len(resp_models)
        geoms = {name: (B[0][gi], B[2][gi]) for name, B in batched.items()}
        geoms['task*'] = (task_stats[t_name][0], task_stats[t_name][2])
        geoms['static'] = (static_stats[0], static_stats[2])
        # subsample targets on large-responder (HELM) queries to bound rows
        targets = (range(r) if r <= 20
                   else rng.choice(r, size=20, replace=False))
        for ti in targets:
            cand = [c for c in range(r) if c != ti]
            errs = E[ti, cand]
            for name, (phi, okm) in geoms.items():
                D = np.linalg.norm(phi[resp_models[cand]]
                                   - phi[resp_models[ti]], axis=1)
                D[~okm[resp_models[cand]]] = np.inf
                out.append((seed, suite, q, name, float(errs[int(np.argmin(D))])))
            out.append((seed, suite, q, 'random', float(errs.mean())))
            out.append((seed, suite, q, 'oracle', float(errs.min())))
    return pd.DataFrame(out, columns=['seed', 'suite', 'query', 'method',
                                      'error'])


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X, Qu, rows, n, _names = load_combined()
    print(f'combined: {len(rows)} rows, {n} models, '
          f'{rows["query"].nunique()} queries, {rows["task"].nunique()} '
          f'hidden tasks, X dim {X.shape[1]}, Qu {Qu.shape}')

    df = pd.concat([run_seed(X, Qu, rows, n, s) for s in range(5)],
                   ignore_index=True)
    df.to_parquet(os.path.join(RESULTS, 'combined_mimicry.parquet'))

    per_seed = df.groupby(['method', 'seed'])['error'].mean().unstack()
    summary = pd.DataFrame({
        'mean_error': per_seed.mean(axis=1),
        'se': per_seed.std(axis=1) / np.sqrt(per_seed.shape[1]),
    }).sort_values('mean_error')
    summary.to_csv(os.path.join(RESULTS, 'combined_summary.csv'))
    print(summary.round(4).to_string())

    by_suite = df.pivot_table(index='method', columns='suite', values='error')
    print('\nby source suite:')
    print(by_suite.round(4).to_string())


if __name__ == '__main__':
    main()
