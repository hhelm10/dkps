#!/usr/bin/env python
"""
pipeline/query_select.py  --  RD1: query-efficient evaluation (missing queries).

Reproduces & extends "Query-Efficient Model Evaluation Using Cached Responses"
(arXiv:2605.07096). Predict each model's full *benchmark* score (mean over all
queries of a dataset) from only m queries, leave-one-family-out (LOFO).

Methods:
  - sample : mean score over the model's m answered queries.
  - irt    : 1PL Rasch; difficulties from the maximal dense |F|x|Q| answer block,
             predict the anchor-Q expit mean.
  - dkps   : perspective space (delta query kernel = paired DKPS) -> LOFO linear
             regression of perspective coords onto reference full scores.
  - pkps   : same but rbf product query kernel -> works with UNPAIRED queries.
  - ens    : convex combo (1-a)*pkps + a*sample, a = m / M  (DKPS-heavy at small m).

Panels:
  budget   : all models answer the SAME m queries (paired); sweep m.
  unpaired : models answer m queries with overlap shared_frac (1=paired, 0=disjoint);
             paired DKPS & IRT lose their shared/dense structure, PKPS does not.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from graspologic.embed import ClassicalMDS

from dkps.unpaired_dkps import ProductKernelPerspectiveSpace, pca_reduce_elbow
from pipeline import loaders as H
from dkps.baselines import irt_fit_difficulties, irt_estimate_ability, irt_predict


def family(model):
    """Developer prefix: 'openai_gpt-4' -> 'openai' (HELM), 'allenai/olmo-3' -> 'allenai' (EEE)."""
    return model.split('/')[0] if '/' in model else model.split('_')[0]


def max_dense_block(amask):
    """Largest fully-observed |F|x|Q| rectangle (greedy over item coverage)."""
    order = np.argsort(-amask.sum(0))
    best_area, best = 0, (np.array([], int), np.array([], int))
    for c in range(1, len(order) + 1):
        Q = order[:c]
        F = np.where(amask[:, Q].all(1))[0]
        if len(F) == 0:
            break
        if len(F) * c > best_area:
            best_area, best = len(F) * c, (F, Q)
    return best


def _prep(resp_X, Qu, qid_code, model_id, query_id, keep):
    """PCA-reduce responses + queries once; return the long df and the median
    pairwise query distance (the bandwidth scale)."""
    resp_red = pca_reduce_elbow(resp_X[keep])
    codes = qid_code[keep]
    uniq = np.unique(codes)
    q_red = pca_reduce_elbow(Qu[uniq])
    code2vec = {c: q_red[j] for j, c in enumerate(uniq)}
    df = pd.DataFrame({'model_id': model_id[keep], 'task_id': 'b', 'query_id': query_id[keep],
                       'embedding': list(resp_red),
                       'query_embedding': [code2vec[c] for c in codes]})
    med = float(np.median(pdist(q_red))) if len(q_red) > 1 else 1.0
    return df, med


def _embed_sigma(df, query_kernel, sigma, mds_dim):
    """Perspective coords for a fixed query kernel / bandwidth."""
    kw = dict(query_kernel=query_kernel, response_kernel='rbf',
              response_bandwidth=H.SubsampleMedianBandwidth(max_n=5000))
    if query_kernel == 'rbf':
        kw['query_bandwidth'] = sigma
    est = ProductKernelPerspectiveSpace(normalization=H.NORMALIZATION, **kw).fit(df)
    dist = np.nan_to_num(est.dist_matrix_.copy(), nan=np.nanmax(est.dist_matrix_))
    n2r = {m: i for i, m in enumerate(est.model_names_)}
    Z = ClassicalMDS(n_components=min(mds_dim, len(est.model_names_) - 1),
                     dissimilarity='precomputed').fit_transform(dist)
    return Z, n2r


def _regressor(predictor):
    if predictor == 'knn':
        return KNeighborsRegressor(n_neighbors=5, weights='distance')
    return LinearRegression()


def _lofo_regress(Z, n2r, models, yact_map, fams, predictor='ols'):
    """Leave-one-family-out regression of perspective coords -> full score."""
    pred = np.full(len(models), np.nan)
    present = [m for m in models if m in n2r]
    coords = {m: Z[n2r[m]] for m in present}
    for i, m in enumerate(models):
        if m not in coords:
            continue
        tr = [mm for mm in present if fams[mm] != fams[m] and np.isfinite(yact_map[mm])]
        if len(tr) < 3:
            continue
        Xtr = np.vstack([coords[mm] for mm in tr])
        ytr = np.array([yact_map[mm] for mm in tr])
        reg = _regressor(predictor)
        if predictor == 'knn':
            reg.set_params(n_neighbors=min(5, len(tr)))
        pred[i] = float(reg.fit(Xtr, ytr).predict(coords[m][None])[0])
    return np.clip(pred, 0, 1)


def run_seed(data, n_paired, n_unpaired, n_models_use, seed, n_query_pool=None,
             mds_dim=8, include_irt=True, predictor='ols'):
    """Each (used) model answers n_paired SHARED anchor queries plus n_unpaired
    random per-model queries. Predict the benchmark score (mean over the query
    pool); LOFO. n_models_use subsamples the cohort; n_query_pool the query pool."""
    (resp_X, Qu, qid_code, model_id, task_id, query_id,
     score_mat, models, tasks, groups, row_score) = data
    rng = np.random.default_rng(seed)

    use = list(rng.choice(models, n_models_use, replace=False)) \
        if n_models_use and n_models_use < len(models) else list(models)
    fams = {mm: family(mm) for mm in use}
    nu = len(use)
    uidx = {mm: i for i, mm in enumerate(use)}

    qids_all = sorted(set(query_id))
    qids = sorted(rng.choice(qids_all, n_query_pool, replace=False)) \
        if n_query_pool and n_query_pool < len(qids_all) else qids_all
    qcode = {q: j for j, q in enumerate(qids)}
    M = len(qids)
    pool_mask = np.isin(query_id, qids)
    m_total = max(1, min(n_paired + n_unpaired, M))

    # shared anchor (paired) + per-model random (unpaired)
    np_ = min(n_paired, M)
    anchor = set(rng.choice(M, np_, replace=False)) if np_ else set()
    non_anchor = np.array([j for j in range(M) if j not in anchor])

    yact_map = {mm: row_score[(model_id == mm) & pool_mask].mean() for mm in use}
    yv = np.array([yact_map[mm] for mm in use])

    amask = np.zeros((nu, M), bool)
    S = np.full((nu, M), np.nan)
    keep = []
    sample_sc = np.full(nu, np.nan)
    rows_by_model = {mm: np.where((model_id == mm) & pool_mask)[0] for mm in use}
    for i, mm in enumerate(use):
        items = set(anchor)
        nu_extra = min(n_unpaired, len(non_anchor))
        if nu_extra > 0:
            items |= set(rng.choice(non_anchor, nu_extra, replace=False))
        sc = []
        for r in rows_by_model[mm]:
            j = qcode[query_id[r]]
            if j in items:
                amask[i, j] = True
                S[i, j] = row_score[r]
                keep.append(r)
                sc.append(row_score[r])
        if sc:
            sample_sc[i] = np.mean(sc)
    keep = np.array(sorted(keep))

    out = {'sample': sample_sc}

    # ---- IRT via maximal dense |F|x|Q| block (binary scores only)
    if include_irt:
        F, Q = max_dense_block(amask)
        irt = np.full(nu, np.nan)
        if len(F) >= 2 and len(Q) >= 1:
            beta_Q, _ = irt_fit_difficulties(np.nan_to_num(S[np.ix_(F, Q)], nan=0.0))
            for i in range(nu):
                a = amask[i, Q]
                if a.any():
                    irt[i] = irt_predict(irt_estimate_ability(S[i, Q[a]], beta_Q[a]), beta_Q)
        out['irt'] = irt

    # ---- DKPS (delta) and PKPS (rbf, query bandwidth + predictor by LOFO-CV)
    df_obs, med = _prep(resp_X, Qu, qid_code, model_id, query_id, keep)
    Zd, n2rd = _embed_sigma(df_obs, 'delta', None, mds_dim)
    out['dkps'] = _lofo_regress(Zd, n2rd, use, yact_map, fams, predictor=predictor)

    # Direct median-scaled bandwidth grid (no analytic M/m factor: the LOFO-CV picks
    # it, and the reference full scores give a dense, low-noise selection signal).
    # One response-kernel pair-loop yields the whole grid (dist_matrices).
    bw = H.SubsampleMedianBandwidth(max_n=5000)
    est = ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel='rbf',
                                        query_bandwidth=bw, response_bandwidth=bw,
                                        normalization=H.NORMALIZATION)
    scales = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    sigmas = [s * med for s in scales]
    names, Ds = est.dist_matrices(df_obs, sigmas)
    n2r = {m: i for i, m in enumerate(names)}
    best = (np.inf, None, None)
    for s in sigmas:
        D = Ds[s]
        D = np.nan_to_num(D, nan=np.nanmax(D)) if np.any(np.isnan(D)) else D
        Z = ClassicalMDS(n_components=min(mds_dim, len(names) - 1),
                         dissimilarity='precomputed').fit_transform(D)
        pred = _lofo_regress(Z, n2r, use, yact_map, fams, predictor=predictor)
        h = np.isfinite(pred) & np.isfinite(yv)
        e = float(np.mean(np.abs(pred[h] - yv[h]))) if h.any() else np.inf
        if e < best[0]:
            best = (e, pred, s / med)
    out['pkps'], chosen_c = best[1], best[2]

    # ---- ensemble PKPS + sample, weight fit per run by LOFO-CV
    bestE = (np.inf, None)
    for a in np.linspace(0, 1, 21):
        pred = np.clip((1 - a) * out['pkps'] + a * sample_sc, 0, 1)
        h = np.isfinite(pred) & np.isfinite(yv)
        e = float(np.mean(np.abs(pred[h] - yv[h]))) if h.any() else np.inf
        if e < bestE[0]:
            bestE = (e, pred)
    out['ens'] = bestE[1]

    rows = []
    for meth, pred in out.items():
        h = np.isfinite(pred) & np.isfinite(yv)
        mae = float(np.mean(np.abs(pred[h] - yv[h]))) if h.any() else np.nan
        rows.append(dict(n_paired=np_, n_unpaired=n_unpaired, n_total=m_total,
                         n_models=nu, n_queries=M, seed=seed, method=meth, mae=mae,
                         c=chosen_c if meth == 'pkps' else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='math')
    ap.add_argument('--panel', choices=['budget', 'pairing', 'n_models'],
                    default='budget')
    ap.add_argument('--totals', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    ap.add_argument('--fixed_total', type=int, default=16)
    ap.add_argument('--paired_values', type=int, nargs='+', default=[0, 2, 4, 8, 12, 16])
    ap.add_argument('--n_models_values', type=int, nargs='+', default=[10, 20, 40, 60, 95])
    ap.add_argument('--nm_paired', type=int, default=8)
    ap.add_argument('--nm_unpaired', type=int, default=8)
    ap.add_argument('--n_seeds', type=int, default=20)
    ap.add_argument('--no_irt', action='store_true', help='skip IRT (e.g. continuous scores)')
    ap.add_argument('--predictor', choices=['ols', 'knn'], default='knn',
                    help='LOFO regressor for the perspective coords -> full score map '
                         '(knn unifies with RD2; ~tied with ols, slightly better unpaired)')
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-pkps-rd1')
    args = ap.parse_args()

    cfg = H.DATASETS[args.dataset]
    data = H.load_helm_math(cfg['parquet'], cfg['tsv'], query_source='google',
                            query_parquet=cfg['query_parquet'], score_col=cfg['score_col'])
    print(f'{args.dataset}: {len(data[7])} models')

    if args.panel == 'budget':             # all-paired, sweep total budget
        specs = [(t, 0, None) for t in args.totals]
        xcol = 'n_total'
    elif args.panel == 'pairing':          # fix total, slide paired -> unpaired
        specs = [(p, args.fixed_total - p, None) for p in args.paired_values]
        xcol = 'n_paired'
    else:                                  # n_models
        specs = [(args.nm_paired, args.nm_unpaired, n) for n in args.n_models_values]
        xcol = 'n_models'
    jobs = [delayed(run_seed)(data, npd, nupd, nm, s, include_irt=not args.no_irt,
                              predictor=args.predictor)
            for (npd, nupd, nm) in specs for s in range(args.n_seeds)]
    res = pd.DataFrame([r for sub in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) for r in sub])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / f'rd1_{args.dataset}_{args.panel}.csv', index=False)
    print(res.pivot_table(index=xcol, columns='method', values='mae').round(4).to_string())


if __name__ == '__main__':
    main()
