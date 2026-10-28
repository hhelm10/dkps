#!/usr/bin/env python
"""Does reading responses help beyond score patterns?  (colleague review, 2026-10)

Replays the query-efficiency sampling of experiments/query_efficiency.run_seed (same RNG
call order, so `sample` and `pkps` reproduce the published CSVs) and adds:

  additive   "strength + difficulty": per-task mean/sd from the reference models, plus the
             target's own offset estimated from its m-query sample scores across tasks
             (ridge-shrunk; shrinkage tuned by nested leave-family-out CV). Reads no responses.
  pkps_add   additive + K-NN on the perspective of the references' additive RESIDUALS
             (what responses add on top of score patterns).
  meanresp   PKPS with a flat query kernel: distances between per-model MEAN response
             embeddings (no query information at all).
  pk_offset  additive, but the target's STRENGTH is read off the perspective (K-NN over
             the references' offsets) instead of its own sample scores.
  hybrid     additive with the strength a nested-tuned blend of the sample-based and the
             perspective-based estimate, plus a nested-weighted K-NN residual: score
             patterns + responses.
  hybrid_mean  the same hybrid on the flat-kernel (mean-response) perspective.
  resid      K-NN on a perspective built from QUERY-RESIDUALIZED responses: each response
             minus the kernel-weighted mean response of other models to similar queries
             (pools the query kernel across all models instead of pairwise).
  hybrid_res the hybrid on that perspective.
  ens_*      the paper's ensemble (alpha * sample + (1 - alpha) * X, one alpha per held-out
             family) with X in {pkps, additive, pkps_add, meanresp}.

All tuning is STRICTLY nested: for a held-out family f, every quantity used to choose a
hyperparameter (shrinkage, alpha) is computed from predictions that never saw f. The paper
pipeline's `ens` tunes alpha on predictions whose training sets include f (a small
transitive leak); it is reproduced here as `ens_pkps_paper` for comparison.

--refs full    references contribute FULL scores on every task (what the paper code does)
--refs sample  references contribute only their own m-query sample scores on observed
               cells (what the paper text says). No full score exists anywhere, so alpha
               cannot be tuned against truth and the ens_* methods are omitted; the
               additive shrinkage is tuned by leave-one-task-out on sample scores.
"""
import argparse
import sys, pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline import loaders as H                                    # noqa: E402
from pipeline.query_select import family                            # noqa: E402
from experiments.query_efficiency import _cv_bandwidth              # noqa: E402

LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
ALPHAS = np.linspace(0, 1, 21)
WS = [(w, g) for w in (0.0, 0.25, 0.5, 0.75, 1.0) for g in (0.0, 0.5, 1.0)]   # (strength blend, residual weight)
K = 5


def knn_predict(Dz, y, tr_idx, te_idx):
    """Inverse-distance K-NN (sklearn KNeighborsRegressor(weights='distance') semantics)."""
    out = np.full(len(te_idx), np.nan)
    tr_idx = tr_idx[np.isfinite(y[tr_idx])]
    if len(tr_idx) < 3:
        return out
    k = min(K, len(tr_idx))
    for a, i in enumerate(te_idx):
        d = Dz[i, tr_idx]
        nb = np.argpartition(d, k - 1)[:k] if k < len(d) else np.arange(len(d))
        dd, yy = d[nb], y[tr_idx[nb]]
        zero = dd == 0
        out[a] = yy[zero].mean() if zero.any() else np.average(yy, weights=1.0 / dd)
    return out


def resid_reps(X, Qk, qidx, mid, names, sigma, chunk=1500):
    """Per-model mean of query-residualized responses: each response minus the
    kernel-weighted mean response of OTHER models to similar queries (leave-own-model-out
    Nadaraya-Watson in the query embedding). sigma -> inf recovers the mean-response
    representation up to a constant; sigma -> 0 compares only identical queries.
    X: (N, p) responses; Qk: (U, q) unique query embeddings; qidx: (N,) row -> unique query."""
    U, p = Qk.shape[0], X.shape[1]
    S = np.zeros((U, p)); np.add.at(S, qidx, X)
    c = np.bincount(qidx, minlength=U).astype(float)
    inv = 1.0 / (2.0 * sigma * sigma)
    Num = np.empty((U, p)); Den = np.empty(U)
    q2 = (Qk ** 2).sum(1)
    for a in range(0, U, chunk):
        d2 = np.maximum(q2[a:a + chunk, None] + q2[None] - 2.0 * Qk[a:a + chunk] @ Qk.T, 0.0)
        Kc = np.exp(-d2 * inv)
        Num[a:a + chunk] = Kc @ S; Den[a:a + chunk] = Kc @ c
    gmean = X.mean(0)
    reps = np.zeros((len(names), p))
    for r, mm in enumerate(names):
        rows = np.where(mid == mm)[0]
        qi = qidx[rows]
        uq, loc = np.unique(qi, return_inverse=True)
        Si = np.zeros((len(uq), p)); np.add.at(Si, loc, X[rows])
        ci = np.bincount(loc, minlength=len(uq)).astype(float)
        d2 = np.maximum(q2[uq, None] + q2[None, uq] - 2.0 * Qk[uq] @ Qk[uq].T, 0.0)
        Kii = np.exp(-d2 * inv)
        num = Num[uq] - Kii @ Si
        den = Den[uq] - Kii @ ci
        ok = den > 1e-8
        mhat = np.where(ok[:, None], num / np.where(ok, den, 1.0)[:, None], gmean)
        reps[r] = (X[rows] - mhat[loc]).mean(0)
    return reps


def task_stats(Yref, tr):
    mu = np.nanmean(Yref[tr], axis=0)
    sd = np.nanstd(Yref[tr], axis=0)
    return mu, np.where(np.isfinite(sd) & (sd > 1e-3), sd, 1e-3)


def offsets(Zs, lam, leave_one_task_out=False):
    """Ridge-shrunk per-model mean of standardized scores; optionally a (model x task)
    matrix of offsets that exclude each cell's own task."""
    fin = np.isfinite(Zs)
    S = np.where(fin, Zs, 0.0).sum(1, keepdims=True)
    T = fin.sum(1, keepdims=True)
    if not leave_one_task_out:
        return (S / np.maximum(T + lam, 1e-9)) * np.ones((1, Zs.shape[1]))
    return (S - np.where(fin, Zs, 0.0)) / np.maximum(T - fin + lam, 1e-9)


def run(data, m, seed, n_models=None, p_task=1.0, refs='full', mds_dim=8):
    (resp_X, Qu, qid_code, model_id, task_id, query_id,
     score_mat, models_all, tasks, groups, row_score, qmed) = data
    rng = np.random.default_rng(seed)
    if n_models is not None and n_models < len(models_all):
        sel = np.sort(rng.choice(len(models_all), n_models, replace=False))
        models = [models_all[i] for i in sel]; score_mat = score_mat[sel]
    else:
        models = list(models_all)
    n, T = len(models), len(tasks)
    fams = np.array([family(mm) for mm in models])
    obs = (rng.random((n, T)) < p_task) if p_task < 1.0 else np.ones((n, T), bool)
    for i in range(n):
        if not obs[i].any(): obs[i, rng.integers(T)] = True
    for t in range(T):
        if not obs[:, t].any(): obs[rng.integers(n), t] = True

    keep, sample_mat = [], np.full((n, T), np.nan)
    for i, mm in enumerate(models):
        for t, tt in enumerate(tasks):
            if not obs[i, t]:
                continue
            idx = groups.get((mm, tt))
            if idx is None or not len(idx):
                continue
            take = idx if len(idx) <= m else rng.choice(idx, m, replace=False)
            keep.append(take)
            sample_mat[i, t] = row_score[take].mean()
    keep = np.concatenate(keep)

    # --- perspectives: PKPS (CV bandwidth), and the flat-kernel / mean-response limit
    df = pd.DataFrame({'model_id': model_id[keep], 'task_id': task_id[keep],
                       'query_id': query_id[keep], 'embedding': list(resp_X[keep]),
                       'query_embedding': list(Qu[qid_code[keep]])})
    bw = H.SubsampleMedianBandwidth(max_n=5000)
    est = H.ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel='linear',
                                          query_bandwidth=bw, response_bandwidth=bw,
                                          normalization='pair')
    est_mn = H.ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel='linear',
                                             query_bandwidth=bw, response_bandwidth=bw,
                                             normalization='model')
    SIG_GRID = list(qmed * np.array([0.03, 0.1, 0.3, 1.0, 3.0]))
    names, Ds = est.dist_matrices(df, SIG_GRID)
    Zs = {sg: H._mds_full(Ds[sg], names, models, mds_dim) for sg in SIG_GRID}
    sig_star = _cv_bandwidth(Zs, sample_mat, obs)
    Zp = Zs.get(sig_star)
    if Zp is None:
        Zp = H._mds_full(Ds[SIG_GRID[2]], names, models, mds_dim)
    _, Dn = est_mn.dist_matrices(df, SIG_GRID)
    Zn_all = {sg: H._mds_full(Dn[sg], names, models, mds_dim) for sg in SIG_GRID}
    sig_mn = _cv_bandwidth(Zn_all, sample_mat, obs)
    Zn = Zn_all[sig_mn] if sig_mn is not None else Zn_all[SIG_GRID[2]]
    est_loc = H.ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel='linear',
                                              query_bandwidth=bw, response_bandwidth=bw,
                                              normalization='local')
    _, Dl = est_loc.dist_matrices(df, SIG_GRID)
    Dl = {sg: np.nan_to_num(D, nan=np.nanmax(D)) for sg, D in Dl.items()}
    Zl_all = {sg: H._mds_full(Dl[sg], names, models, mds_dim) for sg in SIG_GRID}
    sig_loc = _cv_bandwidth(Zl_all, sample_mat, obs)
    Zl = Zl_all[sig_loc] if sig_loc is not None else Zl_all[SIG_GRID[2]]
    mid = model_id[keep]
    Xbar = np.stack([resp_X[keep][mid == mm].mean(0) for mm in names])
    Zm = H._mds_full(squareform(pdist(Xbar)), names, models, mds_dim)
    # query-residualized responses, bandwidth chosen by the same observed-score CV
    uqc, qidx = np.unique(qid_code[keep], return_inverse=True)
    Qk = Qu[uqc].astype(np.float64)
    Xk = resp_X[keep].astype(np.float64)
    RES_GRID = list(qmed * np.array([0.1, 0.3, 1.0, 3.0]))
    Zr_all = {sg: H._mds_full(squareform(pdist(resid_reps(Xk, Qk, qidx, mid, names, sg))),
                              names, models, mds_dim) for sg in RES_GRID}
    sig_res = _cv_bandwidth(Zr_all, sample_mat, obs)
    Zr = Zr_all[sig_res] if sig_res is not None else Zr_all[RES_GRID[2]]
    Dz = {'pkps': squareform(pdist(Zp)), 'meanresp': squareform(pdist(Zm)),
          'resid': squareform(pdist(Zr)), 'pkps_mn': squareform(pdist(Zn)),
          'pkps_loc': squareform(pdist(Zl))}

    Yref = score_mat if refs == 'full' else np.where(obs, sample_mat, np.nan)
    truth = score_mat
    ce = obs & np.isfinite(sample_mat) & np.isfinite(truth)
    all_idx = np.arange(n)
    BASE = ['pkps', 'pkps_mn', 'pkps_loc', 'meanresp', 'resid', 'additive', 'pkps_add', 'pk_offset', 'hybrid', 'hybrid_mean', 'hybrid_res']
    HYB = {'hybrid': 'pkps', 'hybrid_mean': 'meanresp', 'hybrid_res': 'resid'}

    def predict_group(tr, te_idx, lam, loto=False):
        """Predictions for the models in te_idx from the reference set `tr` (bool mask).
        loto: the target's sample-based offset excludes each cell's own task (used only to
        tune against sample scores when no full score exists). Returns the finished
        predictions plus the components the hybrids are assembled from."""
        tr_idx = all_idx[tr]
        mu, sd = task_stats(Yref, tr)
        P = {b: np.full((len(te_idx), T), np.nan) for b in BASE}
        zs_te = (sample_mat[te_idx] - mu) / sd                 # target's standardized samples
        a_te = offsets(zs_te, lam, leave_one_task_out=loto)
        P['additive'] = np.clip(mu + sd * a_te, 0, 1)
        zref = (Yref - mu) / sd
        b_ref = offsets(zref, 0.0 if refs == 'full' else lam)
        R = zref - b_ref                                        # references' additive residuals
        C = {'mu': mu, 'sd': sd, 'a': a_te}
        for nm, key in HYB.items():                             # strength read off the perspective
            bh = knn_predict(Dz[key], b_ref[:, 0], tr_idx, te_idx)
            rh = np.stack([knn_predict(Dz[key], R[:, t], tr_idx, te_idx) for t in range(T)], 1)
            C[nm] = (bh, rh)
        P['pk_offset'] = np.clip(mu + sd * C['hybrid'][0][:, None], 0, 1)
        P['pkps_add'] = np.clip(mu + sd * (a_te + C['hybrid'][1]), 0, 1)
        for t in range(T):
            P['pkps'][:, t] = np.clip(knn_predict(Dz['pkps'], Yref[:, t], tr_idx, te_idx), 0, 1)
            P['meanresp'][:, t] = np.clip(knn_predict(Dz['meanresp'], Yref[:, t], tr_idx, te_idx), 0, 1)
            P['resid'][:, t] = np.clip(knn_predict(Dz['resid'], Yref[:, t], tr_idx, te_idx), 0, 1)
            P['pkps_mn'][:, t] = np.clip(knn_predict(Dz['pkps_mn'], Yref[:, t], tr_idx, te_idx), 0, 1)
            P['pkps_loc'][:, t] = np.clip(knn_predict(Dz['pkps_loc'], Yref[:, t], tr_idx, te_idx), 0, 1)
        return P, C

    def hybrid_pred(C, nm, wg):
        (w, gam), (bh, rh) = wg, C[nm]
        return np.clip(C['mu'] + C['sd'] * (w * C['a'] + (1 - w) * bh[:, None] + gam * rh), 0, 1)

    out = {b: np.full((n, T), np.nan) for b in BASE}
    ens = {b: np.full((n, T), np.nan) for b in BASE} if refs == 'full' else {}
    lam_used, alpha_used = [], {b: [] for b in BASE}
    w_used = {nm: [] for nm in ('hybrid', 'hybrid_mean', 'hybrid_res')}
    g_used = {nm: [] for nm in ('hybrid', 'hybrid_mean', 'hybrid_res')}
    ufams = sorted(set(fams))
    for f in ufams:
        te = fams == f
        tr = ~te
        te_idx = all_idx[te]
        inner_fams = [g for g in ufams if g != f]
        # ---- shrinkage: nested over the other families
        lam_err = np.zeros(len(LAMS))
        for g in inner_fams:
            gi = all_idx[fams == g]
            tr2 = tr & (fams != g)
            if tr2.sum() < 3:
                continue
            mu, sd = task_stats(Yref, tr2)
            zs = (sample_mat[gi] - mu) / sd
            for li, lam in enumerate(LAMS):
                if refs == 'full':                              # score vs the references' full scores
                    pred = np.clip(mu + sd * offsets(zs, lam), 0, 1)
                    msk = ce[gi]
                    lam_err[li] += np.abs(pred - truth[gi])[msk].sum()
                else:                                           # only samples exist: leave-one-task-out
                    pred = np.clip(mu + sd * offsets(zs, lam, leave_one_task_out=True), 0, 1)
                    msk = obs[gi] & np.isfinite(sample_mat[gi])
                    lam_err[li] += np.abs(pred - sample_mat[gi])[msk].sum()
        lam = LAMS[int(np.argmin(lam_err))]
        lam_used.append(lam)
        # ---- nested predictions for the other families (never see f)
        loto = refs != 'full'                # no full score to tune against: predict held-out samples
        tune_y = truth if refs == 'full' else sample_mat
        inner = {b: np.full((n, T), np.nan) for b in BASE}
        w_err = {nm: np.zeros(len(WS)) for nm in HYB}
        inner_C = []
        for g in inner_fams:
            gi = all_idx[fams == g]
            tr2 = tr & (fams != g)
            if tr2.sum() < 3:
                continue
            Pg, Cg = predict_group(tr2, gi, lam, loto=loto)
            inner_C.append((gi, Cg))
            for b in BASE:
                inner[b][gi] = Pg[b]
            msk_g = obs[gi] & np.isfinite(sample_mat[gi]) & np.isfinite(tune_y[gi])
            for nm in HYB:
                for wi, w in enumerate(WS):
                    e = np.abs(hybrid_pred(Cg, nm, w) - tune_y[gi])
                    w_err[nm][wi] += np.nansum(np.where(msk_g, e, 0.0))
        w_star = {nm: WS[int(np.argmin(w_err[nm]))] for nm in HYB}
        for nm in HYB:
            w_used[nm].append(w_star[nm][0]); g_used[nm].append(w_star[nm][1])
            for gi, Cg in inner_C:
                inner[nm][gi] = hybrid_pred(Cg, nm, w_star[nm])
        # ---- test-family predictions (references = every other family)
        P, C = predict_group(tr, te_idx, lam)
        for nm in HYB:
            P[nm] = hybrid_pred(C, nm, w_star[nm])
        for b in BASE:
            out[b][te_idx] = P[b]
        if refs != 'full':
            continue
        for b in BASE:
            msk = ce & tr[:, None] & np.isfinite(inner[b])
            if msk.sum() >= 5:
                errs = [np.abs(np.clip(a * sample_mat[msk] + (1 - a) * inner[b][msk], 0, 1)
                               - truth[msk]).mean() for a in ALPHAS]
                a_f = float(ALPHAS[int(np.argmin(errs))])
            else:
                a_f = 0.5
            alpha_used[b].append(a_f)
            blend = np.clip(a_f * sample_mat[te_idx] + (1 - a_f) * P[b], 0, 1)
            ens[b][te_idx] = np.where(np.isfinite(P[b]), blend, sample_mat[te_idx])

    preds = {'sample': sample_mat, **out, **{f'ens_{b}': v for b, v in ens.items()}}
    if refs == 'full':
        # the paper pipeline's ensemble: alpha tuned on predictions whose training sets
        # include the held-out family (reproduces the published `ens`)
        pk = out['pkps']
        e = np.where(np.isfinite(pk), pk, sample_mat).copy()
        cep = ce & np.isfinite(pk)
        for f in ufams:
            trm = fams != f
            msk = cep & trm[:, None]
            if msk.sum() >= 5:
                errs = [float(np.nanmean(np.abs(np.clip(a * sample_mat[msk] + (1 - a) * pk[msk], 0, 1)
                                                - truth[msk]))) for a in ALPHAS]
                a_f = float(ALPHAS[int(np.argmin(errs))])
            else:
                a_f = 0.5
            blend = np.clip(a_f * sample_mat + (1 - a_f) * pk, 0, 1)
            e = np.where(cep & (~trm)[:, None], blend, e)
        preds['ens_pkps_paper'] = e

    rows = []
    for t, tt in enumerate(tasks):
        for meth, Pm in preds.items():
            h = obs[:, t] & np.isfinite(Pm[:, t]) & np.isfinite(truth[:, t])
            rows.append(dict(m=m, n_models=n, p_task=p_task, refs=refs, seed=seed, task=tt,
                             method=meth, n_cells=int(h.sum()),
                             mae=float(np.abs(Pm[h, t] - truth[h, t]).mean()) if h.any() else np.nan))
    info = dict(m=m, n_models=n, p_task=p_task, refs=refs, seed=seed,
                sig_mult=float(sig_star / qmed) if sig_star is not None else np.nan,
                sig_res=float(sig_res / qmed) if sig_res is not None else np.nan,
                sig_loc=float(sig_loc / qmed) if sig_loc is not None else np.nan,
                lam=float(np.mean(lam_used)),
                **{f'w_{nm}': float(np.mean(v)) if v else np.nan for nm, v in w_used.items()},
                **{f'g_{nm}': float(np.mean(v)) if v else np.nan for nm, v in g_used.items()},
                **{f'alpha_{b}': float(np.mean(v)) if v else np.nan for b, v in alpha_used.items()})
    return rows, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['helm', 'eee'], default='eee')
    ap.add_argument('--budgets', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--refs', choices=['full', 'sample'], nargs='+', default=['full', 'sample'])
    ap.add_argument('--n_models', type=int, default=None)
    ap.add_argument('--p_task', type=float, default=1.0)
    ap.add_argument('--n_seeds', type=int, default=16)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    data = H.load_suite() if args.suite == 'helm' else H.load_eee()
    print(f'suite {args.suite}: {len(data[7])} models, {len(data[8])} tasks')
    jobs = [delayed(run)(data, m, s, n_models=args.n_models, p_task=args.p_task, refs=r)
            for r in args.refs for m in args.budgets for s in range(args.n_seeds)]
    res = Parallel(n_jobs=args.n_jobs, verbose=2)(jobs)
    out = Path('results-check'); out.mkdir(exist_ok=True)
    stem = f'additive_{args.suite}{args.tag}'
    pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(out / f'{stem}.csv', index=False)
    pd.DataFrame([i for _, i in res]).to_csv(out / f'{stem}_info.csv', index=False)
    print('wrote', out / f'{stem}.csv')


if __name__ == '__main__':
    main()
