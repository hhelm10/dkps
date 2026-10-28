"""Phase-2 lever sweep: pairing, estimators, candidate scaling, flagship
rank, exchangeability, conservative gates.

Three collection passes over the combined pool; every lever is a post-hoc
slice of the saved decision tables:

    collect_conf     confidence functionals for the qa pick (min-distance,
                     margin d1/d2, effective local sample sizes, paired-
                     overlap deviation regression), oracle ceilings,
                     candidate-count subsets of the unrestricted pool,
                     flagship rank variants (best / 5th / median per suite)
    collect_pairing  HELM-only pairing sweep at FIXED 50% cache per model:
                     candidate caches aligned with / independent of /
                     disjoint from the flagship's (pairwise overlap
                     100% / 50% / 0%) -- decides whether the paired-overlap
                     estimator is admissible in the unpaired regime
    collect_novel    novel-task traffic: whole tasks absent from the cache;
                     does confidence flag them (gate abstains) or fail
                     silently (violations spike)?

Post-hoc: conformal gate with empirical vs Clopper-Pearson (UCB) cutoff
rules at several calibration sizes; task-shifted calibration re-splits.

Run from repo root: pixi run python -m projects.routing.experiments.lever_sweep2
"""

import os

import numpy as np
import pandas as pd

from .run_cost_routing import parse_params
from .run_helm import RESULTS

SIG = 0.25
K_SUBSETS = (2, 4, 8, 16, 32, 64)
CONTRACT = (0.5, 0.10)


def batched_stats2(Xa, sa, model_groups, n, W):
    """phi, shat, effective local sample size, ok -- per model, batched."""
    q = W.shape[0]
    d = Xa.shape[1]
    phi = np.zeros((q, n, d), dtype=np.float32)
    shat = np.zeros((q, n))
    den = np.zeros((q, n))
    den2 = np.zeros((q, n))
    for m, idx in enumerate(model_groups):
        Wm = W[:, idx]
        den[:, m] = Wm.sum(axis=1)
        den2[:, m] = (Wm * Wm).sum(axis=1)
        phi[:, m, :] = Wm @ Xa[idx]
        shat[:, m] = Wm @ sa[idx]
    ok = den > 1e-12
    phi[ok] /= den[ok][:, None]
    shat[ok] /= den[ok]
    neff = np.zeros((q, n))
    neff[ok] = den[ok] ** 2 / np.maximum(den2[ok], 1e-30)
    return phi, shat, neff, ok


def pairdev_est(Xa, code_a, model_groups, flag, n, W, min_shared=3):
    """Localized regression of REALIZED pairwise deviation on the anchors
    both the flagship and the candidate responded to. hat[q, m] = the
    kernel-weighted mean of ||x_m,j - x_flag,j|| over shared anchors j;
    inf where the pair shares < min_shared anchors (starved / unpaired)."""
    q = W.shape[0]
    fmap = {code_a[i]: i for i in model_groups[flag]}
    hat = np.full((q, n), np.inf)
    nsh = np.zeros(n, dtype=int)
    for m in range(n):
        if m == flag:
            continue
        idx_m = [i for i in model_groups[m] if code_a[i] in fmap]
        if len(idx_m) < min_shared:
            continue
        idx_f = [fmap[code_a[i]] for i in idx_m]
        devs = np.linalg.norm(Xa[idx_m] - Xa[np.array(idx_f)], axis=1)
        Wm = W[:, idx_m]
        s = Wm.sum(axis=1)
        okq = s > 1e-12
        hat[okq, m] = (Wm[okq] @ devs) / s[okq]
        nsh[m] = len(idx_m)
    return hat, nsh


def softdev_est(Xa, ua, model_groups, flag, n, W, delta, knn=8,
                cap=4000):
    """Soft-pairdev: the PKPS-native deviation estimator. Each anchor
    response of model m is compared with the flagship's responses to the
    knn most SIMILAR (not identical) queries, RBF-coupled at bandwidth
    delta; the coupled deviations are then localized to q* as in pairdev.
    Needs no shared queries -- the finite-delta member of the family."""
    q = W.shape[0]
    f_idx = model_groups[flag]
    uf = ua[f_idx]
    Xf = Xa[f_idx]
    hat = np.full((q, n), np.inf)
    for m in range(n):
        idx_m = model_groups[m]
        if m == flag or len(idx_m) == 0 or len(f_idx) == 0:
            continue
        if len(idx_m) > cap:
            idx_m = idx_m[np.linspace(0, len(idx_m) - 1, cap).astype(int)]
        um = ua[idx_m]
        D2 = ((um[:, None, :] - uf[None, :, :]) ** 2).sum(-1)
        k = min(knn, D2.shape[1])
        nn = np.argpartition(D2, k - 1, axis=1)[:, :k]
        rows_i = np.arange(len(idx_m))[:, None]
        Kd = np.exp(-D2[rows_i, nn] / (2.0 * delta ** 2))
        devs = np.linalg.norm(Xa[idx_m][:, None, :] - Xf[nn], axis=2)
        sK = Kd.sum(axis=1)
        okj = sK > 1e-12
        if not okj.any():
            continue
        dtilde = (Kd[okj] * devs[okj]).sum(axis=1) / sK[okj]
        Wm = W[:, np.asarray(idx_m)[okj]]
        s = Wm.sum(axis=1)
        okq = s > 1e-12
        hat[okq, m] = (Wm[okq] @ dtilde) / s[okq]
    return hat


def within_curve(Xa, ua, idx, rng, knn=8, n_bins=15, cap=4000):
    """Per-model regression of squared response distance on query
    distance, from the model's OWN kNN pairs -- the SAME matching
    procedure as the cross terms, so the curve is supported exactly on
    the r-range where it is evaluated (unpaired-valid). Returns (bin
    centers, bin means, r->0 intercept)."""
    if len(idx) < 20:
        return None
    idx = np.asarray(idx)
    if len(idx) > cap:
        idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
    U = ua[idx]
    D2 = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    k = min(knn, len(idx) - 1)
    nn = np.argpartition(D2, k - 1, axis=1)[:, :k]
    rows_i = np.arange(len(idx))[:, None]
    r = np.sqrt(D2[rows_i, nn]).ravel()
    Xm = Xa[idx]
    d2 = ((Xm[:, None, :] - Xm[nn]) ** 2).sum(-1).ravel()
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    which = np.clip(np.searchsorted(edges, r, side='right') - 1,
                    0, n_bins - 1)
    means = np.array([d2[which == b_].mean() if (which == b_).any()
                      else np.nan for b_ in range(n_bins)])
    centers = 0.5 * (edges[:-1] + edges[1:])
    fin = np.isfinite(means)
    kf = min(4, int(fin.sum()))
    coef = np.polyfit(centers[fin][:kf], means[fin][:kf], 1)
    w0 = max(float(np.polyval(coef, 0.0)), 0.0)
    return centers, means, w0


def _w_at(curve, r):
    centers, means, _ = curve
    fin = np.isfinite(means)
    return np.interp(r, centers[fin], means[fin])


def softdev_corr_est(Xa, ua, model_groups, flag, n, W, delta, rng,
                     knn=8, cap=4000):
    """Debiased soft-pairdev: cross-query squared deviations minus the
    within-model query-difference penalties (both models' own-cache
    curves), plus the r->0 noise floors. Targets the same-query deviation
    under an additive shared query effect; fully unpaired."""
    q = W.shape[0]
    f_idx = model_groups[flag]
    uf, Xf = ua[f_idx], Xa[f_idx]
    curves = {m: within_curve(Xa, ua, model_groups[m], rng)
              for m in range(n)}
    cf = curves[flag]
    hat = np.full((q, n), np.inf)
    if cf is None:
        return hat
    for m in range(n):
        idx_m = model_groups[m]
        cm = curves[m]
        if m == flag or len(idx_m) == 0 or len(f_idx) == 0 or cm is None:
            continue
        if len(idx_m) > cap:
            idx_m = idx_m[np.linspace(0, len(idx_m) - 1, cap).astype(int)]
        um = ua[idx_m]
        D2 = ((um[:, None, :] - uf[None, :, :]) ** 2).sum(-1)
        k = min(knn, D2.shape[1])
        nn = np.argpartition(D2, k - 1, axis=1)[:, :k]
        rows_i = np.arange(len(idx_m))[:, None]
        r_nn = np.sqrt(D2[rows_i, nn])
        Kd = np.exp(-r_nn ** 2 / (2.0 * delta ** 2))
        d2 = ((Xa[idx_m][:, None, :] - Xf[nn]) ** 2).sum(axis=2)
        corr = d2 - 0.5 * (_w_at(cm, r_nn) + _w_at(cf, r_nn)) \
            + 0.5 * (cm[2] + cf[2])
        corr = np.maximum(corr, 0.0)
        sK = Kd.sum(axis=1)
        okj = sK > 1e-12
        if not okj.any():
            continue
        c_j = (Kd[okj] * corr[okj]).sum(axis=1) / sK[okj]
        Wm = W[:, np.asarray(idx_m)[okj]]
        s = Wm.sum(axis=1)
        okq = s > 1e-12
        hat[okq, m] = np.sqrt((Wm[okq] @ c_j) / s[okq])
    return hat


def _prep(X, Qu, rows, n, names):
    sizes = [parse_params(nm.split(':', 1)[1]) for nm in names]
    suite_of = [nm.split(':', 1)[0] for nm in names]
    gmean = rows.groupby('model')['score'].mean()
    ranked = {su: [int(i) for i in gmean.loc[[i for i in range(n)
                                              if suite_of[i] == su]]
                   .sort_values(ascending=False).index]
              for su in ('helm', 'eee')}
    return sizes, suite_of, gmean, ranked


def _geometry(X, Qu, rows, anchor_m, hold_groups, n, rng):
    Xa = X[anchor_m]
    sa = rows['score'].to_numpy()[anchor_m]
    model_a = rows['model'].to_numpy()[anchor_m]
    code_a = rows['code'].to_numpy()[anchor_m]
    ua = Qu[code_a]
    model_groups = [np.flatnonzero(model_a == m) for m in range(n)]
    i = rng.integers(0, len(ua), 20000)
    k = rng.integers(0, len(ua), 20000)
    kp = i != k
    med = float(np.median(np.linalg.norm(ua[i[kp]] - ua[k[kp]], axis=1)))
    Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in hold_groups])
    D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
    D2 = D2 - D2.min(axis=1, keepdims=True)
    W = np.exp(-D2 / (2.0 * (SIG * med) ** 2)).astype(np.float32)
    return Xa, sa, model_a, code_a, model_groups, W


COLS = ['pool', 'seed', 'variant', 'split', 'suite', 'task', 'query',
        'n_avail', 'method', 'conf', 'margin', 'neff_pick', 'neff_flag',
        'pairdev', 'dev', 'score', 's_flag']


def _qa_row(phi, ok, neff, gi, avail, flag, devrow, srow):
    d = np.linalg.norm(phi[gi][avail] - phi[gi][flag], axis=1)
    d[~ok[gi][avail]] = np.inf
    o = np.argsort(d)
    r = avail[int(o[0])]
    margin = float(d[o[0]] / d[o[1]]) if len(d) > 1 and \
        np.isfinite(d[o[1]]) and d[o[1]] > 0 else 1.0
    return r, float(d[o[0]]), margin, float(neff[gi][r]), \
        float(neff[gi][flag])


def collect_conf(X, Qu, rows, n, names, seed, n_hold=800):
    sizes, suite_of, gmean, ranked = _prep(X, Qu, rows, n, names)
    rng = np.random.default_rng(seed)
    variants = {}
    for su in ('helm', 'eee'):
        rk = ranked[su]
        variants[(su, 'r1')] = rk[0]
        variants[(su, 'r5')] = rk[4]
        variants[(su, 'rmed')] = rk[len(rk) // 2]
    r1 = {su: variants[(su, 'r1')] for su in ('helm', 'eee')}
    pools = {'le13b': {i for i in range(n) if sizes[i] is not None
                       and sizes[i] <= 13.0},
             'le40b': {i for i in range(n) if sizes[i] is not None
                       and sizes[i] <= 40.0},
             'all': set(range(n))}

    resp = rows.groupby('query')['model'].agg(set)
    suite_q = rows.drop_duplicates('query').set_index('query')['suite']
    elig = [q for q, ms in resp.items()
            if r1[suite_q[q]] in ms
            and len((ms & pools['le13b']) - set(r1.values())) >= 1]
    k_hold = min(n_hold, len(elig))
    take = rng.choice(len(elig), size=k_hold, replace=False)
    eval_q = {elig[i] for i in take[:k_hold // 2]}
    hold_q = eval_q | {elig[i] for i in take[k_hold // 2:]}
    is_hold = rows['query'].isin(hold_q).to_numpy()
    anchor_m = ~is_hold
    assert not rows['query'][anchor_m].isin(hold_q).any()

    hold_groups = list(rows[is_hold].groupby('query'))
    Xa, sa, model_a, code_a, model_groups, W = _geometry(
        X, Qu, rows, anchor_m, hold_groups, n, rng)
    phi, shat, neff, ok = batched_stats2(Xa, sa, model_groups, n, W)
    gmean_a = pd.Series(sa).groupby(pd.Series(model_a)).mean()
    pd_hat = {v: pairdev_est(Xa, code_a, model_groups, f, n, W)[0]
              for v, f in variants.items()}
    ksub = {K: set(rng.choice(sorted(pools['all'] - set(r1.values())),
                              size=K, replace=False))
            for K in K_SUBSETS}

    out = []
    for gi, (q, g) in enumerate(hold_groups):
        suite = g['suite'].iloc[0]
        task = g['task'].iloc[0]
        gm = g['model'].to_numpy()
        srow = dict(zip(gm, g['score'].to_numpy()))
        Xq = X[g.index.to_numpy()]
        split = 'eval' if q in eval_q else 'cal'

        for rank in ('r1', 'r5', 'rmed'):
            flag = variants[(suite, rank)]
            if flag not in srow:
                continue
            x_flag = Xq[list(gm).index(flag)]
            devrow = {m: float(np.linalg.norm(Xq[i] - x_flag))
                      for i, m in enumerate(gm)}
            hat = pd_hat[(suite, rank)]
            for pname, pl in pools.items():
                avail = sorted((set(gm) & pl) - {flag})
                if not avail:
                    continue
                na = len(avail)
                base = (pname, seed, rank, split, suite, task, q, na)

                r, dmin, marg, np_, nf = _qa_row(phi, ok, neff, gi, avail,
                                                 flag, devrow, srow)
                pdv = float(hat[gi][r])
                out.append(base + ('qa', dmin, marg, np_, nf, pdv,
                                   devrow[r], srow[r], srow[flag]))
                hv = hat[gi][avail]
                r_pd = avail[int(np.argmin(hv))]
                out.append(base + ('pairdev', float(np.min(hv)), np.nan,
                                   np.nan, np.nan, np.nan,
                                   devrow[r_pd], srow[r_pd], srow[flag]))
                out.append(base + ('oracle-conf', devrow[r], np.nan, np.nan,
                                   np.nan, np.nan, devrow[r], srow[r],
                                   srow[flag]))
                r_op = min(avail, key=lambda a: devrow[a])
                out.append(base + ('oracle-pick', devrow[r_op], np.nan,
                                   np.nan, np.nan, np.nan, devrow[r_op],
                                   srow[r_op], srow[flag]))
                r_ld = avail[int(np.argmax(
                    [float(gmean_a.get(a, -np.inf)) for a in avail]))]
                out.append(base + ('lead', float(rng.random()), np.nan,
                                   np.nan, np.nan, np.nan, devrow[r_ld],
                                   srow[r_ld], srow[flag]))
                r_rd = avail[rng.integers(0, na)]
                out.append(base + ('random', float(rng.random()), np.nan,
                                   np.nan, np.nan, np.nan, devrow[r_rd],
                                   srow[r_rd], srow[flag]))

            if rank != 'r1':
                continue
            hat_r1 = pd_hat[(suite, 'r1')]
            for K, sub in ksub.items():
                avail = sorted((set(gm) & sub) - {flag})
                if not avail:
                    continue
                base = (f'k{K}', seed, 'r1', split, suite, task, q,
                        len(avail))
                r, dmin, marg, np_, nf = _qa_row(phi, ok, neff, gi, avail,
                                                 flag, devrow, srow)
                out.append(base + ('qa', dmin, marg, np_, nf,
                                   float(hat_r1[gi][r]), devrow[r],
                                   srow[r], srow[flag]))
                r_op = min(avail, key=lambda a: devrow[a])
                out.append(base + ('oracle-pick', devrow[r_op], np.nan,
                                   np.nan, np.nan, np.nan, devrow[r_op],
                                   srow[r_op], srow[flag]))
                r_rd = avail[rng.integers(0, len(avail))]
                out.append(base + ('random', float(rng.random()), np.nan,
                                   np.nan, np.nan, np.nan, devrow[r_rd],
                                   srow[r_rd], srow[flag]))
    return pd.DataFrame(out, columns=COLS)


def collect_pairing(X, Qu, rows, n, names, seed, alignment, n_hold=800,
                    keep=0.5):
    """HELM only, fixed 50% cache per model; the candidates' kept anchor
    queries are aligned with / independent of / disjoint from the
    flagship's kept set (pairwise overlap 100% / 50% / 0%)."""
    sizes, suite_of, gmean, ranked = _prep(X, Qu, rows, n, names)
    rng = np.random.default_rng(seed)
    flag = ranked['helm'][0]
    helm_models = {i for i in range(n) if suite_of[i] == 'helm'}
    pools = {'le13b': {i for i in helm_models if sizes[i] is not None
                       and sizes[i] <= 13.0},
             'all': helm_models}

    is_helm = (rows['suite'] == 'helm').to_numpy()
    resp = rows[is_helm].groupby('query')['model'].agg(set)
    elig = [q for q, ms in resp.items()
            if flag in ms and len((ms & pools['le13b']) - {flag}) >= 1]
    k_hold = min(n_hold, len(elig))
    take = rng.choice(len(elig), size=k_hold, replace=False)
    eval_q = {elig[i] for i in take[:k_hold // 2]}
    hold_q = eval_q | {elig[i] for i in take[k_hold // 2:]}
    is_hold = rows['query'].isin(hold_q).to_numpy()

    anchor_codes = np.unique(rows['code'].to_numpy()[is_helm & ~is_hold])
    S = set(rng.choice(anchor_codes, size=int(keep * len(anchor_codes)),
                       replace=False))
    Sc = set(anchor_codes) - S
    code_arr = rows['code'].to_numpy()
    model_arr = rows['model'].to_numpy()
    keep_m = np.zeros(len(rows), dtype=bool)
    base_m = is_helm & ~is_hold
    for m in helm_models:
        if m == flag or alignment == 'aligned':
            kset = S
        elif alignment == 'disjoint':
            kset = Sc
        else:                                   # independent
            kset = set(rng.choice(anchor_codes,
                                  size=int(keep * len(anchor_codes)),
                                  replace=False))
        rows_m = base_m & (model_arr == m)
        keep_m |= rows_m & np.isin(code_arr, list(kset))
    anchor_m = keep_m
    assert not rows['query'][anchor_m].isin(hold_q).any()

    hold_groups = list(rows[is_hold & is_helm].groupby('query'))
    Xa, sa, model_a, code_a, model_groups, W = _geometry(
        X, Qu, rows, anchor_m, hold_groups, n, rng)
    phi, shat, neff, ok = batched_stats2(Xa, sa, model_groups, n, W)
    hat, nsh = pairdev_est(Xa, code_a, model_groups, flag, n, W)
    ua = Qu[code_a]
    ii = rng.integers(0, len(ua), 20000)
    kk = rng.integers(0, len(ua), 20000)
    kpp = ii != kk
    med = float(np.median(np.linalg.norm(ua[ii[kpp]] - ua[kk[kpp]],
                                         axis=1)))
    soft = {f'soft-{c:g}x': softdev_est(Xa, ua, model_groups, flag, n, W,
                                        delta=c * med)
            for c in (0.05, 0.15)}
    soft['soft-corr'] = softdev_corr_est(Xa, ua, model_groups, flag, n,
                                         W, delta=0.05 * med, rng=rng)

    out = []
    for gi, (q, g) in enumerate(hold_groups):
        gm = g['model'].to_numpy()
        srow = dict(zip(gm, g['score'].to_numpy()))
        Xq = X[g.index.to_numpy()]
        x_flag = Xq[list(gm).index(flag)]
        devrow = {m: float(np.linalg.norm(Xq[i] - x_flag))
                  for i, m in enumerate(gm)}
        split = 'eval' if q in eval_q else 'cal'
        task = g['task'].iloc[0]
        for pname, pl in pools.items():
            avail = sorted((set(gm) & pl) - {flag})
            if not avail:
                continue
            base = (pname, seed, alignment, split, 'helm', task, q,
                    len(avail))
            r, dmin, marg, np_, nf = _qa_row(phi, ok, neff, gi, avail,
                                             flag, devrow, srow)
            out.append(base + ('qa', dmin, marg, np_, nf,
                               float(hat[gi][r]), devrow[r], srow[r],
                               srow[flag]))
            hv = hat[gi][avail]
            if np.isfinite(hv).any():
                r_pd = avail[int(np.argmin(hv))]
                out.append(base + ('pairdev', float(np.min(hv)), np.nan,
                                   np.nan, np.nan, np.nan, devrow[r_pd],
                                   srow[r_pd], srow[flag]))
            for sname, shat_m in soft.items():
                sv = shat_m[gi][avail]
                if not np.isfinite(sv).any():
                    continue
                r_sf = avail[int(np.argmin(sv))]
                out.append(base + (sname, float(np.min(sv)), np.nan,
                                   np.nan, np.nan, np.nan, devrow[r_sf],
                                   srow[r_sf], srow[flag]))
            out.append(base + ('oracle-conf', devrow[r], np.nan, np.nan,
                               np.nan, np.nan, devrow[r], srow[r],
                               srow[flag]))
            r_rd = avail[rng.integers(0, len(avail))]
            out.append(base + ('random', float(rng.random()), np.nan,
                               np.nan, np.nan, np.nan, devrow[r_rd],
                               srow[r_rd], srow[flag]))
    return pd.DataFrame(out, columns=COLS)


def collect_novel(X, Qu, rows, n, names, seed, n_novel_tasks=2, n_hold=400):
    """Whole tasks absent from the cache; their queries appear only as
    eval traffic. Gate must abstain on them, not fail silently."""
    sizes, suite_of, gmean, ranked = _prep(X, Qu, rows, n, names)
    rng = np.random.default_rng(seed)
    r1 = {su: ranked[su][0] for su in ('helm', 'eee')}
    pools = {'le13b': {i for i in range(n) if sizes[i] is not None
                       and sizes[i] <= 13.0},
             'all': set(range(n))}

    resp = rows.groupby('query')['model'].agg(set)
    suite_q = rows.drop_duplicates('query').set_index('query')['suite']
    task_q = rows.drop_duplicates('query').set_index('query')['task']
    elig = [q for q, ms in resp.items()
            if r1[suite_q[q]] in ms
            and len((ms & pools['le13b']) - set(r1.values())) >= 1]
    elig_tasks = pd.Series([task_q[q] for q in elig])
    novel = []
    for su in ('helm', 'eee'):
        ts = elig_tasks[elig_tasks.str.startswith(su)].value_counts()
        ts = ts[ts >= 30].index.to_numpy()
        novel += list(rng.choice(ts, size=min(n_novel_tasks, len(ts)),
                                 replace=False))
    novel = set(novel)
    nov_q = [q for q in elig if task_q[q] in novel]
    seen_q = [q for q in elig if task_q[q] not in novel]
    nov_eval = set(rng.choice(nov_q, size=min(200, len(nov_q)),
                              replace=False))
    take = rng.choice(len(seen_q), size=min(n_hold + 200, len(seen_q)),
                      replace=False)
    seen_eval = {seen_q[i] for i in take[:200]}
    cal_q = {seen_q[i] for i in take[200:]}
    hold_q = nov_eval | seen_eval | cal_q
    is_hold = rows['query'].isin(hold_q).to_numpy()
    is_novel_task = rows['task'].isin(novel).to_numpy()
    anchor_m = ~is_hold & ~is_novel_task
    assert not rows['query'][anchor_m].isin(hold_q).any()
    assert not rows['task'][anchor_m].isin(novel).any()

    hold_groups = list(rows[is_hold].groupby('query'))
    Xa, sa, model_a, code_a, model_groups, W = _geometry(
        X, Qu, rows, anchor_m, hold_groups, n, rng)
    phi, shat, neff, ok = batched_stats2(Xa, sa, model_groups, n, W)

    out = []
    for gi, (q, g) in enumerate(hold_groups):
        suite = g['suite'].iloc[0]
        task = g['task'].iloc[0]
        flag = r1[suite]
        gm = g['model'].to_numpy()
        if flag not in gm:
            continue
        srow = dict(zip(gm, g['score'].to_numpy()))
        Xq = X[g.index.to_numpy()]
        x_flag = Xq[list(gm).index(flag)]
        devrow = {m: float(np.linalg.norm(Xq[i] - x_flag))
                  for i, m in enumerate(gm)}
        split = 'cal' if q in cal_q else \
            ('eval-novel' if q in nov_eval else 'eval-seen')
        for pname, pl in pools.items():
            avail = sorted((set(gm) & pl) - {flag})
            if not avail:
                continue
            base = (pname, seed, 'novel', split, suite, task, q, len(avail))
            r, dmin, marg, np_, nf = _qa_row(phi, ok, neff, gi, avail,
                                             flag, devrow, srow)
            out.append(base + ('qa', dmin, marg, np_, nf, np.nan,
                               devrow[r], srow[r], srow[flag]))
            r_rd = avail[rng.integers(0, len(avail))]
            out.append(base + ('random', float(rng.random()), np.nan,
                               np.nan, np.nan, np.nan, devrow[r_rd],
                               srow[r_rd], srow[flag]))
    return pd.DataFrame(out, columns=COLS)


def gate2(cal_conf, cal_dev, ev_conf, eps, alpha, rule='emp',
          min_prefix=20, delta=0.10):
    """Confidence cutoff from calibration; rule 'emp' thresholds the
    empirical violation rate, 'ucb' the Clopper-Pearson upper bound."""
    o = np.argsort(cal_conf)
    conf_s, bad = cal_conf[o], (cal_dev[o] > eps).astype(float)
    k = np.cumsum(bad)
    m = np.arange(1, len(bad) + 1)
    if rule == 'emp':
        crit = k / m
    else:
        from scipy.stats import beta
        crit = np.where(k < m, beta.ppf(1 - delta, k + 1, m - k), 1.0)
    okc = np.flatnonzero(crit <= alpha)
    okc = okc[okc >= min_prefix - 1]
    t = conf_s[okc.max()] if len(okc) else -np.inf
    return ev_conf <= t


def main():
    from .run_combined import load_combined
    os.makedirs(RESULTS, exist_ok=True)
    X, Qu, rows, n, names = load_combined()
    print(f'combined: {len(rows)} rows, {n} models', flush=True)

    df = pd.concat([collect_conf(X, Qu, rows, n, names, s)
                    for s in range(5)], ignore_index=True)
    df.to_parquet(os.path.join(RESULTS, 'lever2_conf.parquet'))
    print('conf pass done', flush=True)

    dfp = pd.concat([collect_pairing(X, Qu, rows, n, names, s, al)
                     for al in ('aligned', 'independent', 'disjoint')
                     for s in range(3)], ignore_index=True)
    dfp.to_parquet(os.path.join(RESULTS, 'lever2_pairing.parquet'))
    print('pairing pass done', flush=True)

    dfn = pd.concat([collect_novel(X, Qu, rows, n, names, s)
                     for s in range(3)], ignore_index=True)
    dfn.to_parquet(os.path.join(RESULTS, 'lever2_novel.parquet'))
    print('novel pass done', flush=True)


if __name__ == '__main__':
    main()
