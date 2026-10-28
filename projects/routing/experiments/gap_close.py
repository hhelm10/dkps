"""Closing the qa -> pairdev gap with unpaired and semi-paired confidence.

The gated quantity decomposes as

    E||x_m - x_f||^2 = ||mu_m - mu_f||^2 + tr S_m + tr S_f - 2 tr Cov(m,f)

Means and per-model variances are unpaired-estimable (PKPS-style, each
model's own pool); the co-movement term is a joint moment that requires
shared queries. The estimator ladder:

    qa        ||phi_m - phi_f||                  unpaired (means only)
    var-ub    sqrt(d^2 + s2_m + s2_f)            unpaired upper bound (Cov=0)
    var-norm  d / sqrt(s2_m + s2_f)              unpaired signal-to-noise
    pd-K      pairdev on K purchased shared      semi-paired: K generations
              anchors per candidate              of each candidate on own
                                                 traffic (K = 10/30/100)
    pairdev   full shared-anchor regression      paired ceiling
    oracle    realized deviation                 cheating reference

plus calibration-fit linear combos of the features (leak-free: fit on the
cal split only). Run from repo root:
pixi run python -m projects.routing.experiments.gap_close
"""

import os

import numpy as np
import pandas as pd

from .lever_sweep2 import (_geometry, _prep, gate2, pairdev_est,
                           softdev_est)
from .run_cost_routing import parse_params
from .run_helm import RESULTS

KS = (10, 30, 100)
CONTRACT = (0.5, 0.10)


def batched_stats3(Xa, sa, model_groups, n, W):
    """phi, localized residual variance (tr of it), n_eff, ok."""
    q = W.shape[0]
    d = Xa.shape[1]
    sq = (Xa.astype(np.float64) ** 2).sum(axis=1)
    phi = np.zeros((q, n, d), dtype=np.float32)
    m2 = np.zeros((q, n))
    den = np.zeros((q, n))
    den2 = np.zeros((q, n))
    for m, idx in enumerate(model_groups):
        Wm = W[:, idx]
        den[:, m] = Wm.sum(axis=1)
        den2[:, m] = (Wm * Wm).sum(axis=1)
        phi[:, m, :] = Wm @ Xa[idx]
        m2[:, m] = Wm @ sq[idx]
    ok = den > 1e-12
    phi[ok] /= den[ok][:, None]
    m2[ok] /= den[ok]
    s2 = np.maximum(m2 - (phi.astype(np.float64) ** 2).sum(-1), 0.0)
    neff = np.zeros((q, n))
    neff[ok] = den[ok] ** 2 / np.maximum(den2[ok], 1e-30)
    return phi, s2, neff, ok


def pairdev_sub(Xa, code_a, model_groups, flag, n, W, K, rng,
                min_shared=3):
    """pairdev restricted to K randomly purchased shared anchors per
    candidate (the K-generations-per-candidate budget)."""
    fmap = {code_a[i]: i for i in model_groups[flag]}
    q = W.shape[0]
    hat = np.full((q, n), np.inf)
    for m in range(n):
        if m == flag:
            continue
        idx_m = [i for i in model_groups[m] if code_a[i] in fmap]
        if len(idx_m) < min_shared:
            continue
        if len(idx_m) > K:
            idx_m = list(rng.choice(idx_m, size=K, replace=False))
        idx_f = np.array([fmap[code_a[i]] for i in idx_m])
        devs = np.linalg.norm(Xa[idx_m] - Xa[idx_f], axis=1)
        Wm = W[:, idx_m]
        s = Wm.sum(axis=1)
        okq = s > 1e-12
        hat[okq, m] = (Wm[okq] @ devs) / s[okq]
    return hat


def collect(X, Qu, rows, n, names, seed, n_hold=800, n_eval=None,
            n_calA=None, n_calB=None, light=False):
    sizes = [parse_params(nm.split(':', 1)[1]) for nm in names]
    _, suite_of, gmean, ranked = _prep(X, Qu, rows, n, names)
    flags = {su: ranked[su][0] for su in ('helm', 'eee')}
    rng = np.random.default_rng(seed)
    pools = {'le13b': {i for i in range(n) if sizes[i] is not None
                       and sizes[i] <= 13.0},
             'all': set(range(n))}

    resp = rows.groupby('query')['model'].agg(set)
    suite_q = rows.drop_duplicates('query').set_index('query')['suite']
    elig = [q for q, ms in resp.items()
            if flags[suite_q[q]] in ms
            and len((ms & pools['le13b']) - set(flags.values())) >= 1]
    if n_eval is None:
        k_hold = min(n_hold, len(elig))
        n_eval = k_hold // 2
        n_calA = n_calB = k_hold // 4
    k_hold = n_eval + n_calA + n_calB
    assert k_hold <= len(elig), (k_hold, len(elig))
    take = rng.choice(len(elig), size=k_hold, replace=False)
    eval_q = {elig[i] for i in take[:n_eval]}
    # calA: the paired sample the gate's calibration already requires,
    # recycled as pd-cal's shared-anchor pool; calB calibrates the gate
    calA_q = {elig[i] for i in take[n_eval:n_eval + n_calA]}
    calB_q = {elig[i] for i in take[n_eval + n_calA:]}
    hold_q = eval_q | calA_q | calB_q
    is_hold = rows['query'].isin(hold_q).to_numpy()
    anchor_m = ~is_hold
    assert not rows['query'][anchor_m].isin(hold_q).any()

    hold_groups = list(rows[is_hold].groupby('query'))
    Xa, sa, model_a, code_a, model_groups, W = _geometry(
        X, Qu, rows, anchor_m, hold_groups, n, rng)
    phi, s2, neff, ok = batched_stats3(Xa, sa, model_groups, n, W)
    pd_full = {su: pairdev_est(Xa, code_a, model_groups, f, n, W)[0]
               for su, f in flags.items()}
    pd_k = {} if light else {
        (su, K): pairdev_sub(Xa, code_a, model_groups, f, n, W, K, rng)
        for su, f in flags.items() for K in KS}

    # pd-cal: the calibration sample the contract already requires (calA),
    # recycled as the pairdev anchor pool; calB alone calibrates gates
    calA_m = rows['query'].isin(calA_q).to_numpy()
    Xc = X[calA_m]
    code_c = rows['code'].to_numpy()[calA_m]
    model_c = rows['model'].to_numpy()[calA_m]
    groups_c = [np.flatnonzero(model_c == m) for m in range(n)]
    ua_c = Qu[code_c]
    ua = Qu[code_a]
    i = rng.integers(0, len(ua), 20000)
    k = rng.integers(0, len(ua), 20000)
    kp = i != k
    med = float(np.median(np.linalg.norm(ua[i[kp]] - ua[k[kp]], axis=1)))
    Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in hold_groups])
    D2c = ((Ue[:, None, :] - ua_c[None, :, :]) ** 2).sum(-1)
    D2c = D2c - D2c.min(axis=1, keepdims=True)
    Wc = np.exp(-D2c / (2.0 * (0.25 * med) ** 2)).astype(np.float32)
    ones_c = np.ones((1, len(Xc)), dtype=np.float32)
    ones_a = np.ones((1, len(Xa)), dtype=np.float32)
    pd_cal = {su: pairdev_est(Xc, code_c, groups_c, f, n, Wc)[0]
              for su, f in flags.items()}
    pd_cal_st = {} if light else {
        su: pairdev_est(Xc, code_c, groups_c, f, n, ones_c)[0]
        for su, f in flags.items()}
    pd30_st = {} if light else {
        su: pairdev_sub(Xa, code_a, model_groups, f, n, ones_a, 30, rng)
        for su, f in flags.items()}
    # soft-pairdev under the SAME protocol (finite-delta coupling)
    soft_h = {su: softdev_est(Xa, ua, model_groups, f, n, W,
                              delta=0.05 * med)
              for su, f in flags.items()}

    out = []
    for gi, (q, g) in enumerate(hold_groups):
        suite = g['suite'].iloc[0]
        flag = flags[suite]
        gm = g['model'].to_numpy()
        srow = dict(zip(gm, g['score'].to_numpy()))
        Xq = X[g.index.to_numpy()]
        x_flag = Xq[list(gm).index(flag)]
        devrow = {m: float(np.linalg.norm(Xq[i] - x_flag))
                  for i, m in enumerate(gm)}
        split = 'eval' if q in eval_q else ('cal' if q in calB_q
                                            else 'calA')
        for pname, pl in pools.items():
            avail = sorted((set(gm) & pl) - set(flags.values()))
            if not avail:
                continue
            av = np.array(avail)
            d = np.linalg.norm(phi[gi][av] - phi[gi][flag], axis=1)
            d[~ok[gi][av]] = np.inf
            s2sum = s2[gi][av] + s2[gi][flag]
            varub = np.sqrt(d ** 2 + s2sum)
            varnorm = d / np.sqrt(s2sum + 1e-12)
            confs = {'qa': d, 'pairdev': pd_full[suite][gi][av],
                     'pd-cal': pd_cal[suite][gi][av],
                     'soft': soft_h[suite][gi][av]}
            if not light:
                confs.update({'var-ub': varub, 'var-norm': varnorm,
                              'pd-cal-st': pd_cal_st[suite][0][av],
                              'pd30-st': pd30_st[suite][0][av]})
                for K in KS:
                    confs[f'pd{K}'] = pd_k[(suite, K)][gi][av]
            base = (pname, seed, split, suite, q)
            j_qa = int(np.argmin(d))
            feat = [float(d[j_qa]), float(np.sqrt(s2sum[j_qa])),
                    float(confs['pairdev'][j_qa]),
                    float(confs['pd-cal'][j_qa]),
                    float(confs['pd30'][j_qa]) if not light else np.nan]
            for meth, cv in confs.items():
                j = int(np.argmin(cv))
                r = int(av[j])
                rf = feat if meth == 'qa' else [np.nan] * 5
                out.append(base + (meth, float(cv[j]), devrow[r],
                                   srow[r], srow[flag], *rf))
            r_qa = int(av[j_qa])
            out.append(base + ('oracle-conf', devrow[r_qa], devrow[r_qa],
                               srow[r_qa], srow[flag],
                               *[np.nan] * 5))
    return pd.DataFrame(out, columns=[
        'pool', 'seed', 'split', 'suite', 'query', 'method', 'conf',
        'dev', 'score', 's_flag', 'f_d', 'f_s2', 'f_pdfull', 'f_pdcal',
        'f_pd30'])


def analyze(df, eps=CONTRACT[0], alpha=CONTRACT[1]):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    def run_gate(ss, cc, ce):
        cc, ce = (np.where(np.isfinite(x), x, 1e6) for x in (cc, ce))
        cal, ev = ss[ss.split == 'cal'], ss[ss.split == 'eval']
        off = gate2(cc, cal['dev'].to_numpy(), ce, eps, alpha)
        dv, sc, fl = (ev[k].to_numpy() for k in ('dev', 'score', 's_flag'))
        return (float(off.mean()),
                float((dv[off] > eps).mean()) if off.any() else 0.0,
                float(np.where(off, sc, fl).mean() / fl.mean()),
                float(spearmanr(ce, dv).statistic))

    print(f'== contract ({eps}, {alpha:.0%}); vol / viol / ret / '
          'spearman ==')
    for pool in ('le13b', 'all'):
        print(f'-- pool {pool} --')
        sub = df[df.pool == pool]
        for meth in ('qa', 'var-ub', 'var-norm', 'pd10', 'pd30', 'pd100',
                     'pd30-st', 'pd-cal', 'pd-cal-st', 'pairdev',
                     'oracle-conf'):
            per_seed = []
            for _, ss in sub[sub.method == meth].groupby('seed'):
                per_seed.append(run_gate(
                    ss, ss[ss.split == 'cal']['conf'].to_numpy(),
                    ss[ss.split == 'eval']['conf'].to_numpy()))
            v = np.mean(per_seed, axis=0)
            print(f'   {meth:11s} vol {v[0]:.3f}  viol {v[1]:.3f}  '
                  f'ret {v[2]:.3f}  spearman {v[3]:.3f}')
        # cal-fit combos on the qa pick: unpaired and semi-paired feature sets
        qa = sub[sub.method == 'qa']
        for name, cols in (('combo-unpaired', ['f_d', 'f_s2']),
                           ('combo-pdcal', ['f_d', 'f_s2', 'f_pdcal']),
                           ('combo-pd30', ['f_d', 'f_s2', 'f_pd30'])):
            per_seed = []
            for _, ss in qa.groupby('seed'):
                cal, ev = ss[ss.split == 'cal'], ss[ss.split == 'eval']

                def F(d):
                    M = d[cols].to_numpy()
                    fin = np.isfinite(M)
                    cap = np.nanmax(np.where(fin, M, np.nan), axis=0)
                    return np.where(fin, M, cap)
                reg = LinearRegression().fit(F(cal), cal['dev'].to_numpy())
                per_seed.append(run_gate(ss, reg.predict(F(cal)),
                                         reg.predict(F(ev))))
            v = np.mean(per_seed, axis=0)
            print(f'   {name:11s} vol {v[0]:.3f}  viol {v[1]:.3f}  '
                  f'ret {v[2]:.3f}  spearman {v[3]:.3f}')


def run_cal_curve(X, Qu, rows, n, names, sizes=(50, 100, 200, 400, 800,
                                                1600), seeds=3,
                  eps=CONTRACT[0], alpha=CONTRACT[1]):
    """Empirical sufficiency curve: certified volume vs total paired
    sample N (split half pd-cal anchors / half gate calibration)."""
    from scipy.stats import spearmanr  # noqa: F401

    def one_gate(ss):
        cc = ss[ss.split == 'cal']['conf'].to_numpy()
        ce = ss[ss.split == 'eval']['conf'].to_numpy()
        cc, ce = (np.where(np.isfinite(x), x, 1e6) for x in (cc, ce))
        off = gate2(cc, ss[ss.split == 'cal']['dev'].to_numpy(), ce,
                    eps, alpha)
        dv = ss[ss.split == 'eval']['dev'].to_numpy()
        return (float(off.mean()),
                float((dv[off] > eps).mean()) if off.any() else 0.0)

    out = []
    for N in sizes:
        try:
            dfs = [collect(X, Qu, rows, n, names, s, n_eval=800,
                           n_calA=N // 2, n_calB=N - N // 2, light=True)
                   for s in range(seeds)]
        except AssertionError as e:
            print(f'N={N}: not enough eligible queries ({e}); stopping')
            break
        df = pd.concat(dfs, ignore_index=True)
        for pool in ('le13b', 'all'):
            for meth in ('qa', 'pd-cal', 'pairdev'):
                sub = df[(df.pool == pool) & (df.method == meth)]
                per = [one_gate(ss) for _, ss in sub.groupby('seed')]
                v = np.mean(per, axis=0)
                se = np.std([p[0] for p in per]) / np.sqrt(len(per))
                out.append({'N': N, 'pool': pool, 'method': meth,
                            'vol': v[0], 'viol': v[1], 'vol_se': se})
    cur = pd.DataFrame(out)
    cur.to_parquet(os.path.join(RESULTS, 'cal_curve.parquet'))
    for pool in ('le13b', 'all'):
        print(f'\n== paired-sample curve, pool {pool} '
              f'(vol @ ({eps}, {alpha:.0%}); viol in parens) ==')
        piv_v = cur[cur.pool == pool].pivot(index='method', columns='N',
                                            values='vol')
        piv_x = cur[cur.pool == pool].pivot(index='method', columns='N',
                                            values='viol')
        for m in piv_v.index:
            cells = '  '.join(
                f'{piv_v.loc[m, c]:.2f}({piv_x.loc[m, c]:.2f})'
                for c in piv_v.columns)
            print(f'  {m:8s} {cells}')
    return cur


def run_cal_curve_nested(X, Qu, rows, n, names,
                         sizes=(100, 200, 400, 800, 1100), seeds=10,
                         n_eval=800, eps=CONTRACT[0], alpha=CONTRACT[1]):
    """Fast + paired version of the sufficiency curve: ONE geometry per
    seed (hold-out = eval + maximal calA + maximal calB), then each N
    reuses nested subsets calA_N / calB_N. qa and pairdev confidences do
    not depend on the calibration sample at all; pd-cal is recomputed per
    N over the (small) calA_N anchor rows. Cross-N comparisons are paired
    within seed."""
    from .run_eee import batched_localized_stats  # noqa: F401
    sizes = sorted(sizes)
    Nmax = sizes[-1]
    szs = [parse_params(nm.split(':', 1)[1]) for nm in names]
    _, suite_of, gmean, ranked = _prep(X, Qu, rows, n, names)
    flags = {su: ranked[su][0] for su in ('helm', 'eee')}
    pools = {'le13b': {i for i in range(n) if szs[i] is not None
                       and szs[i] <= 13.0},
             'all': set(range(n))}

    resp = rows.groupby('query')['model'].agg(set)
    suite_q = rows.drop_duplicates('query').set_index('query')['suite']
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        elig = [q for q, ms in resp.items()
                if flags[suite_q[q]] in ms
                and len((ms & pools['le13b']) - set(flags.values())) >= 1]
        k_hold = n_eval + Nmax
        assert k_hold <= len(elig), (k_hold, len(elig))
        take = rng.choice(len(elig), size=k_hold, replace=False)
        eval_q = {elig[i] for i in take[:n_eval]}
        calA_all = [elig[i] for i in take[n_eval:n_eval + Nmax // 2]]
        calB_all = [elig[i] for i in take[n_eval + Nmax // 2:]]
        hold_q = eval_q | set(calA_all) | set(calB_all)
        is_hold = rows['query'].isin(hold_q).to_numpy()
        anchor_m = ~is_hold
        assert not rows['query'][anchor_m].isin(hold_q).any()

        # geometry over eval + calB queries only (calA never needs conf)
        need_conf = eval_q | set(calB_all)
        is_conf = rows['query'].isin(need_conf).to_numpy()
        hold_groups = list(rows[is_conf].groupby('query'))
        Xa, sa, model_a, code_a, model_groups, W = _geometry(
            X, Qu, rows, anchor_m, hold_groups, n, rng)
        phi, s2, neff, ok = batched_stats3(Xa, sa, model_groups, n, W)
        pd_full = {su: pairdev_est(Xa, code_a, model_groups, f, n, W)[0]
                   for su, f in flags.items()}
        ua = Qu[code_a]
        ii = rng.integers(0, len(ua), 20000)
        kk = rng.integers(0, len(ua), 20000)
        kp = ii != kk
        med = float(np.median(np.linalg.norm(ua[ii[kp]] - ua[kk[kp]],
                                             axis=1)))
        Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in hold_groups])

        # per-N pd-cal hats over nested calA subsets
        pd_cal_N = {}
        for N in sizes:
            calA_m = rows['query'].isin(set(calA_all[:N // 2])).to_numpy()
            Xc = X[calA_m]
            code_c = rows['code'].to_numpy()[calA_m]
            model_c = rows['model'].to_numpy()[calA_m]
            groups_c = [np.flatnonzero(model_c == m) for m in range(n)]
            ua_c = Qu[code_c]
            D2c = ((Ue[:, None, :] - ua_c[None, :, :]) ** 2).sum(-1)
            D2c = D2c - D2c.min(axis=1, keepdims=True)
            Wc = np.exp(-D2c / (2.0 * (0.25 * med) ** 2)) \
                .astype(np.float32)
            pd_cal_N[N] = {su: pairdev_est(Xc, code_c, groups_c, f, n,
                                           Wc)[0]
                           for su, f in flags.items()}

        # per-decision records: conf per method (pd-cal keyed by N)
        recs = []
        for gi, (q, g) in enumerate(hold_groups):
            suite = g['suite'].iloc[0]
            flag = flags[suite]
            gm = g['model'].to_numpy()
            srow = dict(zip(gm, g['score'].to_numpy()))
            Xq = X[g.index.to_numpy()]
            x_flag = Xq[list(gm).index(flag)]
            devrow = {m: float(np.linalg.norm(Xq[i] - x_flag))
                      for i, m in enumerate(gm)}
            split = 'eval' if q in eval_q else 'cal'
            for pname, pl in pools.items():
                avail = sorted((set(gm) & pl) - set(flags.values()))
                if not avail:
                    continue
                av = np.array(avail)
                d = np.linalg.norm(phi[gi][av] - phi[gi][flag], axis=1)
                d[~ok[gi][av]] = np.inf
                confs = {'qa': d, 'pairdev': pd_full[suite][gi][av]}
                for N in sizes:
                    confs[f'pd-cal@{N}'] = pd_cal_N[N][suite][gi][av]
                for meth, cv in confs.items():
                    j = int(np.argmin(cv))
                    r = int(av[j])
                    recs.append((pname, split, q, meth, float(cv[j]),
                                 devrow[r], srow[r], srow[flag]))
        df = pd.DataFrame(recs, columns=['pool', 'split', 'query',
                                         'method', 'conf', 'dev',
                                         'score', 's_flag'])

        def one_gate(ss, calB_N):
            cal = ss[(ss.split == 'cal') & ss['query'].isin(calB_N)]
            ev = ss[ss.split == 'eval']
            cc, ce = cal['conf'].to_numpy(), ev['conf'].to_numpy()
            cc, ce = (np.where(np.isfinite(x), x, 1e6) for x in (cc, ce))
            off = gate2(cc, cal['dev'].to_numpy(), ce, eps, alpha)
            dv = ev['dev'].to_numpy()
            return (float(off.mean()),
                    float((dv[off] > eps).mean()) if off.any() else 0.0)

        for N in sizes:
            calB_N = set(calB_all[:N - N // 2])
            for pool in ('le13b', 'all'):
                for meth in ('qa', 'pairdev', f'pd-cal@{N}'):
                    ss = df[(df.pool == pool) & (df.method == meth)]
                    v = one_gate(ss, calB_N)
                    out.append({'seed': seed, 'N': N, 'pool': pool,
                                'method': meth.split('@')[0],
                                'vol': v[0], 'viol': v[1]})
        print(f'seed {seed} done', flush=True)

    cur = pd.DataFrame(out)
    cur.to_parquet(os.path.join(RESULTS, 'cal_curve_nested.parquet'))
    agg = cur.groupby(['pool', 'method', 'N']).agg(
        vol=('vol', 'mean'), se=('vol', lambda x: x.std() / np.sqrt(len(x))),
        viol=('viol', 'mean')).reset_index()
    for pool in ('le13b', 'all'):
        print(f'\n== paired-sample curve, pool {pool} '
              f'(vol +- se, viol; {seeds} seeds, paired across N) ==')
        for meth in ('qa', 'pd-cal', 'pairdev'):
            sub = agg[(agg.pool == pool) & (agg.method == meth)] \
                .sort_values('N')
            cells = '  '.join(
                f'N={int(r.N)}: {r.vol:.2f}±{r.se:.02f} ({r.viol:.2f})'
                for r in sub.itertuples())
            print(f'  {meth:8s} {cells}')
    return cur


def predict_n(X, Qu, rows, n, names, seed=0, n_probe=500, m_anchor=1000,
              sigma_frac=0.25):
    """Order-of-magnitude predictor of the required paired sample:
    N* = s^2 / (kappa * tau^2), per suite. kappa = kernel coverage
    n_eff/N (query embeddings only); s^2 = within-neighborhood variance
    of same-query deviations (noise); tau^2 = across-query variance of
    the localized expected deviation (the gate's usable signal, debiased
    for estimation noise). Reference pair: flagship vs the suite's
    runner-up."""
    _, suite_of, gmean, ranked = _prep(X, Qu, rows, n, names)
    rng = np.random.default_rng(seed)
    out = {}
    for su in ('helm', 'eee'):
        flag = ranked[su][0]
        sub = rows[rows.suite == su]
        rf = sub[sub['model'] == flag].drop_duplicates('query')
        f_map = dict(zip(rf['query'], rf.index))
        q_map = dict(zip(rf['query'], rf['code']))
        # reference pair: best-ranked candidate with enough shared queries
        common, c_map = [], {}
        for cand in ranked[su][1:]:
            rc = sub[sub['model'] == cand].drop_duplicates('query')
            c_map = dict(zip(rc['query'], rc.index))
            common = sorted(set(f_map) & set(c_map))
            if len(common) >= 100:
                break
        if len(common) < 100:
            print(f'{su}: no candidate shares >=100 queries; skipped')
            continue
        d = np.linalg.norm(X[[f_map[q] for q in common]]
                           - X[[c_map[q] for q in common]], axis=1)
        U = Qu[[q_map[q] for q in common]].astype(np.float64)

        perm = rng.permutation(len(common))
        pi = perm[:min(n_probe, len(perm) // 2)]
        ai = perm[len(perm) // 2:len(perm) // 2 + m_anchor]
        ii = rng.integers(0, len(ai), 20000)
        kk = rng.integers(0, len(ai), 20000)
        kp = ii != kk
        med = float(np.median(np.linalg.norm(
            U[ai][ii[kp]] - U[ai][kk[kp]], axis=1)))
        D2 = ((U[pi][:, None, :] - U[ai][None, :, :]) ** 2).sum(-1)
        W = np.exp(-D2 / (2.0 * (sigma_frac * med) ** 2))
        sw = W.sum(axis=1)
        n_eff = sw ** 2 / np.maximum((W ** 2).sum(axis=1), 1e-30)
        kappa = float(n_eff.mean()) / len(ai)
        mu = (W @ d[ai]) / sw
        s2_loc = float((((W * (d[ai][None, :] - mu[:, None]) ** 2)
                         .sum(axis=1)) / sw).mean())
        tau2 = float(np.var(mu)) - s2_loc * float((1.0 / n_eff).mean())
        n_star = s2_loc / (kappa * max(tau2, 1e-12))
        out[su] = dict(kappa=kappa, n_eff_at_1000=float(n_eff.mean()),
                       s2=s2_loc, tau2=tau2, N_star=n_star)
        print(f'{su}: kappa {kappa:.3f} (n_eff {n_eff.mean():.0f} at '
              f'N=1000), s2 {s2_loc:.3f}, tau2 {tau2:.4f} '
              f'-> N* ~ {n_star:.0f}')
    return out


if __name__ == '__main__':
    main()
