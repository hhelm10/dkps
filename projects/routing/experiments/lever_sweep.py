"""Lever sweep for tolerance-gated offloading on the combined pool.

One decision-collection pass records, per (pool cap, hold query), the pick,
confidence, realized behavioral deviation, and realized score of every
estimator; every lever is then a slice of that table:

    contract (eps, alpha)   post-hoc conformal gate grid
    pool cap                le13b / le40b / all (same geometry pass)
    estimator               qa-dist @ sigma | score-gap | oracle-conf |
                            oracle-pick | constant-conf (static/lead/random)
    bandwidth sigma         one localized geometry per sigma; static = inf
    anchor cache density    per-model row subsample (separate passes)
    calibration size        post-hoc subsample of the calibration split
    traffic heterogeneity   per-suite stratification; per-suite calibration

oracle-conf gates qa's own pick with its realized deviation (the ceiling of
any confidence estimator); oracle-pick additionally picks the hindsight-best
candidate (the frontier ceiling). score-gap tests whether a score signal
predicts behavioral deviation at all.

Run from repo root: pixi run python -m projects.routing.experiments.lever_sweep
"""

import os

import numpy as np
import pandas as pd

from .run_cost_routing import parse_params
from .run_helm import RESULTS

SIGMAS = (0.125, 0.25, 0.5, 1.0)
POOLS = (('le13b', 13.0), ('le40b', 40.0), ('all', None))
CONTRACTS = ((0.3, 0.10), (0.5, 0.10), (0.5, 0.20))


def collect(X, Qu, rows, n, names, seed, anchor_frac=1.0, sigmas=SIGMAS,
            n_hold=800, pools=POOLS):
    from .run_eee import batched_localized_stats
    sizes = [parse_params(nm.split(':', 1)[1]) for nm in names]
    suite_of_model = [nm.split(':', 1)[0] for nm in names]
    rng = np.random.default_rng(seed)
    gmean = rows.groupby('model')['score'].mean()
    flags = {su: int(gmean.loc[[i for i in range(n)
                                if suite_of_model[i] == su]].idxmax())
             for su in ('helm', 'eee')}
    flag_set = set(flags.values())
    pool_sets = {}
    for pname, cap in pools:
        if cap is None:
            pool_sets[pname] = set(range(n)) - flag_set
        else:
            pool_sets[pname] = {i for i in range(n) if i not in flag_set
                                and sizes[i] is not None and sizes[i] <= cap}

    # shared hold set: eligibility under the smallest pool so every pool cap
    # sees the same traffic (le13b is a subset of the others)
    resp = rows.groupby('query')['model'].agg(set)
    suite_q = rows.drop_duplicates('query').set_index('query')['suite']
    elig = [q for q, ms in resp.items()
            if flags[suite_q[q]] in ms and len(ms & pool_sets['le13b']) >= 1]
    k_hold = min(n_hold, len(elig))
    take = rng.choice(len(elig), size=k_hold, replace=False)
    eval_q = {elig[i] for i in take[:k_hold // 2]}
    cal_q = {elig[i] for i in take[k_hold // 2:]}
    hold_q = eval_q | cal_q
    is_hold = rows['query'].isin(hold_q).to_numpy()
    anchor_m = ~is_hold
    assert not rows['query'][anchor_m].isin(hold_q).any()

    if anchor_frac < 1.0:
        keep = np.zeros(len(rows), dtype=bool)
        a_idx = np.flatnonzero(anchor_m)
        model_arr = rows['model'].to_numpy()
        for m in range(n):
            idx_m = a_idx[model_arr[a_idx] == m]
            k_m = max(1, int(round(anchor_frac * len(idx_m)))) \
                if len(idx_m) else 0
            if k_m:
                keep[rng.choice(idx_m, size=k_m, replace=False)] = True
        anchor_m = keep

    Xa = X[anchor_m]
    sa = rows['score'].to_numpy()[anchor_m]
    model_a = rows['model'].to_numpy()[anchor_m]
    ua = Qu[rows['code'].to_numpy()[anchor_m]]
    model_groups = [np.flatnonzero(model_a == m) for m in range(n)]
    i = rng.integers(0, len(ua), 20000)
    k = rng.integers(0, len(ua), 20000)
    kp = i != k
    med = float(np.median(np.linalg.norm(ua[i[kp]] - ua[k[kp]], axis=1)))

    hold_groups = list(rows[is_hold].groupby('query'))
    Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in hold_groups])
    D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
    D2 = D2 - D2.min(axis=1, keepdims=True)
    geoms = {}
    for f in sigmas:
        W = np.exp(-D2 / (2.0 * (f * med) ** 2)).astype(np.float32)
        geoms[f'qa-{f:g}x'] = batched_localized_stats(
            Xa, sa, model_groups, n, W)
    st = batched_localized_stats(Xa, sa, model_groups, n,
                                 np.ones((1, len(Xa)), np.float32))
    gmean_a = pd.Series(sa).groupby(pd.Series(model_a)).mean()
    ref_key = f'qa-{(0.25 if 0.25 in sigmas else sigmas[0]):g}x'
    ref = geoms[ref_key]                       # reference estimator

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
        split = 'eval' if q in eval_q else 'cal'
        for pname in pool_sets:
            avail = sorted(set(gm) & pool_sets[pname])
            if not avail:
                continue
            picks, r_qa = [], None
            for est, (phi, shat, okm) in geoms.items():
                d = np.linalg.norm(phi[gi][avail] - phi[gi][flag], axis=1)
                d[~okm[gi][avail]] = np.inf
                r = avail[int(np.argmin(d))]
                picks.append((est, r, float(np.min(d))))
                if est == ref_key:
                    r_qa = r
            # score-gap: does a score signal predict behavioral deviation?
            phi_r, shat_r, okm_r = ref
            sv = shat_r[gi][avail].copy()
            sv[~okm_r[gi][avail]] = -np.inf
            if np.isfinite(sv).any():
                r_sg = avail[int(np.argmax(sv))]
                picks.append(('score-gap', r_sg,
                              float(shat_r[gi][flag] - sv.max())))
            picks.append(('oracle-conf', r_qa, devrow[r_qa]))
            r_op = min(avail, key=lambda a: devrow[a])
            picks.append(('oracle-pick', r_op, devrow[r_op]))
            d_st = np.linalg.norm(st[0][0][avail] - st[0][0][flag], axis=1)
            d_st[~st[2][0][avail]] = np.inf
            picks.append(('static', avail[int(np.argmin(d_st))],
                          float(rng.random())))
            picks.append(('lead', avail[int(np.argmax(
                [float(gmean_a.get(a, -np.inf)) for a in avail]))],
                float(rng.random())))
            picks.append(('random', avail[rng.integers(0, len(avail))],
                          float(rng.random())))
            for est, r, conf in picks:
                out.append((pname, seed, anchor_frac, split, suite, q, est,
                            conf, devrow[r], srow[r], srow[flag]))
    return pd.DataFrame(out, columns=[
        'pool', 'seed', 'anchor_frac', 'split', 'suite', 'query', 'method',
        'conf', 'dev', 'score', 's_flag'])


def gate(cal, ev, eps, alpha, min_prefix=20, by_suite=False):
    """Conformal gate: largest confidence cutoff with calibration violation
    <= alpha; returns (volume, achieved violation, retention) on eval."""
    if by_suite:
        parts = [gate(cal[cal.suite == s], ev[ev.suite == s], eps, alpha,
                      min_prefix) + (len(ev[ev.suite == s]),)
                 for s in ('helm', 'eee')]
        tot = sum(p[3] for p in parts)
        return tuple(sum(p[j] * p[3] for p in parts) / tot for j in range(3))
    c = cal.sort_values('conf')
    bad = (c['dev'].to_numpy() > eps).astype(float)
    cum = np.cumsum(bad) / np.arange(1, len(bad) + 1)
    ok = np.flatnonzero(cum <= alpha)
    ok = ok[ok >= min_prefix - 1]
    t = c['conf'].to_numpy()[ok.max()] if len(ok) else -np.inf
    off = ev['conf'].to_numpy() <= t
    dv, sc, fl = (ev[k].to_numpy() for k in ('dev', 'score', 's_flag'))
    return (float(off.mean()),
            float((dv[off] > eps).mean()) if off.any() else 0.0,
            float(np.where(off, sc, fl).mean() / fl.mean()))


def analyze(df, contracts=CONTRACTS, n_cal=None):
    """Per (pool, anchor_frac, method): confidence quality and certified
    volume at each contract (constant-conf methods report their fixed
    population violation instead)."""
    from scipy.stats import spearmanr
    recs = []
    gated = [m for m in df['method'].unique()
             if m not in ('static', 'lead', 'random')]
    for (pool, af, meth), sub in df.groupby(['pool', 'anchor_frac',
                                             'method']):
        rec = {'pool': pool, 'anchor_frac': af, 'method': meth}
        ev_all = sub[sub.split == 'eval']
        rec['spearman'] = float(spearmanr(ev_all['conf'],
                                          ev_all['dev']).statistic) \
            if meth in gated else np.nan
        for eps, alpha in contracts:
            key = f'({eps:g},{alpha:.0%})'
            if meth in gated:
                per_seed = []
                for seed, ss in sub.groupby('seed'):
                    cal = ss[ss.split == 'cal']
                    if n_cal is not None and len(cal) > n_cal:
                        cal = cal.sample(n_cal, random_state=int(seed))
                    per_seed.append(gate(cal, ss[ss.split == 'eval'],
                                         eps, alpha))
                v = np.mean(per_seed, axis=0)
                rec[f'vol{key}'] = v[0]
                rec[f'viol{key}'] = v[1]
                rec[f'ret{key}'] = v[2]
            else:
                rec[f'viol{key}'] = float((ev_all['dev'] > eps).mean())
        recs.append(rec)
    return pd.DataFrame(recs)


def main():
    from .run_combined import load_combined
    os.makedirs(RESULTS, exist_ok=True)
    X, Qu, rows, n, names = load_combined()
    print(f'combined: {len(rows)} rows, {n} models')

    parts = [collect(X, Qu, rows, n, names, seed) for seed in range(5)]
    for af in (0.05, 0.1, 0.25, 0.5):
        parts += [collect(X, Qu, rows, n, names, seed, anchor_frac=af,
                          sigmas=(0.25,)) for seed in range(3)]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(os.path.join(RESULTS, 'lever_decisions.parquet'))

    summary = analyze(df)
    summary.to_csv(os.path.join(RESULTS, 'lever_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
