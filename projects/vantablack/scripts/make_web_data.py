"""Data for the interactive Moltbook page (web/moltbook_monitor.html).

The page plays the Moltbook window (Jan 30 - Feb 9, 2026; the paper's 8 activity-time steps) as an
activity monitor and a perspective monitor, and lets the viewer stop collection at the end of any step
k >= 3. For every k, the method was run on the data up to the end of step k, on the 2,574 agents of the
paper's run (real_tstep.py --T 8 --stop-step k --cohort-tag 01-30_02-09_T8_ms6); the page replays that run's own intermediate results. Nothing on the page is
simulated or recomputed in the browser.

Exported (no content of posts or comments):
  activity     actions per hour of the 2,574 agents analyzed (the reference run's cohort)
  monitor      positions of all 2,574 agents (named only if detected at some stopping point), centered
               per step and projected on the two leading directions, for the perspective monitor
  stops[k]     per stopping point: cohort size, d, every peeling round (p, observed scan statistic
               and its 199 permutation null in the round's size band, membership scores of the
               members, a histogram of the other tested agents' scores, and the two-groups fit), and per detected
               subnetwork: members with scores, per-increment set scores, trajectories (as Figure 4b),
               and the summary numbers used in its description
  heat[k]      pair scores among the members of the subnetworks and 30 other agents: at every
               increment (rank normal scores) and across increments (majority statistic)

    python scripts/make_web_data.py            # writes web/moltbook_data.js
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import tstep  # noqa: E402
from pipeline.ingest_moltnet import ORIGIN  # noqa: E402
from pipeline.paths import DERIVED  # noqa: E402

D_DIR = os.path.join(DERIVED, 'real_tstep')
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, 'web', 'moltbook_data.js')
START, END = '2026-01-30', '2026-02-09'
STOPS = range(3, 9)
REF_TAG = '01-30_02-09_T8_ms6'                 # the paper's run: 2,574 agents
rng = np.random.default_rng(0)
r2 = lambda x, n=2: None if x is None or not np.isfinite(x) else round(float(x), n)


def stem(names, min_len=4):
    """The prefix (case-insensitive, at least min_len characters) shared by at least 40% of the
    handles (and at least two) that maximizes length x number of handles sharing it, or None."""
    best = None
    low = [n.lower() for n in names]
    for n in low:
        for L in range(len(n), min_len - 1, -1):
            p = n[:L]
            c = sum(x.startswith(p) for x in low)
            if c >= max(2, int(np.ceil(0.4 * len(low)))):
                if best is None or L * c > len(best[0]) * best[1]:                # longest x most shared
                    best = (p, c)
    if best is None:
        return None
    orig = next(n for n in names if n.lower().startswith(best[0]))[:len(best[0])]
    return dict(stem=orig, count=best[1])


def load(k):
    """The run that stopped after step k, on the reference run's agents (for k = 8, the reference
    cohort itself; for k < 8, those of its agents active in at least two of the k steps)."""
    pat = f'01-30_02-09_T{k}*_stop{k}of8.pkl' if k == 8 else f'01-30_02-09_T{k}*_stop{k}of8_coh-{REF_TAG}.pkl'
    f = glob.glob(os.path.join(D_DIR, pat))
    if not f:
        return None, None
    tag = os.path.basename(f[0])[:-4]
    return pickle.load(open(f[0], 'rb')), tag


def centered(Z, obs):
    Zc = Z - np.nanmean(np.where(obs[:, :, None], Z, np.nan), 0, keepdims=True)
    return np.where(obs[:, :, None], Zc, np.nan)


def export_stop(k):
    run, tag = load(k)
    if run is None:
        return None, None
    res, summ = run['res'], run['summary']
    names = np.array(run['cohort'])
    N, T = len(names), summ['T']
    rows = pd.read_parquet(os.path.join(D_DIR, f'{tag}_rows.parquet'), columns=['agent_ix', 'step'])
    obs = np.bincount(rows.agent_ix.values * T + rows.step.values, minlength=N * T).reshape(N, T) > 0
    D = res['D']
    Zm = centered(res['positions'], obs)
    first = np.array([np.flatnonzero(obs[i])[0] for i in range(N)])
    last = np.array([np.flatnonzero(obs[i])[-1] for i in range(N)])
    netall = np.nan_to_num(Zm[np.arange(N), last] - Zm[np.arange(N), first])
    alpha = 0.05

    rounds, subs = [], []
    keep = np.arange(N)
    grp = np.zeros(N, int)
    for g_i, g in enumerate(res['groups']):
        Qr = tstep.step_scores(D[keep])
        b = int(g['band']) if g['band'] is not None else int(np.nanargmax(g['obs']))
        null = g['nulls'][:, b]
        null = null[np.isfinite(null)]
        mem = np.asarray(g['members'])
        loc = np.searchsorted(keep, mem)
        sc = g['score_all']
        isin = np.zeros(len(keep), bool); isin[loc] = True
        sig = bool(g['p'] <= alpha and g['subnetwork'])
        rd = dict(round=g_i + 1, p=r2(g['p'], 3), n_tested=int(g['n_tested']), band=b,
                  band_label=['2', '3-10', '11-30', '31-100', '101+'][b],
                  obs=r2(g['obs'][b]), null=[r2(x) for x in np.sort(rng.permutation(null))],
                  n_exceed=int((null >= g['obs'][b]).sum()), significant=sig, size=int(len(mem)))
        if sc is not None and len(mem):
            other = np.asarray(sc)[~isin]
            other = other[np.isfinite(other)]
            fin = np.asarray(sc)[np.isfinite(sc)]                         # -inf: fewer than two increments
            lo, hi = float(min(other.min(), fin.min())), float(fin.max())
            edges = np.linspace(lo, hi, 41)
            post, fit = tstep.two_groups(np.asarray(sc), params=True)
            rd.update(hist=np.histogram(other, edges)[0].tolist(), hist_lo=r2(lo), hist_hi=r2(hi),
                      member_scores=[r2(x) for x in np.asarray(sc)[loc]],
                      max_other=r2(other.max()), n_finite=int(np.isfinite(sc).sum()),
                      fit=None if fit is None else {a: r2(v, 4) for a, v in fit.items()},
                      post_members=int((post[loc] > 0.5).sum()), post_others=int((np.delete(post, loc) > 0.5).sum()))
        rounds.append(rd)
        if sig:
            s_i = len(subs)
            grp[mem] = s_i + 1
            order = np.argsort(-np.asarray(sc)[loc])
            Zs = tstep._set_steps(Qr[:, loc][:, :, loc][:, None])[:, 0]       # per-increment set score
            zS = float(tstep.majority_z(Zs[:, None])[0][0])
            # pair statistics: within the subnetwork vs all pairs tested in the round
            Zp = tstep.pair_scores(Qr)
            iu = np.triu_indices(len(keep), 1)
            allp = Zp[iu]; allp = allp[np.isfinite(allp)]
            wi = Zp[np.ix_(loc, loc)][np.triu_indices(len(loc), 1)]
            wi = wi[np.isfinite(wi)]
            # trajectories, as Figure 4b
            u = netall[mem].mean(0); e1 = u / np.linalg.norm(u)
            W = netall - (netall @ e1)[:, None] * e1
            e2 = np.linalg.svd(W - W.mean(0), full_matrices=False)[2][0]
            Y = Zm @ np.stack([e1, e2], 1)
            bg = rng.choice(np.setdiff1d(np.arange(N), mem), 40, replace=False)
            tr = lambda i: [None if not np.isfinite(Y[i, t, 0]) else [r2(Y[i, t, 0] - Y[i, first[i], 0], 3),
                                                                    r2(Y[i, t, 1] - Y[i, first[i], 1], 3)]
                            for t in range(T)]
            nm = names[mem][order].tolist()
            subs.append(dict(
                id=s_i + 1, round=g_i + 1, p=r2(g['p'], 3), size=int(len(mem)),
                members=[dict(name=n, score=r2(s, 1)) for n, s in zip(nm, np.asarray(sc)[loc][order])],
                step_scores=[r2(x) for x in Zs], set_z=r2(zS), n_obs=int(np.isfinite(Zs).sum()),
                n_above=int((Zs > 2).sum()), top_steps=[int(t) + 1 for t in np.argsort(-np.nan_to_num(Zs, nan=-9))[:2]],
                pair_within=r2(np.median(wi)), pair_q99=r2(np.quantile(allp, 0.99)),
                pair_share_top1=r2(np.mean(wi > np.quantile(allp, 0.99)), 2),
                n_exceed=rd['n_exceed'], n_null=len(null), obs=rd['obs'],
                score_min=r2(np.asarray(sc)[loc].min(), 1), score_max=r2(np.asarray(sc)[loc].max(), 1),
                max_other=rd.get('max_other'),
                n_other_above=int((np.delete(np.asarray(sc), loc) >= np.asarray(sc)[loc].min()).sum()),
                stem=stem(nm),
                traj=dict(members=[tr(i) for i in mem[order]], others=[tr(i) for i in bg])))
        if g['p'] > alpha or len(mem) == 0:
            break
        keep = np.delete(keep, loc)

    # heat map: members of the subnetworks + 30 others, full-cohort ranks (as Figure 4c)
    mem_all = np.concatenate([np.sort(np.flatnonzero(grp == s['id'])) for s in subs]) if subs else np.array([], int)
    oth = rng.choice(np.flatnonzero(grp == 0), 30, replace=False)
    ordr = np.r_[mem_all, oth]
    Q = tstep.step_scores(D)
    H = tstep.pair_scores(Q[:, ordr][:, :, ordr]).astype(float)
    q1 = lambda M_: [[None if not np.isfinite(v) else round(float(v), 1) for v in row] for row in M_]
    heat = dict(n=len(ordr), sizes=[s['size'] for s in subs], z=q1(H),
                steps=[q1(Q[t][np.ix_(ordr, ordr)]) for t in range(Q.shape[0])])   # per-increment pair scores
    stop = dict(k=k, T=T, N=int(N), actions=int(summ.get('cohort_actions') or len(rows)), d=int(res['d']),
                n_factors=int(res['n_factors']), B=int(summ['B']),
                step_bounds=summ['step_bounds'], rounds=rounds, subnetworks=subs, heat=heat)
    return stop, (names, grp, Zm, obs)


def main():
    """Activity of the reference run's agents only (the agents the page analyzes); the time steps are
    the platform-wide activity-time steps of the window."""
    run8, tag8 = load(8)
    rows = pd.read_parquet(os.path.join(D_DIR, f'{tag8}_rows.parquet'), columns=['agent', 't'])
    t = rows.t.values
    h0 = (pd.Timestamp(START, tz='UTC') - ORIGIN).total_seconds() / 3600
    h1 = (pd.Timestamp(END, tz='UTC') - ORIGIN).total_seconds() / 3600
    hours = np.floor(t - h0).astype(int)
    nh = int(np.ceil(h1 - h0))
    per_hour = np.bincount(hours[(hours >= 0) & (hours < nh)], minlength=nh)
    first_h = pd.Series(hours).groupby(rows.agent.values).min()
    new_agents = np.bincount(first_h.values.clip(0, nh - 1), minlength=nh)
    bounds_h = [(pd.Timestamp(b) - pd.Timestamp(START, tz='UTC')).total_seconds() / 3600 for b in run8['summary']['step_bounds']]
    data = dict(start=START, end=END, hours=nh, per_hour=per_hour.tolist(), new_agents=new_agents.tolist(),
                step_bounds_h=[round(b, 3) for b in bounds_h], n_actions=int(len(rows)),
                n_agents=int(rows.agent.nunique()), stops={})
    del rows, run8

    full = None
    for k in STOPS:
        s, extra = export_stop(k)
        if s is None:
            print(f'stop {k}: no run yet')
            continue
        data['stops'][k] = s
        print(f'stop {k}: N={s["N"]}, d={s["d"]}, rounds={len(s["rounds"])}, subnetworks={len(s["subnetworks"])}')
        if k == 8:
            full = extra
    if full is not None:                                                   # perspective monitor (k = 8 positions)
        names, grp, Zm, obs = full
        N, T = obs.shape
        found = {m['name'] for st in data['stops'].values() for sb in st['subnetworks'] for m in sb['members']}
        isf = np.isin(names, list(found))
        pick = np.arange(N)                                                # every agent analyzed
        X = Zm[pick].reshape(-1, Zm.shape[2])
        ok = np.isfinite(X[:, 0])
        V = np.linalg.svd(X[ok] - X[ok].mean(0), full_matrices=False)[2][:2]
        P = (Zm[pick] - X[ok].mean(0)) @ V.T
        P = P / np.nanstd(P)
        data['monitor'] = dict(name=[str(names[i]) if isf[i] else None for i in pick],     # names of detected members only
                               xy=[[None if not np.isfinite(P[i, t, 0]) else [round(float(P[i, t, 0]), 3), round(float(P[i, t, 1]), 3)]
                                    for t in range(T)] for i in range(len(pick))])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        fh.write('window.MOLTBOOK = ' + json.dumps(data, separators=(',', ':')) + ';\n')
    print('wrote', OUT, f'{os.path.getsize(OUT) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
