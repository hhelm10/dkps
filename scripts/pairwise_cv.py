"""Leave-two-out pairwise ranking with the PAPER estimator (per HH: the
lite fixed-kernel run made qubric geometry look weak; this adds the same
per-draw pooled CV as the MAE evals).

Per pool context (both systems' LLM/family groups excluded) and probe
draw: candidates = sigma in {2,4,8} x {kNN-5, ridge(16,0.1) LOO-honest},
selected by pool reference error; blend alpha fit on pool with the
selected store. Overwrites figures/pairwise_cost.json (same schema; the
fig_cost renderer is unchanged).
"""
import json
import os
import re
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

MS = (1, 3, 5, 10, 20)
N_DRAW = 10
SIGS = (2, 4, 8)
ALPH = np.linspace(0, 1, 21)


def run(bench):
    if bench == 'swe':
        systems, qs, y, B, _ = load_panel(panel='data/judge/q100.json')
        labels = json.load(open('data/leaderboard/verified_labels.json'))
        tags = []
        for s in systems:
            m = re.search(r'^\s+model_display:\s*(.*)$',
                          labels[s].get('metadata_yaml', ''), re.M)
            tags.append(m.group(1).strip() if m else f'__solo__{s}')
        M, Q = B.shape
        X = np.load('data/judge/q100_emb_openai_small.npz')['X'] \
            .reshape(M, Q, -1)
        z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
        ids = [str(x) for x in z['ids']]
        V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in qs]]
    else:
        d = json.load(open('data/terminal_bench/tb2_panel.json'))
        DROP = {'Droid__GPT-5.3-Codex', 'Droid__Claude-Opus-4.6',
                'just-another-coding-agent__GLM-5'}
        all_sys = [s for s in d['systems'] if s not in DROP]
        tasks = d['tasks']
        Q = len(tasks)
        cov = {s: sum(os.path.exists(
            f'data/terminal_bench/tb2_txt/{s}/{t}.txt')
            for t in tasks) / Q for s in all_sys}
        systems = [s for s in all_sys if cov[s] >= 0.9]
        M = len(systems)
        sidx = [d['systems'].index(s) for s in systems]
        y = np.array(d['y'])[sidx]
        Bm = np.array([[v if v is not None else np.nan for v in row]
                       for row in d['B']])[sidx]
        chosen = json.load(open('data/terminal_bench/tb2_chosen.json'))
        B = np.zeros((M, Q))
        for i, s in enumerate(systems):
            for j, t in enumerate(tasks):
                c = chosen.get(s, {}).get(t)
                B[i, j] = (float(c['reward'] > 0.5)
                           if c and c.get('reward') is not None
                           else float(np.nan_to_num(Bm[i, j]) > 0.5))
        meta = d['meta']
        tags = ['|'.join(meta[s]['llm_tags']) or f'__solo__{s}'
                for s in systems]
        X = consensus_center(
            np.load('data/terminal_bench/tb2_emb_openai_small.npz')['X'],
            np.tile(np.arange(Q), len(all_sys))) \
            .reshape(len(all_sys), Q, -1)[
                [all_sys.index(s) for s in systems]]
        z = np.load('data/terminal_bench/tb2_query_vecs_64.npz',
                    allow_pickle=True)
        ids = [str(x) for x in z['ids']]
        V = np.asarray(z['vecs'], np.float32)[[ids.index(t) for t in tasks]]
        X = X.astype(np.float64)

    tag = np.array(tags)
    groups = sorted(set(tags))
    gidx = {g: np.where(tag == g)[0] for g in groups}
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)
    KQs = {s_: np.exp(-D2q / (2 * med / s_)).astype(np.float32)
           for s_ in SIGS}
    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(N_DRAW)]
             for m in MS}
    contexts = {}
    for ga, gb in combinations(groups, 2):
        contexts[frozenset((ga, gb))] = np.concatenate([gidx[ga], gidx[gb]])
    for g in groups:
        if len(gidx[g]) >= 2:
            contexts[frozenset((g,))] = gidx[g]
    print(bench, len(contexts), 'contexts', flush=True)

    ctx_pred = {}
    for ci, (excl, tgts) in enumerate(contexts.items()):
        pool = np.array([i for i in range(M) if tag[i] not in excl])
        P = len(pool)
        med_p = np.median(X[pool], axis=0, keepdims=True)
        Zc = X - med_p
        Zc = (Zc / np.maximum(np.linalg.norm(Zc, axis=-1, keepdims=True),
                              1e-9)).astype(np.float32)
        im = ItemModel(B[pool], y[pool], 10.0)
        Ws = {s_: np.einsum('qp,jpd->jqd', KQs[s_], Zc[pool],
                            optimize=True) for s_ in SIGS}
        out = {m: {} for m in MS}
        yp = y[pool]
        for m in MS:
            for cols in draws[m]:
                irt_t = np.array([im.predict(cols, B[t, cols])
                                  for t in tgts])
                irt_p = np.array([im.predict(cols, B[p, cols])
                                  for p in pool])
                cand = {}
                for s_ in SIGS:
                    KQ = KQs[s_]
                    Wp = Ws[s_][:, cols].reshape(P, -1)
                    Xi_t = Zc[tgts][:, cols].reshape(len(tgts), -1)
                    Xi_p = Zc[pool][:, cols].reshape(P, -1)
                    At = (Xi_t @ Wp.T) / KQ[cols].sum()
                    Ap = (Xi_p @ Wp.T) / KQ[cols].sum()
                    KQc = KQ[np.ix_(cols, cols)]
                    Wc = np.einsum('qp,jpd->jqd', KQc, Zc[pool][:, cols],
                                   optimize=True)
                    App = np.einsum('jqd,jqd->j', Zc[pool][:, cols], Wc) \
                        / KQc.sum()
                    Wct = np.einsum('qp,jpd->jqd', KQc, Zc[tgts][:, cols],
                                    optimize=True)
                    Att = np.einsum('jqd,jqd->j', Zc[tgts][:, cols], Wct) \
                        / KQc.sum()
                    Arr = np.einsum('jqd,jqd->j', Zc[pool],
                                    Ws[s_]) / KQ.sum()
                    Dt = np.sqrt(np.maximum(
                        Att[:, None] + Arr[None] - 2 * At, 0))
                    Dp = np.sqrt(np.maximum(
                        App[:, None] + Arr[None] - 2 * Ap, 0))
                    # kNN-5
                    Dp2 = Dp.copy()
                    np.fill_diagonal(Dp2, np.inf)
                    nn_p = np.argsort(Dp2, 1)[:, :5]
                    w = 1 / (np.take_along_axis(Dp2, nn_p, 1) + 1e-12)
                    gp = (w * yp[nn_p]).sum(1) / w.sum(1)
                    nn_t = np.argsort(Dt, 1)[:, :5]
                    wt = 1 / (np.take_along_axis(Dt, nn_t, 1) + 1e-12)
                    gt = (wt * yp[nn_t]).sum(1) / wt.sum(1)
                    cand[(s_, 'knn')] = (np.abs(gp - yp).mean(), gp, gt)
                    # ridge on pooled MDS coords (targets in MDS, not fit)
                    Dall = np.zeros((P + len(tgts), P + len(tgts)),
                                    np.float32)
                    Dall[:P, :P] = Dp
                    Dall[P:, :P] = Dt
                    Dall[:P, P:] = Dt.T
                    Dtt = np.sqrt(np.maximum(
                        Att[:, None] + Att[None]
                        - 2 * ((Xi_t @ np.einsum(
                            'qp,jpd->jqd', KQc, Zc[tgts][:, cols],
                            optimize=True).reshape(len(tgts), -1).T)
                            / KQc.sum()), 0))
                    Dall[P:, P:] = Dtt
                    n_ = P + len(tgts)
                    J = np.eye(n_) - 1 / n_
                    Bmm = -0.5 * J @ (Dall ** 2) @ J
                    ew, ev = np.linalg.eigh(Bmm)
                    order = np.argsort(ew)[::-1][:16]
                    Zm = ev[:, order] * np.sqrt(np.maximum(ew[order],
                                                           1e-12))
                    Zr, Zt = Zm[:P], Zm[P:]
                    G = Zr.T @ Zr + 0.1 * np.eye(16)
                    Gi = np.linalg.inv(G)
                    beta = Gi @ (Zr.T @ (yp - yp.mean()))
                    h = np.einsum('jr,rs,js->j', Zr, Gi, Zr)
                    pr = Zr @ beta + yp.mean()
                    loo = yp - (yp - pr) / np.maximum(1 - h, 1e-6)
                    gp_r = np.clip(loo, 0, 1)
                    gt_r = np.clip(Zt @ beta + yp.mean(), 0, 1)
                    cand[(s_, 'ridge')] = (np.abs(gp_r - yp).mean(),
                                           gp_r, gt_r)
                _, gp, gt = min(cand.values(), key=lambda v: v[0])
                a = ALPH[int(np.argmin([np.abs(al * irt_p + (1 - al) * gp
                                               - yp).mean()
                                        for al in ALPH]))]
                samp_t = B[np.ix_(tgts, cols)].mean(1)
                for k, v in (('sample', samp_t), ('irt', irt_t),
                             ('geom', gt), ('blend', a * irt_t
                                            + (1 - a) * gt)):
                    out[m].setdefault(k, np.zeros(len(tgts)))
                    out[m][k] = out[m][k] + v / len(draws[m])
        ctx_pred[excl] = (tgts, out)
        if ci % 200 == 0:
            print(bench, f'ctx {ci}', flush=True)

    res = {}
    for m in MS:
        num = {k: 0 for k in ('sample', 'irt', 'geom', 'blend')}
        den = dict.fromkeys(num, 0)
        num_all = dict(num)
        den_all = dict(num)
        for excl, (tgts, out) in ctx_pred.items():
            gs = list(excl)
            if len(gs) == 2:
                A_i = [i for i, t in enumerate(tgts) if tag[t] == gs[0]]
                B_i = [i for i, t in enumerate(tgts) if tag[t] == gs[1]]
                pairs = [(a, b) for a in A_i for b in B_i]
            else:
                pairs = list(combinations(range(len(tgts)), 2))
            for a, b in pairs:
                ta, tb = tgts[a], tgts[b]
                gap = abs(y[ta] - y[tb])
                td = np.sign(y[ta] - y[tb])
                if td == 0:
                    continue
                for k in num:
                    ok = int(np.sign(out[m][k][a] - out[m][k][b]) == td)
                    num_all[k] += ok
                    den_all[k] += 1
                    if gap >= 0.05:
                        num[k] += ok
                        den[k] += 1
        res[m] = {'gap05': {k: round(num[k] / den[k], 4) for k in num},
                  'all': {k: round(num_all[k] / den_all[k], 4)
                          for k in num}}
        print(bench, m, res[m]['gap05'], flush=True)
    return res


if __name__ == '__main__':
    out = {b: run(b) for b in ('swe', 'tb2')}
    json.dump(out, open('figures/pairwise_cost.json', 'w'), indent=1)
    print('wrote figures/pairwise_cost.json')
