"""Leave-two-out pairwise ranking, matched to the paper MAE pipeline
(per HH 2026-09-14): consensus centering (global, as in fig_protocols /
tb2_eval), full paper sigma grid, ridge-on-MDS only (no kNN, per HH),
three ridge configs, per-draw pooled CV from LOO-honest pool errors,
alpha on the 101-point grid.

Fast path: with global centering the M x M PKPS distance matrix per
(sigma, m, draw) is context-independent -> precomputed once in the
parent and shared with fork workers. Per context the work is just a
2PL fit + submatrix MDS/ridge per candidate. Overwrites
figures/pairwise_cost.json (same schema; fig_cost.py unchanged).
"""
import json
import os
import re
import sys
from itertools import combinations
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

MS = (1, 3, 5, 10, 20)
N_DRAW = 10
SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
RIDGE = ((16, 0.1), (16, 1.0), (8, 0.1))
ALPH = np.linspace(0, 1, 101)
WORKERS = 20

G = {}  # read-only worker state, inherited via fork


def _ctx_work(item):
    excl, tgts = item
    M, B, y, tag = G['M'], G['B'], G['y'], G['tag']
    D_of, draws = G['D_of'], G['draws']
    pool = np.array([i for i in range(M) if tag[i] not in excl])
    P = len(pool)
    sub = np.concatenate([pool, tgts])
    yp = y[pool]
    im = ItemModel(B[pool], yp, 10.0)
    out = {m: {} for m in MS}
    for m in MS:
        for di, cols in enumerate(draws[m]):
            irt_t = np.array([im.predict(cols, B[t, cols])
                              for t in tgts])
            irt_p = np.array([im.predict(cols, B[p, cols])
                              for p in pool])
            cand = {}
            for s_ in SIGS:
                Ds = D_of[(s_, m, di)][np.ix_(sub, sub)]
                n_ = len(sub)
                J = np.eye(n_, dtype=np.float32) - np.float32(1 / n_)
                Bmm = -0.5 * J @ (Ds ** 2) @ J
                ew, ev = np.linalg.eigh(Bmm)
                order = np.argsort(ew)[::-1]
                for r_dim, alpha in RIDGE:
                    o = order[:r_dim]
                    Z = ev[:, o] * np.sqrt(np.maximum(ew[o], 1e-12))
                    Zr, Zt = Z[:P], Z[P:]
                    Gm = Zr.T @ Zr + alpha * np.eye(r_dim,
                                                    dtype=np.float32)
                    Gi = np.linalg.inv(Gm)
                    beta = Gi @ (Zr.T @ (yp - yp.mean()))
                    h = np.einsum('jr,rs,js->j', Zr, Gi, Zr)
                    pr = Zr @ beta + yp.mean()
                    loo = yp - (yp - pr) / np.maximum(1 - h, 1e-6)
                    gp = np.clip(loo, 0, 1)
                    gt = np.clip(Zt @ beta + yp.mean(), 0, 1)
                    cand[(s_, r_dim, alpha)] = (np.abs(gp - yp).mean(),
                                                gp, gt)
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
    return excl, tgts, out


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
        X = consensus_center(
            np.load('data/judge/q100_emb_openai_small.npz')['X'],
            np.tile(np.arange(Q), M)) \
            .reshape(M, Q, -1).astype(np.float32)
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
                [all_sys.index(s) for s in systems]].astype(np.float32)
        z = np.load('data/terminal_bench/tb2_query_vecs_64.npz',
                    allow_pickle=True)
        ids = [str(x) for x in z['ids']]
        V = np.asarray(z['vecs'], np.float32)[[ids.index(t) for t in tasks]]

    tag = np.array(tags)
    groups = sorted(set(tags))
    gidx = {g: np.where(tag == g)[0] for g in groups}
    M = len(tag)
    Q = B.shape[1]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)
    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(N_DRAW)]
             for m in MS}

    # global-centering fast path: PKPS distances are context-independent
    D_of = {}
    for s_ in SIGS:
        KQ = np.exp(-D2q / (2 * med / s_)).astype(np.float32)
        W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
        Arr = np.einsum('jqd,jqd->j', X, W) / KQ.sum()
        for m in MS:
            for di, cols in enumerate(draws[m]):
                Xi = X[:, cols].reshape(M, -1)
                A_tr = (Xi @ W[:, cols].reshape(M, -1).T) / KQ[cols].sum()
                KQc = KQ[np.ix_(cols, cols)]
                Wc = np.einsum('qp,jpd->jqd', KQc, X[:, cols],
                               optimize=True)
                Att = np.einsum('jqd,jqd->j', X[:, cols], Wc) / KQc.sum()
                D_of[(s_, m, di)] = np.sqrt(np.maximum(
                    Att[:, None] + Arr[None] - 2 * A_tr, 0)) \
                    .astype(np.float32)
    print(bench, 'distance matrices ready', flush=True)

    contexts = {}
    for ga, gb in combinations(groups, 2):
        contexts[frozenset((ga, gb))] = np.concatenate([gidx[ga], gidx[gb]])
    for g in groups:
        if len(gidx[g]) >= 2:
            contexts[frozenset((g,))] = gidx[g]
    print(bench, len(contexts), 'contexts', flush=True)

    G.update(M=M, B=B, y=y, tag=tag, D_of=D_of, draws=draws)
    items = list(contexts.items())
    ctx_pred = {}
    with get_context('fork').Pool(WORKERS) as pool_:
        for ci, (excl, tgts, out) in enumerate(
                pool_.imap_unordered(_ctx_work, items, chunksize=4)):
            ctx_pred[excl] = (tgts, out)
            if ci % 100 == 0:
                print(bench, f'ctx {ci}/{len(items)}', flush=True)

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
