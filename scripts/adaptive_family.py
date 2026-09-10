"""Adaptive-probe results under the publication config (family-out +
per-draw CV over kNN/ridge, LOO-honest) for the hero table.

Per target: 2PL fit on family-out references, simulated-CAT item path,
per-target panels through the same geom CV as the random evals (qubric
and raw), pooled alpha blend. Deterministic (no draws); bootstrap CIs
over systems; paired blend-IRT deltas.

Usage: python scripts/adaptive_family.py swe|tb2
Writes figures/{q100,tb2}_adaptive_family.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
ALPHAS = np.linspace(0, 1, 101)
MS = (1, 3, 5, 10, 20)
DROP = {'Droid__GPT-5.3-Codex', 'Droid__Claude-Opus-4.6',
        'just-another-coding-agent__GLM-5'}


def load_swe():
    from pillars import vendor_tag
    systems, q100, y, B, _ = load_panel(panel='data/judge/q100.json')
    M, Q = B.shape
    labels = json.load(open('data/leaderboard/verified_labels.json'))
    fam = [vendor_tag(labels, s) for s in systems]
    allowed = np.array([[j != i and not (fam[i] and fam[j] == fam[i])
                         for j in range(M)] for i in range(M)])
    Xq = consensus_center(
        np.load('data/judge/q100_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), M)).reshape(M, Q, -1).astype(np.float32)
    HT = np.load('data/judge/q100_raw_emb_openai_small.npz')['HT'] \
        .reshape(M, Q, -1)
    Xr = HT - np.median(HT, axis=0, keepdims=True)
    Xr /= np.maximum(np.linalg.norm(Xr, axis=-1, keepdims=True), 1e-9)
    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]
    ks = (3, 5)
    return y, B, allowed, Xq, Xr.astype(np.float32), V, ks, \
        'figures/q100_adaptive_family.json'


def load_tb2():
    d = json.load(open('data/terminal_bench/tb2_panel.json'))
    all_sys = [s for s in d['systems'] if s not in DROP]
    tasks = d['tasks']
    Q = len(tasks)
    cov = {s: sum(os.path.exists(
        f'data/terminal_bench/tb2_txt/{s}/{t}.txt') for t in tasks) / Q
        for s in all_sys}
    systems = [s for s in all_sys if cov[s] >= 0.9]
    M = len(systems)
    sidx = [d['systems'].index(s) for s in systems]
    y = np.array(d['y'])[sidx]
    Bmean = np.array([[v if v is not None else np.nan for v in row]
                      for row in d['B']])[sidx]
    chosen = json.load(open('data/terminal_bench/tb2_chosen.json'))
    B = np.zeros((M, Q))
    for i, s in enumerate(systems):
        for j, t in enumerate(tasks):
            c = chosen.get(s, {}).get(t)
            if c is not None and c.get('reward') is not None:
                B[i, j] = float(c['reward'] > 0.5)
            else:
                B[i, j] = float(np.nan_to_num(Bmean[i, j]) > 0.5)
    meta = d['meta']
    fams = [set(meta[s]['llm_tags']) for s in systems]
    allowed = np.array([[j != i and not (fams[i] and fams[i] & fams[j])
                         for j in range(M)] for i in range(M)])
    Xq = consensus_center(
        np.load('data/terminal_bench/tb2_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), len(all_sys))) \
        .reshape(len(all_sys), Q, -1)[
            [all_sys.index(s) for s in systems]].astype(np.float32)
    HT = np.load('data/terminal_bench/tb2_raw_emb_openai_small.npz')['HT'] \
        .reshape(len(all_sys), Q, -1)[[all_sys.index(s) for s in systems]]
    Xr = HT - np.median(HT, axis=0, keepdims=True)
    Xr /= np.maximum(np.linalg.norm(Xr, axis=-1, keepdims=True), 1e-9)
    z = np.load('data/terminal_bench/tb2_query_vecs_64.npz',
                allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(t) for t in tasks]]
    ks = (3, 5, 7, 10, 15)
    return y, B, allowed, Xq, Xr.astype(np.float32), V, ks, \
        'figures/tb2_adaptive_family.json'


def main(bench):
    y, B, allowed, Xq, Xr, V, KS, out_path = \
        load_swe() if bench == 'swe' else load_tb2()
    M, Q = B.shape
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)
    kerns = {}
    for tag, X in (('qubric', Xq), ('raw', Xr)):
        for s_ in SIGS:
            KQ = np.exp(-D2q / (2 * med / s_))
            W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
            kerns[(tag, s_)] = (KQ, W,
                                np.einsum('jqd,jqd->j', X, W) / KQ.sum())
    Xof = {'qubric': Xq, 'raw': Xr}

    def pkps_D(tag, cols, s_):
        X = Xof[tag]
        KQ, W, Arr = kerns[(tag, s_)]
        Xi = X[:, cols].reshape(M, -1)
        A_tr = (Xi @ W[:, cols].reshape(M, -1).T) / KQ[cols].sum()
        KQc = KQ[np.ix_(cols, cols)]
        Wc = np.einsum('qp,jpd->jqd', KQc, X[:, cols], optimize=True)
        Att = np.einsum('jqd,jqd->j', X[:, cols], Wc) / KQc.sum()
        return np.sqrt(np.maximum(Att[:, None] + Arr[None] - 2 * A_tr, 0))

    print('fitting per-target 2PL + CAT paths...')
    models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0)
              for i in range(M)]
    adaptive = [models[i].adaptive_path(B[i]) for i in range(M)]

    rng_b = np.random.default_rng(1)

    def ci(e):
        v = np.array([e[rng_b.integers(0, M, M)].mean()
                      for _ in range(2000)])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def geom_store(tag, cols_of):
        cand = {}
        for s_ in SIGS:
            Dc = {c: pkps_D(tag, np.asarray(c), s_)
                  for c in {tuple(cols_of[i]) for i in range(M)}}
            for k in KS:
                errs, store = [], {}
                for i in range(M):
                    D = Dc[tuple(cols_of[i])]
                    refs = np.where(allowed[i])[0]
                    Dref = D[:, refs].copy()
                    for r, j in enumerate(refs):
                        Dref[j, r] = np.inf
                    Dref[i] = D[i, refs]
                    nn = np.argsort(Dref, 1)[:, :k]
                    w = 1 / (np.take_along_axis(Dref, nn, 1) + 1e-12)
                    store[i] = (w * y[refs][nn]).sum(1) / w.sum(1)
                    errs.append(np.abs(store[i][refs] - y[refs]).mean())
                cand[(s_, k)] = (float(np.mean(errs)), store)
            for r_dim, alpha in ((16, 0.1), (16, 1.0), (8, 0.1)):
                errs, store = [], {}
                for i in range(M):
                    D = Dc[tuple(cols_of[i])]
                    n_ = len(D)
                    J = np.eye(n_) - 1 / n_
                    Bm = -0.5 * J @ (D ** 2) @ J
                    ew, ev = np.linalg.eigh(Bm)
                    order = np.argsort(ew)[::-1][:r_dim]
                    Z = ev[:, order] * np.sqrt(np.maximum(ew[order], 1e-12))
                    refs = np.where(allowed[i])[0]
                    Zr, yr = Z[refs], y[refs]
                    G = Zr.T @ Zr + alpha * np.eye(r_dim)
                    Ginv = np.linalg.inv(G)
                    beta = Ginv @ (Zr.T @ (yr - yr.mean()))
                    pred = np.clip(Z @ beta + yr.mean(), 0, 1)
                    h = np.einsum('jr,rs,js->j', Zr, Ginv, Zr)
                    pr = Zr @ beta + yr.mean()
                    loo = yr - (yr - pr) / np.maximum(1 - h, 1e-6)
                    pred = pred.copy()
                    pred[refs] = np.clip(loo, 0, 1)
                    store[i] = pred
                    errs.append(np.abs(pred[refs] - yr).mean())
                cand[(s_, 'r', r_dim, alpha)] = (float(np.mean(errs)), store)
        return min(cand.values(), key=lambda v: v[0])[1]

    out = {'ms': list(MS), 'by_m': {}}
    for m in MS:
        cols_of = {i: np.array(adaptive[i][0][:m]) for i in range(M)}
        irt_t = np.array([adaptive[i][1][m - 1] for i in range(M)])
        store = geom_store('qubric', cols_of)
        store_r = geom_store('raw', cols_of)
        curves = np.zeros((M, len(ALPHAS)))
        for i in range(M):
            refs = np.where(allowed[i])[0]
            irt_ref = np.array([models[i].predict(cols_of[i],
                                                  B[j, cols_of[i]])
                                for j in refs])
            curves[i] = np.abs(ALPHAS[None] * irt_ref[:, None]
                               + (1 - ALPHAS[None]) * store[i][refs, None]
                               - y[refs, None]).mean(0)
        a = ALPHAS[int(curves.mean(0).argmin())]
        geo_t = np.array([store[i][i] for i in range(M)])
        raw_t = np.array([store_r[i][i] for i in range(M)])
        e = {'irt': np.abs(irt_t - y), 'geom': np.abs(geo_t - y),
             'raw': np.abs(raw_t - y),
             'blend': np.abs(a * irt_t + (1 - a) * geo_t - y)}
        out['by_m'][m] = {n: dict(mae=float(v.mean()), ci=ci(v),
                                  sem=float(v.std(ddof=1) / np.sqrt(M)))
                          for n, v in e.items()}
        out['by_m'][m]['errs'] = {n: [round(float(x), 6) for x in v]
                                  for n, v in e.items()}
        out['by_m'][m]['errs']['sample'] = [
            round(float(abs(B[i, cols_of[i]].mean() - y[i])), 6)
            for i in range(M)]
        out['by_m'][m]['sample'] = dict(
            mae=float(np.mean(out['by_m'][m]['errs']['sample'])),
            ci=ci(np.array(out['by_m'][m]['errs']['sample'])))
        d_bi = e['blend'] - e['irt']
        out['by_m'][m]['delta_blend_irt'] = dict(mean=float(d_bi.mean()),
                                                 ci=ci(d_bi))
        print(m, {n: round(v.mean(), 4) for n, v in e.items()})
        json.dump(out, open(out_path, 'w'), indent=2)
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1])
