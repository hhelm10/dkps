"""Generic-rubric GEOMETRY row for the hero table (per HH: 'hero table
needs to include the generic rubric (not with an IRT blend)').

Same publication config as the qubric/raw series: leave-one-family-out,
consensus centering, PKPS kernel, per-draw pooled CV over sigma x
{kNN k-grid} u {ridge-on-MDS, LOO-honest}, B=50 shared random draws
(identical rng(0) stream as fig_protocols/tb2_eval) plus the adaptive
CAT paths (identical family-out 2PL fits as adaptive_family).

Appends 'generic' (mae/ci/sem + errs) into:
  figures/q100_protocols.json          protocols.family.by_m[m]
  figures/tb2_protocols.json           protocols.family.by_m[m]
  figures/{q100,tb2}_adaptive_family.json   by_m[m]

Usage: python scripts/generic_eval.py swe|tb2
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
RIDGE = ((16, 0.1), (16, 1.0), (8, 0.1))
MS = (1, 3, 5, 10, 20)
B_DRAWS = 50
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
    Xg = consensus_center(
        np.load('data/judge/q100_generic_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), M)).reshape(M, Q, -1).astype(np.float32)
    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]
    return y, B, allowed, Xg, V, (3, 5), \
        'figures/q100_protocols.json', 'figures/q100_adaptive_family.json'


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
    Xg = consensus_center(
        np.load('data/terminal_bench/tb2_generic_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), len(all_sys))) \
        .reshape(len(all_sys), Q, -1)[
            [all_sys.index(s) for s in systems]].astype(np.float32)
    z = np.load('data/terminal_bench/tb2_query_vecs_64.npz',
                allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(t) for t in tasks]]
    return y, B, allowed, Xg, V, (3, 5, 7, 10, 15), \
        'figures/tb2_protocols.json', 'figures/tb2_adaptive_family.json'


def main(bench):
    y, B, allowed, X, V, KS, rand_path, adap_path = \
        load_swe() if bench == 'swe' else load_tb2()
    M, Q = B.shape
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)
    kerns = {}
    for s_ in SIGS:
        KQ = np.exp(-D2q / (2 * med / s_))
        W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
        kerns[s_] = (KQ, W, np.einsum('jqd,jqd->j', X, W) / KQ.sum())

    def pkps_D(cols, s_):
        KQ, W, Arr = kerns[s_]
        Xi = X[:, cols].reshape(M, -1)
        A_tr = (Xi @ W[:, cols].reshape(M, -1).T) / KQ[cols].sum()
        KQc = KQ[np.ix_(cols, cols)]
        Wc = np.einsum('qp,jpd->jqd', KQc, X[:, cols], optimize=True)
        Att = np.einsum('jqd,jqd->j', X[:, cols], Wc) / KQc.sum()
        return np.sqrt(np.maximum(Att[:, None] + Arr[None] - 2 * A_tr, 0))

    rng_b = np.random.default_rng(1)

    def ci(e):
        v = np.array([e[rng_b.integers(0, M, M)].mean()
                      for _ in range(2000)])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def geom_tgt(cols):
        """CV-selected target predictions for one probe panel."""
        cand = {}
        for s_ in SIGS:
            D = pkps_D(cols, s_)
            for k in KS:
                errs, tgt = [], np.zeros(M)
                for i in range(M):
                    refs = np.where(allowed[i])[0]
                    Dref = D[:, refs].copy()
                    for r, j in enumerate(refs):
                        Dref[j, r] = np.inf
                    Dref[i] = D[i, refs]
                    nn = np.argsort(Dref, 1)[:, :k]
                    w = 1 / (np.take_along_axis(Dref, nn, 1) + 1e-12)
                    pred = (w * y[refs][nn]).sum(1) / w.sum(1)
                    errs.append(np.abs(pred[refs] - y[refs]).mean())
                    tgt[i] = pred[i]
                cand[(s_, k)] = (float(np.mean(errs)), tgt)
            n_ = len(D)
            J = np.eye(n_) - 1 / n_
            Bm = -0.5 * J @ (D ** 2) @ J
            ew, ev = np.linalg.eigh(Bm)
            order = np.argsort(ew)[::-1]
            for r_dim, alpha in RIDGE:
                Z = ev[:, order[:r_dim]] * np.sqrt(
                    np.maximum(ew[order[:r_dim]], 1e-12))
                errs, tgt = [], np.zeros(M)
                for i in range(M):
                    refs = np.where(allowed[i])[0]
                    Zr, yr = Z[refs], y[refs]
                    G = Zr.T @ Zr + alpha * np.eye(r_dim)
                    Ginv = np.linalg.inv(G)
                    beta = Ginv @ (Zr.T @ (yr - yr.mean()))
                    tgt[i] = float(np.clip(Z[i] @ beta + yr.mean(), 0, 1))
                    h = np.einsum('jr,rs,js->j', Zr, Ginv, Zr)
                    pr = Zr @ beta + yr.mean()
                    loo = yr - (yr - pr) / np.maximum(1 - h, 1e-6)
                    errs.append(np.abs(np.clip(loo, 0, 1) - yr).mean())
                cand[(s_, 'r', r_dim, alpha)] = (float(np.mean(errs)), tgt)
        return min(cand.values(), key=lambda v: v[0])[1]

    # --- random probes, shared draw stream (rng(0), all MS) ---
    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(B_DRAWS)]
             for m in MS}
    out = json.load(open(rand_path))
    fam_by_m = out['protocols']['family']['by_m']
    for m in MS:
        if 'generic' in fam_by_m[str(m)]:
            print(f'random m={m}: cached')
            continue
        acc = np.zeros(M)
        for cols in draws[m]:
            acc += np.abs(geom_tgt(cols) - y) / B_DRAWS
        fam_by_m[str(m)]['generic'] = dict(
            mae=float(acc.mean()), ci=ci(acc),
            sem=float(acc.std(ddof=1) / np.sqrt(M)))
        fam_by_m[str(m)].setdefault('errs', {})['generic'] = \
            [round(float(v), 6) for v in acc]
        json.dump(out, open(rand_path, 'w'), indent=2)
        print(f'random m={m}: generic mae {acc.mean():.4f}', flush=True)

    # --- adaptive CAT paths (identical family-out 2PL fits) ---
    out_a = json.load(open(adap_path))
    if all('generic' in out_a['by_m'][str(m)] for m in MS):
        print('adaptive: cached')
        return
    print('fitting per-target 2PL + CAT paths...', flush=True)
    models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0)
              for i in range(M)]
    adaptive = [models[i].adaptive_path(B[i]) for i in range(M)]
    for m in MS:
        if 'generic' in out_a['by_m'][str(m)]:
            continue
        cols_of = {i: np.array(adaptive[i][0][:m]) for i in range(M)}
        # per-target panels: CV over the union, per-panel D like
        # adaptive_family.geom_store
        cand = {}
        for s_ in SIGS:
            Dc = {c: pkps_D(np.asarray(c), s_)
                  for c in {tuple(cols_of[i]) for i in range(M)}}
            for k in KS:
                errs, tgt = [], np.zeros(M)
                for i in range(M):
                    D = Dc[tuple(cols_of[i])]
                    refs = np.where(allowed[i])[0]
                    Dref = D[:, refs].copy()
                    for r, j in enumerate(refs):
                        Dref[j, r] = np.inf
                    Dref[i] = D[i, refs]
                    nn = np.argsort(Dref, 1)[:, :k]
                    w = 1 / (np.take_along_axis(Dref, nn, 1) + 1e-12)
                    pred = (w * y[refs][nn]).sum(1) / w.sum(1)
                    errs.append(np.abs(pred[refs] - y[refs]).mean())
                    tgt[i] = pred[i]
                cand[(s_, k)] = (float(np.mean(errs)), tgt)
            for r_dim, alpha in RIDGE:
                errs, tgt = [], np.zeros(M)
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
                    tgt[i] = float(np.clip(Z[i] @ beta + yr.mean(), 0, 1))
                    h = np.einsum('jr,rs,js->j', Zr, Ginv, Zr)
                    pr = Zr @ beta + yr.mean()
                    loo = yr - (yr - pr) / np.maximum(1 - h, 1e-6)
                    errs.append(np.abs(np.clip(loo, 0, 1) - yr).mean())
                cand[(s_, 'r', r_dim, alpha)] = (float(np.mean(errs)), tgt)
        tgt = min(cand.values(), key=lambda v: v[0])[1]
        e = np.abs(tgt - y)
        out_a['by_m'][str(m)]['generic'] = dict(
            mae=float(e.mean()), ci=ci(e),
            sem=float(e.std(ddof=1) / np.sqrt(M)))
        out_a['by_m'][str(m)].setdefault('errs', {})['generic'] = \
            [round(float(v), 6) for v in e]
        json.dump(out_a, open(adap_path, 'w'), indent=2)
        print(f'adaptive m={m}: generic mae {e.mean():.4f}', flush=True)
    print('wrote', adap_path)


if __name__ == '__main__':
    main(sys.argv[1])
