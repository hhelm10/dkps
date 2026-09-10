"""Figure 4 row 2: TB2 protocol curves (LOSO / leave-one-family-out /
leave-one-harness-out), random probes B=20 with shared draws.

Eval panel: kept systems with >=90% trace coverage (target AND reference);
y = official all-trials mean; B = the judged replicate's binary reward
(fallback: rounded panel mean where the chosen trial lacked a result).
Series: sample, IRT (2PL), raw-trace geometry, qubric geometry, qubric +
IRT blend -- pooled (sigma, k, alpha) per draw from honest reference
errors, bootstrap CIs. Writes figures/tb2_protocols.json (same schema as
q100_protocols.json).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS = (3, 5, 7, 10, 15)  # extended per tb2_tune.json (larger k wins on TB2)
RIDGE = ((16, 0.1), (16, 1.0), (8, 0.1))  # per tb2_tune3: ridge on MDS coords
ALPHAS = np.linspace(0, 1, 101)
MS = (1, 3, 5, 10, 20)
B_DRAWS = 50
DROP = {'Droid__GPT-5.3-Codex', 'Droid__Claude-Opus-4.6',
        'just-another-coding-agent__GLM-5'}
MIN_COV = 0.9
OUT = 'figures/tb2_protocols.json'


def main():
    d = json.load(open('data/terminal_bench/tb2_panel.json'))
    all_sys = [s for s in d['systems'] if s not in DROP]
    tasks = d['tasks']
    Q = len(tasks)
    cov = {s: sum(os.path.exists(f'data/terminal_bench/tb2_txt/{s}/{t}.txt')
                  for t in tasks) / Q for s in all_sys}
    systems = [s for s in all_sys if cov[s] >= MIN_COV]
    print(f'eval panel: {len(systems)} systems (dropped '
          f'{len(all_sys) - len(systems)} with coverage < {MIN_COV:.0%})')
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
    hs = [meta[s]['harness'] for s in systems]
    masks = {
        'system': np.array([[j != i for j in range(M)] for i in range(M)]),
        'family': np.array([[j != i and not (fams[i] and fams[i] & fams[j])
                             for j in range(M)] for i in range(M)]),
        'harness': np.array([[j != i and hs[i] != hs[j] for j in range(M)]
                             for i in range(M)]),
    }
    for k_, a in masks.items():
        print(f'{k_}: mean refs {a.sum(1).mean():.1f}')

    # embeddings: full-grid arrays subset to eval systems
    Xq = consensus_center(
        np.load('data/terminal_bench/tb2_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), len(all_sys))) \
        .reshape(len(all_sys), Q, -1)[
            [all_sys.index(s) for s in systems]].astype(np.float32)
    HT = np.load('data/terminal_bench/tb2_raw_emb_openai_small.npz')['HT'] \
        .reshape(len(all_sys), Q, -1)[[all_sys.index(s) for s in systems]]
    Xr = HT - np.median(HT, axis=0, keepdims=True)
    Xr /= np.maximum(np.linalg.norm(Xr, axis=-1, keepdims=True), 1e-9)
    Xr = Xr.astype(np.float32)

    z = np.load('data/terminal_bench/tb2_query_vecs_64.npz',
                allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(t) for t in tasks]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)
    kerns = {}
    for tag, X in (('qubric', Xq), ('raw', Xr)):
        for s_ in SIGS:
            KQ = np.exp(-D2q / (2 * med / s_))
            W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
            kerns[(tag, s_)] = (KQ, W)
    Xof = {'qubric': Xq, 'raw': Xr}

    def pkps_D(tag, cols, s_):
        X = Xof[tag]
        KQ, W = kerns[(tag, s_)]
        Xi = X[:, cols].reshape(M, -1)
        A_tr = (Xi @ W[:, cols].reshape(M, -1).T) / KQ[cols].sum()
        KQc = KQ[np.ix_(cols, cols)]
        Wc = np.einsum('qp,jpd->jqd', KQc, X[:, cols], optimize=True)
        Att = np.einsum('jqd,jqd->j', X[:, cols], Wc) / KQc.sum()
        Arr = np.einsum('jqd,jqd->j', X, W) / KQ.sum()
        return np.sqrt(np.maximum(Att[:, None] + Arr[None] - 2 * A_tr, 0))

    rng_b = np.random.default_rng(1)

    def ci(e):
        v = np.array([e[rng_b.integers(0, M, M)].mean() for _ in range(2000)])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(B_DRAWS)]
             for m in MS}

    out = {'ms': list(MS), 'b_draws': B_DRAWS,
           'n_systems': M, 'protocols': {}}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    for name, allowed in masks.items():
        if name in out['protocols']:
            print(f'=== {name}: cached ===')
            continue
        print(f'=== protocol: {name} ===')
        models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0)
                  for i in range(M)]
        pop_t = np.array([y[allowed[i]].mean() for i in range(M)])
        res = {'pop': dict(mae=float(np.abs(pop_t - y).mean()),
                           ci=ci(np.abs(pop_t - y))), 'by_m': {}}

        def geom(tag, cols):
            cand = {}
            for s_ in SIGS:
                D = pkps_D(tag, cols, s_)
                for k in KS:
                    errs, store = [], {}
                    for i in range(M):
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
                # ridge on classical-MDS coords (tb2_tune3): target row is
                # in the MDS but never in the ridge fit
                n_ = len(D)
                J = np.eye(n_) - 1 / n_
                Bm = -0.5 * J @ (D ** 2) @ J
                ew, ev = np.linalg.eigh(Bm)
                order = np.argsort(ew)[::-1]
                for r_dim, alpha in RIDGE:
                    Z = ev[:, order[:r_dim]] * np.sqrt(
                        np.maximum(ew[order[:r_dim]], 1e-12))
                    errs, store = [], {}
                    for i in range(M):
                        refs = np.where(allowed[i])[0]
                        Zr, yr = Z[refs], y[refs]
                        G = Zr.T @ Zr + alpha * np.eye(r_dim)
                        Ginv = np.linalg.inv(G)
                        beta = Ginv @ (Zr.T @ (yr - yr.mean()))
                        pred = np.clip(Z @ beta + yr.mean(), 0, 1)
                        # honest reference predictions: LOO via hat matrix
                        # (kNN store excludes self; ridge must too)
                        h = np.einsum('jr,rs,js->j', Zr, Ginv, Zr)
                        pred_ref = Z[refs] @ beta + yr.mean()
                        loo = yr - (yr - pred_ref) / np.maximum(1 - h, 1e-6)
                        pred = pred.copy()
                        pred[refs] = np.clip(loo, 0, 1)
                        store[i] = pred
                        errs.append(np.abs(pred[refs] - yr).mean())
                    cand[(s_, 'r', r_dim, alpha)] = (float(np.mean(errs)),
                                                     store)
            return min(cand.values(), key=lambda v: v[0])[1]

        for m in MS:
            acc = {n: np.zeros(M)
                   for n in ('sample', 'irt', 'raw', 'geom', 'blend')}
            for cols in draws[m]:
                acc['sample'] += np.abs(B[:, cols].mean(1) - y) / B_DRAWS
                irt_t = np.array([models[i].predict(cols, B[i, cols])
                                  for i in range(M)])
                store = geom('qubric', cols)
                store_r = geom('raw', cols)
                curves = np.zeros((M, len(ALPHAS)))
                for i in range(M):
                    refs = np.where(allowed[i])[0]
                    irt_ref = np.array([models[i].predict(cols, B[j, cols])
                                        for j in refs])
                    curves[i] = np.abs(
                        ALPHAS[None] * irt_ref[:, None]
                        + (1 - ALPHAS[None]) * store[i][refs, None]
                        - y[refs, None]).mean(0)
                a = ALPHAS[int(curves.mean(0).argmin())]
                geo_t = np.array([store[i][i] for i in range(M)])
                raw_t = np.array([store_r[i][i] for i in range(M)])
                acc['irt'] += np.abs(irt_t - y) / B_DRAWS
                acc['geom'] += np.abs(geo_t - y) / B_DRAWS
                acc['raw'] += np.abs(raw_t - y) / B_DRAWS
                acc['blend'] += np.abs(a * irt_t + (1 - a) * geo_t
                                       - y) / B_DRAWS
            res['by_m'][m] = {n: dict(mae=float(acc[n].mean()), ci=ci(acc[n]),
                                       sem=float(acc[n].std(ddof=1)
                                                 / np.sqrt(M)))
                              for n in acc}
            res['by_m'][m]['errs'] = {n: [round(float(v), 6) for v in acc[n]]
                                      for n in acc}
            d_bi = acc['blend'] - acc['irt']
            res['by_m'][m]['delta_blend_irt'] = dict(mean=float(d_bi.mean()),
                                                     ci=ci(d_bi))
            print(m, {n: round(acc[n].mean(), 4) for n in acc})
        out['protocols'][name] = res
        json.dump(out, open(OUT, 'w'), indent=2)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
