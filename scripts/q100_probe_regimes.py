"""Probe-selection regimes for the q100 table (per HH 2026-09-04):

  random     B=20 random probe sets per budget m; (sigma, k, alpha) selected
             per draw on pooled reference errors (deployable); per-system
             errors averaged over draws, then bootstrap CI over systems.
  geom-var   trace-native selection, label-free: rank instances by total
             dispersion of centered embeddings across reference systems
             (per target, computed on its allowed references only), take
             top-m. Pooled (sigma, k, alpha) selection as in the canonical
             table.

Writes figures/q100_probe_regimes.json and prints both tables.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS = (3, 5)
ALPHAS = np.linspace(0, 1, 101)
BOOT = 2000
MS = (1, 3, 5, 10, 20)
B_DRAWS = 20


def main():
    systems, q100, y, B, allowed = load_panel(panel='data/judge/q100.json')
    M, Q = B.shape
    models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0) for i in range(M)]
    X = np.load('data/judge/q100_emb_openai_small.npz')['X']
    Xc = consensus_center(X, np.tile(np.arange(Q), M)) \
        .reshape(M, Q, -1).astype(np.float32)
    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)
    kern = {}
    for s_ in SIGS:
        KQ = np.exp(-D2q / (2 * med / s_))
        W = np.einsum('qp,jpd->jqd', KQ, Xc)
        kern[s_] = (KQ, W, np.einsum('jqd,jqd->j', Xc, W) / KQ.sum())

    def pkps_D(cols, s_):
        KQ, W, Arr = kern[s_]
        A_tr = np.einsum('iqd,jqd->ij', Xc[:, cols], W[:, cols]) / KQ[cols].sum()
        KQc = KQ[np.ix_(cols, cols)]
        Wc = np.einsum('qp,jpd->jqd', KQc, Xc[:, cols])
        Att = np.einsum('iqd,iqd->i', Xc[:, cols], Wc) / KQc.sum()
        return np.sqrt(np.maximum(Att[:, None] + Arr[None] - 2 * A_tr, 0))

    def eval_cols(cols_of):
        """cols_of: dict target -> probe columns. Pooled (sigma,k) + global
        alpha on references. Returns per-system error vectors."""
        irt_of = {i: np.array([models[j].predict(cols_of[i], B[j, cols_of[i]])
                               for j in range(M)]) for i in range(M)}
        cand = {}
        for s_ in SIGS:
            Dc = {}
            for i in range(M):
                key = tuple(cols_of[i])
                if key not in Dc:
                    Dc[key] = pkps_D(cols_of[i], s_)
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
        (s_b, k_b), (_, store) = min(cand.items(), key=lambda kv: kv[1][0])
        curves = np.zeros((M, len(ALPHAS)))
        for i in range(M):
            refs = np.where(allowed[i])[0]
            curves[i] = np.abs(ALPHAS[None] * irt_of[i][refs, None]
                               + (1 - ALPHAS[None]) * store[i][refs, None]
                               - y[refs, None]).mean(0)
        a = ALPHAS[int(curves.mean(0).argmin())]
        e = {n: np.zeros(M) for n in ('irt', 'geom', 'blend')}
        for i in range(M):
            e['irt'][i] = abs(irt_of[i][i] - y[i])
            e['geom'][i] = abs(store[i][i] - y[i])
            e['blend'][i] = abs(a * irt_of[i][i]
                                + (1 - a) * store[i][i] - y[i])
        return e, (s_b, k_b, float(a))

    rng = np.random.default_rng(0)
    out = {}

    def report(tag, e_sys, kern_info=None):
        rng2 = np.random.default_rng(1)
        row = {}
        for n in ('irt', 'geom', 'blend'):
            errs = e_sys[n]
            vals = np.array([errs[rng2.integers(0, M, M)].mean()
                             for _ in range(BOOT)])
            row[n] = dict(mae=float(errs.mean()),
                          ci=[float(np.percentile(vals, 2.5)),
                              float(np.percentile(vals, 97.5))])
        d = e_sys['blend'] - e_sys['irt']
        vals = np.array([d[rng2.integers(0, M, M)].mean() for _ in range(BOOT)])
        row['delta'] = dict(mean=float(d.mean()),
                            ci=[float(np.percentile(vals, 2.5)),
                                float(np.percentile(vals, 97.5))])
        if kern_info:
            row['kernel'] = kern_info
        f = lambda r: f"{r['mae']:.4f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"  # noqa: E731
        print(f"{tag:>14} irt {f(row['irt'])}  geom {f(row['geom'])}  "
              f"blend {f(row['blend'])}  d {row['delta']['mean']:+.4f} "
              f"[{row['delta']['ci'][0]:+.3f},{row['delta']['ci'][1]:+.3f}]")
        return row

    # ---------------- random, B=20 draws ----------------
    print('=== RANDOM PROBES (B=20 draws per m) ===')
    out['random'] = {}
    for m in MS:
        acc = {n: np.zeros(M) for n in ('irt', 'geom', 'blend')}
        for b in range(B_DRAWS):
            cols = rng.choice(Q, m, replace=False)
            e, _ = eval_cols({i: cols for i in range(M)})
            for n in acc:
                acc[n] += e[n] / B_DRAWS
        out['random'][m] = report(f'm={m}', acc)

    # ---------------- geometry-selected (label-free) ----------------
    print('=== GEOMETRY-SELECTED PROBES (dispersion, label-free) ===')
    out['geom_var'] = {}
    disp_of = {}
    for i in range(M):
        refs = np.where(allowed[i])[0]
        disp = np.linalg.norm(Xc[refs], axis=-1).mean(0) * 0  # placeholder
        disp = (Xc[refs] ** 2).sum(-1).mean(0)  # since centered: dispersion
        disp_of[i] = np.argsort(-disp)
    for m in MS:
        cols_of = {i: disp_of[i][:m] for i in range(M)}
        e, kinfo = eval_cols(cols_of)
        out['geom_var'][m] = report(f'm={m}', e, kinfo)

    json.dump(out, open('figures/q100_probe_regimes.json', 'w'), indent=2)
    print('wrote figures/q100_probe_regimes.json')


if __name__ == '__main__':
    main()
