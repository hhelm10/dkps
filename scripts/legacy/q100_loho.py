"""Leave-one-harness-out table (per HH 2026-09-08): references must share
the target's HARNESS but same-LLM references are ALLOWED --
(scaffold keyword heuristic, as pillars.py). Strictly harder than the
canonical leave-one-LLM-out protocol; the delta vs the main tables isolates
the contribution of scaffold siblings to prediction quality.

Rows (both regimes): adaptive 2PL IRT, qubric PKPS geometry, qubric +
adaptive blend (+ random regime with random panels), pooled (sigma, k,
alpha) on the restricted references, bootstrap CIs, paired deltas.

Writes figures/q100_loho.json.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from pillars import harness_tag  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS = (3, 5)
ALPHAS = np.linspace(0, 1, 101)
MS = (1, 3, 5, 10, 20)
B_DRAWS = 20


def main():
    systems, q100, y, B, allowed_llm = load_panel(panel='data/judge/q100.json')
    M, Q = B.shape
    scaf = [harness_tag(s) for s in systems]
    n_tag = sum(t is not None for t in scaf)
    allowed = np.array(
        [[j != i and not (scaf[i] and scaf[j] == scaf[i]) for j in range(M)]
         for i in range(M)])
    print(f'harness tags: {n_tag}/{M} systems; mean references per target: '
          f'{allowed.sum(1).mean():.1f} (LLM-out only: '
          f'{allowed_llm.sum(1).mean():.1f})')

    models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0) for i in range(M)]
    adaptive = [models[i].adaptive_path(B[i]) for i in range(M)]
    Xc = consensus_center(np.load('data/judge/q100_emb_openai_small.npz')['X'],
                          np.tile(np.arange(Q), M)) \
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

    rng_b = np.random.default_rng(1)

    def ci(e):
        v = np.array([e[rng_b.integers(0, M, M)].mean() for _ in range(2000)])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def eval_cols(cols_of, irt_t):
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
            cols = cols_of[i]
            refs = np.where(allowed[i])[0]
            irt_ref = np.array([models[i].predict(cols, B[j, cols])
                                for j in refs])
            curves[i] = np.abs(ALPHAS[None] * irt_ref[:, None]
                               + (1 - ALPHAS[None]) * store[i][refs, None]
                               - y[refs, None]).mean(0)
        a = ALPHAS[int(curves.mean(0).argmin())]
        geo_t = np.array([store[i][i] for i in range(M)])
        return (np.abs(irt_t - y), np.abs(geo_t - y),
                np.abs(a * irt_t + (1 - a) * geo_t - y))

    out = {'n_harness_tagged': n_tag,
           'mean_refs': float(allowed.sum(1).mean()), 'adaptive': {},
           'random': {}}
    print('=== adaptive panels (LOHO) ===')
    for m in MS:
        cols_of = {i: np.array(adaptive[i][0][:m]) for i in range(M)}
        irt_t = np.array([adaptive[i][1][m - 1] for i in range(M)])
        e_irt, e_geo, e_bl = eval_cols(cols_of, irt_t)
        d = e_bl - e_irt
        vd = np.array([d[rng_b.integers(0, M, M)].mean() for _ in range(2000)])
        out['adaptive'][m] = dict(
            irt=dict(mae=float(e_irt.mean()), ci=ci(e_irt)),
            geom=dict(mae=float(e_geo.mean()), ci=ci(e_geo)),
            blend=dict(mae=float(e_bl.mean()), ci=ci(e_bl)),
            delta=dict(mean=float(d.mean()),
                       ci=[float(np.percentile(vd, 2.5)),
                           float(np.percentile(vd, 97.5))]))
        print(m, round(e_irt.mean(), 4), round(e_geo.mean(), 4),
              round(e_bl.mean(), 4))
    print('=== random panels (LOHO, B=20) ===')
    rng = np.random.default_rng(0)
    for m in MS:
        acc = {n: np.zeros(M) for n in ('irt', 'geom', 'blend')}
        for b in range(B_DRAWS):
            cols = rng.choice(Q, m, replace=False)
            irt_t = np.array([models[i].predict(cols, B[i, cols])
                              for i in range(M)])
            e_irt, e_geo, e_bl = eval_cols({i: cols for i in range(M)}, irt_t)
            acc['irt'] += e_irt / B_DRAWS
            acc['geom'] += e_geo / B_DRAWS
            acc['blend'] += e_bl / B_DRAWS
        d = acc['blend'] - acc['irt']
        vd = np.array([d[rng_b.integers(0, M, M)].mean() for _ in range(2000)])
        out['random'][m] = {n: dict(mae=float(acc[n].mean()), ci=ci(acc[n]))
                            for n in acc}
        out['random'][m]['delta'] = dict(
            mean=float(d.mean()), ci=[float(np.percentile(vd, 2.5)),
                                      float(np.percentile(vd, 97.5))])
        print(m, {n: round(acc[n].mean(), 4) for n in acc})
    json.dump(out, open('figures/q100_loho.json', 'w'), indent=2)
    print('wrote figures/q100_loho.json')


if __name__ == '__main__':
    main()
