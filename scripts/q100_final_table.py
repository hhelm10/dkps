"""Paper-grade q100 results table: per-query-set bandwidth CV + bootstrap CIs.

Per HH (2026-09-04): for each target's probe set (its informative panel at
budget m), the PKPS bandwidth sigma and kNN k are selected on that target's
leave-one-LLM-out references evaluated on the SAME probe set -- the
hyperparameters a practitioner could actually choose. Alpha (blend weight)
likewise per-target from references.

Uncertainty: 95% bootstrap CIs over the 107 target systems (2000 resamples),
plus the PAIRED bootstrap CI on (blend - IRT) -- the trace-increment claim.

Writes figures/q100_final_table.json and prints the table.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (1, 2, 4, 8, 16, 32)          # sigma^2 = median / s
KS = (3, 5)
ALPHAS = np.linspace(0, 1, 101)
BOOT = 2000
MS = (1, 3, 5, 10, 20)


def boot_ci(errs, rng, stat=np.mean):
    n = len(errs)
    vals = np.array([stat(errs[rng.integers(0, n, n)]) for _ in range(BOOT)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    systems, q100, y, B, allowed = load_panel(panel='data/judge/q100.json')
    M, Q = B.shape
    models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0) for i in range(M)]
    info_order = [mm.informative_order() for mm in models]
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

    rng = np.random.default_rng(0)
    out = {}
    print(f'{"m":>3} {"IRT":>16} {"geom(CVed)":>16} {"blend":>16} '
          f'{"blend-IRT delta":>20}')
    for m in MS:
        e_irt = np.zeros(M)
        e_geo = np.zeros(M)
        e_bl = np.zeros(M)
        for i in range(M):
            cols = np.array(info_order[i][:m])
            refs = np.where(allowed[i])[0]
            irt_all = np.array([models[j].predict(cols, B[j, cols])
                                for j in range(M)])
            # per-query-set (sigma, k) selection on references
            best = (np.inf, None, None, None)
            for s_ in SIGS:
                D = pkps_D(cols, s_)
                Dref = D[:, refs].copy()
                for r, j in enumerate(refs):
                    Dref[j, r] = np.inf
                Dref[i] = D[i, refs]
                for k in KS:
                    nn = np.argsort(Dref, 1)[:, :k]
                    w = 1 / (np.take_along_axis(Dref, nn, 1) + 1e-12)
                    geo_all = (w * y[refs][nn]).sum(1) / w.sum(1)
                    ref_err = np.abs(geo_all[refs] - y[refs]).mean()
                    if ref_err < best[0]:
                        best = (ref_err, s_, k, geo_all)
            _, s_b, k_b, geo_all = best
            # per-target alpha on references under the chosen kernel
            E = np.abs(ALPHAS[None] * irt_all[refs, None]
                       + (1 - ALPHAS[None]) * geo_all[refs, None]
                       - y[refs, None]).mean(0)
            a = ALPHAS[int(E.argmin())]
            e_irt[i] = abs(irt_all[i] - y[i])
            e_geo[i] = abs(geo_all[i] - y[i])
            e_bl[i] = abs(a * irt_all[i] + (1 - a) * geo_all[i] - y[i])
        row = {}
        for name, e in (('irt', e_irt), ('geom', e_geo), ('blend', e_bl)):
            lo, hi = boot_ci(e, rng)
            row[name] = dict(mae=float(e.mean()), ci=[lo, hi])
        d = e_bl - e_irt
        lo, hi = boot_ci(d, rng)
        row['delta_blend_minus_irt'] = dict(mean=float(d.mean()), ci=[lo, hi],
                                            frac_boot_below_zero=float(
            np.mean([d[rng.integers(0, M, M)].mean() < 0
                     for _ in range(BOOT)])))
        out[m] = row
        f = lambda r: f"{r['mae']:.4f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"  # noqa: E731
        dd = row['delta_blend_minus_irt']
        print(f"{m:3d} {f(row['irt']):>16} {f(row['geom']):>16} "
              f"{f(row['blend']):>16}  {dd['mean']:+.4f} "
              f"[{dd['ci'][0]:+.3f},{dd['ci'][1]:+.3f}]")
    json.dump(out, open('figures/q100_final_table.json', 'w'), indent=2)
    print('wrote figures/q100_final_table.json')


if __name__ == '__main__':
    main()
