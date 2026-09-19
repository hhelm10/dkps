"""Estimator-side tuning for TB2 qubric geometry (per HH: 'are we
optimizing pkps?' -- the SWE estimator layer was tuned over weeks; this
gives TB2 the same treatment, reference-side only).

Search space (family-out protocol, pooled reference error, B=10 draws,
m in {1, 5, 20}):
  kernel     RBF over instruction PCA-64 qvecs, sigma^2 = med/s for
             s in {1..256}; plus 'uniform' (mean-pool) and 'identity'
             (no cross-task smoothing)
  estimator  inverse-distance kNN, k in {1,3,5,7,10,15};
             softmax over ALL refs, temperature t*median(D),
             t in {0.05, 0.1, 0.2, 0.5, 1.0}
Selection is per-m pooled over references (honest); reports the winner
config and its target MAE, vs the current default. Writes
figures/tb2_tune.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS = (1, 3, 5, 7, 10, 15)
TAUS = (0.05, 0.1, 0.2, 0.5, 1.0)
MS = (1, 5, 20)
B_DRAWS = 10
DROP = {'Droid__GPT-5.3-Codex', 'Droid__Claude-Opus-4.6',
        'just-another-coding-agent__GLM-5'}


def main():
    d = json.load(open('data/terminal_bench/tb2_panel.json'))
    all_sys = [s for s in d['systems'] if s not in DROP]
    tasks = d['tasks']
    Q = len(tasks)
    cov = {s: sum(os.path.exists(f'data/terminal_bench/tb2_txt/{s}/{t}.txt')
                  for t in tasks) / Q for s in all_sys}
    systems = [s for s in all_sys if cov[s] >= 0.9]
    M = len(systems)
    sidx = [d['systems'].index(s) for s in systems]
    y = np.array(d['y'])[sidx]
    meta = d['meta']
    fams = [set(meta[s]['llm_tags']) for s in systems]
    allowed = np.array([[j != i and not (fams[i] and fams[i] & fams[j])
                         for j in range(M)] for i in range(M)])

    X = consensus_center(
        np.load('data/terminal_bench/tb2_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), len(all_sys))) \
        .reshape(len(all_sys), Q, -1)[
            [all_sys.index(s) for s in systems]].astype(np.float32)

    z = np.load('data/terminal_bench/tb2_query_vecs_64.npz',
                allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(t) for t in tasks]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)

    kern_specs = [('rbf', s) for s in SIGS] + [('uniform', 0),
                                              ('identity', 0)]
    kerns = {}
    for kind, s_ in kern_specs:
        if kind == 'rbf':
            KQ = np.exp(-D2q / (2 * med / s_))
        elif kind == 'uniform':
            KQ = np.ones((Q, Q), np.float32)
        else:
            KQ = np.eye(Q, dtype=np.float32)
        W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
        kerns[(kind, s_)] = (KQ, W)

    def pkps_D(spec, cols):
        KQ, W = kerns[spec]
        Xi = X[:, cols].reshape(M, -1)
        A_tr = (Xi @ W[:, cols].reshape(M, -1).T) / KQ[cols].sum()
        KQc = KQ[np.ix_(cols, cols)]
        Wc = np.einsum('qp,jpd->jqd', KQc, X[:, cols], optimize=True)
        Att = np.einsum('jqd,jqd->j', X[:, cols], Wc) / KQc.sum()
        Arr = np.einsum('jqd,jqd->j', X, W) / KQ.sum()
        return np.sqrt(np.maximum(Att[:, None] + Arr[None] - 2 * A_tr, 0))

    def estimators(D):
        """-> {est_name: store} where store[i] = predictions for all
        systems using i's reference set."""
        out = {}
        base = np.median(D[D > 0]) + 1e-12
        for i_est, param in ([('knn', k) for k in KS]
                             + [('soft', t) for t in TAUS]):
            errs, store = [], {}
            for i in range(M):
                refs = np.where(allowed[i])[0]
                Dref = D[:, refs].copy()
                for r, j in enumerate(refs):
                    Dref[j, r] = np.inf
                Dref[i] = D[i, refs]
                if i_est == 'knn':
                    nn = np.argsort(Dref, 1)[:, :param]
                    w = 1 / (np.take_along_axis(Dref, nn, 1) + 1e-12)
                    store[i] = (w * y[refs][nn]).sum(1) / w.sum(1)
                else:
                    w = np.exp(-Dref / (param * base))
                    w[~np.isfinite(w)] = 0
                    w = np.where(np.isinf(Dref), 0, w) + 1e-15
                    store[i] = (w * y[refs][None, :].repeat(M, 0)).sum(1) \
                        / w.sum(1)
                errs.append(np.abs(store[i][refs] - y[refs]).mean())
            out[(i_est, param)] = (float(np.mean(errs)), store)
        return out

    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(B_DRAWS)]
             for m in MS}
    results = {}
    for m in MS:
        agg = {}
        tgt = {}
        for cols in draws[m]:
            for spec in kern_specs:
                D = pkps_D(spec, np.asarray(cols))
                for est, (ref_err, store) in estimators(D).items():
                    key = (spec, est)
                    agg[key] = agg.get(key, 0) + ref_err / B_DRAWS
                    e_t = np.abs(np.array([store[i][i] for i in range(M)])
                                 - y).mean()
                    tgt[key] = tgt.get(key, 0) + e_t / B_DRAWS
        best = min(agg, key=agg.get)
        default = ((('rbf', 4), ('knn', 3))
                   if (('rbf', 4), ('knn', 3)) in agg else best)
        results[m] = dict(
            best=dict(kernel=list(best[0]), est=list(best[1]),
                      ref_mae=round(agg[best], 4),
                      target_mae=round(tgt[best], 4)),
            default=dict(target_mae=round(tgt[default], 4)),
            top5=[dict(kernel=list(k[0]), est=list(k[1]),
                       ref=round(agg[k], 4), tgt=round(tgt[k], 4))
                  for k in sorted(agg, key=agg.get)[:5]])
        print(f'm={m}: best {best} ref {agg[best]:.4f} '
              f'target {tgt[best]:.4f} (default target '
              f'{tgt[default]:.4f})')
    json.dump(results, open('figures/tb2_tune.json', 'w'), indent=2)
    print('wrote figures/tb2_tune.json')


if __name__ == '__main__':
    main()
