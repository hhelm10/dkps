"""TB2 tuning round 3: representation content + estimator family.

  (a) SECTION SUBSETS: single sections, leave-one-out, and all -- tests
      the outcome-collapse hypothesis (drop outcome-adjacent sections
      like verification/final_state to expose behavioral signal).
  (b) RIDGE ON PERSPECTIVE COORDS: classical-MDS r-dim coords per draw,
      ridge fit on references (target row excluded from the fit, included
      in the MDS, per the DKPS convention) -- the ICML estimator family,
      never tried on TB2.

Family-out, pooled reference selection, B=8, m in {1,5,20}.
Writes figures/tb2_tune3.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from dkps.traces.qubric import DEFAULT_SECTIONS, consensus_center  # noqa: E402

SIGS = (2, 4, 8)
KS = (5, 7, 15)
MS = (1, 5, 20)
B_DRAWS = 8
RIDGE_R = (4, 8, 16)
RIDGE_A = (0.1, 1.0, 10.0)
DROP = {'Droid__GPT-5.3-Codex', 'Droid__Claude-Opus-4.6',
        'just-another-coding-agent__GLM-5'}
SECS = list(DEFAULT_SECTIONS)


def cmds(D, r):
    n = len(D)
    J = np.eye(n) - 1 / n
    Bm = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1][:r]
    return v[:, idx] * np.sqrt(np.maximum(w[idx], 1e-12))


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

    X6 = consensus_center(
        np.load('data/terminal_bench/tb2_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), len(all_sys))) \
        .reshape(len(all_sys), Q, len(SECS), -1)[
            [all_sys.index(s) for s in systems]].astype(np.float32)

    z = np.load('data/terminal_bench/tb2_query_vecs_64.npz',
                allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(t) for t in tasks]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)

    subsets = ([('all', tuple(range(len(SECS))))]
               + [(f'only:{s}', (i,)) for i, s in enumerate(SECS)]
               + [(f'drop:{s}', tuple(j for j in range(len(SECS))
                                      if j != i))
                  for i, s in enumerate(SECS)]
               + [('behav(U+L+E)', (0, 1, 3))])

    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(B_DRAWS)]
             for m in MS}

    res = {}
    for sub_name, sub in subsets:
        X = X6[:, :, sub, :].reshape(M, Q, -1)
        X /= np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)
        for s_ in SIGS:
            KQ = np.exp(-D2q / (2 * med / s_))
            W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
            Arr = np.einsum('jqd,jqd->j', X, W) / KQ.sum()
            for m in MS:
                for cols in draws[m]:
                    cols = np.asarray(cols)
                    Xi = X[:, cols].reshape(M, -1)
                    A_tr = (Xi @ W[:, cols].reshape(M, -1).T) \
                        / KQ[cols].sum()
                    KQc = KQ[np.ix_(cols, cols)]
                    Wc = np.einsum('qp,jpd->jqd', KQc, X[:, cols],
                                   optimize=True)
                    Att = np.einsum('jqd,jqd->j', X[:, cols], Wc) \
                        / KQc.sum()
                    D = np.sqrt(np.maximum(
                        Att[:, None] + Arr[None] - 2 * A_tr, 0))
                    # kNN estimators
                    for k in KS:
                        errs, tgts = [], []
                        for i in range(M):
                            refs = np.where(allowed[i])[0]
                            Dref = D[:, refs].copy()
                            for r_, j in enumerate(refs):
                                Dref[j, r_] = np.inf
                            Dref[i] = D[i, refs]
                            nn = np.argsort(Dref, 1)[:, :k]
                            w = 1 / (np.take_along_axis(Dref, nn, 1)
                                     + 1e-12)
                            pred = (w * y[refs][nn]).sum(1) / w.sum(1)
                            errs.append(np.abs(pred[refs]
                                               - y[refs]).mean())
                            tgts.append(abs(pred[i] - y[i]))
                        key = (sub_name, s_, f'knn{k}', m)
                        a = res.setdefault(key, [0.0, 0.0])
                        a[0] += np.mean(errs) / B_DRAWS
                        a[1] += np.mean(tgts) / B_DRAWS
                    # ridge on MDS coords
                    for r in RIDGE_R:
                        Z = cmds(D, r)
                        for alpha in RIDGE_A:
                            errs, tgts = [], []
                            for i in range(M):
                                refs = np.where(allowed[i])[0]
                                Zr_, yr = Z[refs], y[refs]
                                G = Zr_.T @ Zr_ + alpha * np.eye(r)
                                beta = np.linalg.solve(G, Zr_.T @ (yr - yr.mean()))
                                pred_all = Z @ beta + yr.mean()
                                pred_all = np.clip(pred_all, 0, 1)
                                errs.append(np.abs(pred_all[refs]
                                                   - yr).mean())
                                tgts.append(abs(pred_all[i] - y[i]))
                            key = (sub_name, s_, f'ridge{r}a{alpha}', m)
                            a = res.setdefault(key, [0.0, 0.0])
                            a[0] += np.mean(errs) / B_DRAWS
                            a[1] += np.mean(tgts) / B_DRAWS
    out = {}
    for m in MS:
        sub = {k: v for k, v in res.items() if k[3] == m}
        ranked = sorted(sub, key=lambda k: sub[k][0])
        out[m] = [dict(subset=k[0], sigma=k[1], est=k[2],
                       ref=round(sub[k][0], 4), tgt=round(sub[k][1], 4))
                  for k in ranked[:8]]
        b = ranked[0]
        print(f'm={m}: best {b[0]} sigma={b[1]} {b[2]} '
              f'ref={sub[b][0]:.4f} target={sub[b][1]:.4f}')
    json.dump(out, open('figures/tb2_tune3.json', 'w'), indent=2)
    print('wrote figures/tb2_tune3.json')


if __name__ == '__main__':
    main()
