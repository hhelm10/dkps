"""Figure 6: rubric field-count sensitivity under the real pipeline.

Bank of 32 fields, r in {1..32}, <=200 seeded subsets per r (additive
Gram terms make the sweep tractable); m in {1,5,20} panels; reference
libraries n=107 (solid) and n=20 (dashed); mean over subsets (thick)
and best subset (thin), +/-1 SEM.

`main()` recomputes project/trubric/data/rubric_sens_paper.json from
the bank-32 embedding caches (hours); `render(json)` is cheap.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from trubric_common import DATA, ART, save_artifact, save_text_artifact  # noqa

import json
import sys

import numpy as np

from rubric16 import bank, panel  # noqa: E402
from pillars import vendor_tag  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

FIELDS, SECTIONS, ROOT, PREFIX = bank(32)
F = 32
RS = (1, 2, 4, 6, 8, 12, 16, 24, 32)
MSP = (1, 5, 20)
NS = (107, 20)          # reference-library sizes; 20 = seeded subsample
N_SUB = 200
SIGS = (2, 4, 8)
OUT_JSON = 'project/trubric/data/rubric_sens_paper.json'


def main():
    labels, systems, q20 = panel()
    M, Q = len(systems), 20
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    fam = [vendor_tag(labels, s) for s in systems]
    allowed = np.array([[j != i and not (fam[i] and fam[j] == fam[i])
                         for j in range(M)] for i in range(M)])
    lib20 = np.zeros(M, bool)
    lib20[np.random.default_rng(11).choice(M, 20, replace=False)] = True
    allowed_n = {107: allowed, 20: allowed & lib20[None, :]}
    man = json.load(open(f'{ROOT}/manifest.json'))
    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q20]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)
    rng = np.random.default_rng(7)
    draws = {1: [np.array([q]) for q in range(Q)],
             5: [rng.choice(Q, 5, replace=False) for _ in range(30)],
             20: [np.arange(Q)]}

    def precompute(arm):
        """T[(s_, m, di)] = per-field Gram terms:
        A_tr_f (M,M), Att_f (M,), Arr_f (M,) stacked over f."""
        X = np.load(f'data/judge/{PREFIX}_emb_{arm}.npz')['X']
        Xc = consensus_center(X, np.tile(np.arange(Q), M)) \
            .reshape(M, Q, F, -1).astype(np.float32)
        T = {}
        for s_ in SIGS:
            KQ = np.exp(-D2q / (2 * med / s_)).astype(np.float32)
            for f in range(F):
                Xf = Xc[:, :, f, :]
                W = np.einsum('qp,jpd->jqd', KQ, Xf, optimize=True)
                Arr_f = np.einsum('jqd,jqd->j', Xf, W) / KQ.sum()
                for m in MSP:
                    for di, cols in enumerate(draws[m]):
                        Xi = Xf[:, cols].reshape(M, -1)
                        A_tr = (Xi @ W[:, cols].reshape(M, -1).T) \
                            / KQ[cols].sum()
                        KQc = KQ[np.ix_(cols, cols)]
                        Wc = np.einsum('qp,jpd->jqd', KQc, Xf[:, cols],
                                       optimize=True)
                        Att = np.einsum('jqd,jqd->j', Xf[:, cols], Wc) \
                            / KQc.sum()
                        key = (s_, m, di)
                        if key not in T:
                            T[key] = (np.zeros((F, M, M), np.float32),
                                      np.zeros((F, M), np.float32),
                                      np.zeros((F, M), np.float32))
                        T[key][0][f] = A_tr
                        T[key][1][f] = Att
                        T[key][2][f] = Arr_f
        return T

    def estimators(D, allow):
        cand = {}
        # kNN k=5
        errs, tgt = [], np.zeros(M)
        for i in range(M):
            refs = np.where(allow[i])[0]
            Dref = D[:, refs].copy()
            for r_, j in enumerate(refs):
                Dref[j, r_] = np.inf
            Dref[i] = D[i, refs]
            nn = np.argsort(Dref, 1)[:, :5]
            w = 1 / (np.take_along_axis(Dref, nn, 1) + 1e-12)
            pred = (w * y[refs][nn]).sum(1) / w.sum(1)
            errs.append(np.abs(pred[refs] - y[refs]).mean())
            tgt[i] = pred[i]
        cand['knn5'] = (float(np.mean(errs)), tgt)
        # ridge on MDS coords, LOO-honest
        n_ = len(D)
        J = np.eye(n_) - 1 / n_
        Bm = -0.5 * J @ (D ** 2) @ J
        ew, ev = np.linalg.eigh(Bm)
        rdim = 16 if allow.sum(1).min() > 20 else 8
        order = np.argsort(ew)[::-1][:rdim]
        Z = ev[:, order] * np.sqrt(np.maximum(ew[order], 1e-12))
        errs, tgt = [], np.zeros(M)
        for i in range(M):
            refs = np.where(allow[i])[0]
            Zr, yr = Z[refs], y[refs]
            G = Zr.T @ Zr + 0.1 * np.eye(rdim)
            Gi = np.linalg.inv(G)
            beta = Gi @ (Zr.T @ (yr - yr.mean()))
            tgt[i] = float(np.clip(Z[i] @ beta + yr.mean(), 0, 1))
            h = np.einsum('jr,rs,js->j', Zr, Gi, Zr)
            pr = Zr @ beta + yr.mean()
            loo = yr - (yr - pr) / np.maximum(1 - h, 1e-6)
            errs.append(np.abs(np.clip(loo, 0, 1) - yr).mean())
        cand['ridge'] = (float(np.mean(errs)), tgt)
        return cand

    results = {}
    for arm in ('generic', 'qspec'):
        T = precompute(arm)
        print(f'{arm}: gram terms ready', flush=True)
        res = {}
        controls = {'orig6': list(range(6)), 'all16': list(range(16)),
                    'all32': list(range(32))}
        for n_refs in NS:
          allow = allowed_n[n_refs]
          for m in MSP:
            subsets = {r: [tuple(s) for s in man['masks'][str(r)][:N_SUB]]
                       for r in RS}
            per_r = {}
            for r in RS:
                maes = []
                for S in subsets[r]:
                    Sarr = np.asarray(S)
                    tgt_err = np.zeros(M)
                    for di in range(len(draws[m])):
                        cand_all = {}
                        for s_ in SIGS:
                            A_tr, Att, Arr = T[(s_, m, di)]
                            Ats = A_tr[Sarr].mean(0)
                            Atts = Att[Sarr].mean(0)
                            Arrs = Arr[Sarr].mean(0)
                            D = np.sqrt(np.maximum(
                                Atts[:, None] + Arrs[None] - 2 * Ats, 0))
                            for nm, v in estimators(D, allow).items():
                                cand_all[(s_, nm)] = v
                        tgt = min(cand_all.values(),
                                  key=lambda v: v[0])[1]
                        tgt_err += np.abs(tgt - y) / len(draws[m])
                    maes.append(float(tgt_err.mean()))
                maes = np.array(maes)
                per_r[str(r)] = dict(
                    mean=float(maes.mean()),
                    sem=float(maes.std(ddof=1) / np.sqrt(len(maes)))
                    if len(maes) > 1 else 0.0,
                    p10=float(np.percentile(maes, 10)),
                    p90=float(np.percentile(maes, 90)),
                    min=float(maes.min()), max=float(maes.max()),
                    n=len(maes))
            for cname, S in controls.items():
                Sarr = np.asarray(S)
                tgt_err = np.zeros(M)
                for di in range(len(draws[m])):
                    cand_all = {}
                    for s_ in SIGS:
                        A_tr, Att, Arr = T[(s_, m, di)]
                        D = np.sqrt(np.maximum(
                            Att[Sarr].mean(0)[:, None]
                            + Arr[Sarr].mean(0)[None]
                            - 2 * A_tr[Sarr].mean(0), 0))
                        for nm, v in estimators(D, allow).items():
                            cand_all[(s_, nm)] = v
                    tgt = min(cand_all.values(), key=lambda v: v[0])[1]
                    tgt_err += np.abs(tgt - y) / len(draws[m])
                per_r[cname] = float(tgt_err.mean())
            res.setdefault(str(n_refs), {})[str(m)] = per_r
            print(arm, f'n={n_refs} m={m}',
                  {r: round(per_r[str(r)]['mean'], 4) for r in RS},
                  'orig6', round(per_r['orig6'], 4), flush=True)
        results[arm] = res
        json.dump(results, open(OUT_JSON, 'w'), indent=1)
    render(results)


def render(results):
    import matplotlib
    matplotlib.use('Agg')
    import hv_style
    hv_style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    SZ = hv_style.SIZES
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.7), sharex=True,
                             sharey=True)
    for ax, m0 in zip(axes, ('1', '5', '20')):
        for arm, role in (('generic', 'generic'), ('qspec', 'focus')):
            st = hv_style.ROLES[role]
            old_style = '107' not in results[arm]   # pre-n-sweep cache
            variants = ((None, '-'),) if old_style \
                else (('107', '-'), ('20', '--'))
            for n_refs, ls in variants:
                res = (results[arm][m0] if old_style
                       else results[arm].get(n_refs, {}).get(m0))
                if res is None:
                    continue
                rs = np.array(RS)
                mean = np.array([res[str(r)]['mean'] for r in RS])
                sem = np.array([res[str(r)].get('sem', 0) for r in RS])
                best = np.array([res[str(r)]['min'] for r in RS])
                ax.fill_between(rs, mean - sem, mean + sem,
                                color=st['color'], alpha=.18, lw=0)
                ax.plot(rs, mean, color=st['color'], lw=2.8, ls=ls,
                        marker='o', ms=4)
                ax.plot(rs, best, color=st['color'], lw=1.3, ls=ls,
                        alpha=.85)
        ax.set_title(f'$m = {m0}$', fontsize=SZ['label'],
                     color=hv_style.INK_TITLE)
        ax.set_xlabel('number of rubric fields $r$',
                      fontsize=SZ['subtitle'])
        ax.set_xticks(RS)
        ax.tick_params(labelsize=SZ['tick'])
    axes[0].set_yticks((0.07, 0.08, 0.09, 0.10, 0.11))
    axes[0].set_ylabel('MAE$(\\hat{y}, y)$',
                       fontsize=SZ['subtitle'])
    ink = hv_style.INK
    legs = (
        (.17, [Line2D([], [], color=hv_style.ROLES['generic']
                          ['color'], lw=2.8, label='generic'),
                   Line2D([], [], color=hv_style.ROLES['focus']['color'],
                          lw=2.8, label='trubric')]),
        (.50, [Line2D([], [], color=ink, lw=2.8, marker='o', ms=4,
                          label='average over subsets'),
                   Line2D([], [], color=ink, lw=1.3, alpha=.85,
                          label='best subset')]),
        (.83, [Line2D([], [], color=ink, lw=2.2, ls='-',
                          label='$n = 107$'),
                   Line2D([], [], color=ink, lw=2.2, ls='--',
                          label='$n = 20$')]),
    )
    for xc, handles in legs:
        fig.legend(handles=handles, loc='upper center',
                   bbox_to_anchor=(xc, 0.02), ncol=3,
                   fontsize=SZ['legend'] - 1, handlelength=2.6,
                   frameon=False, columnspacing=1.6)
    fig.tight_layout()
    save_artifact(fig, 'trubric_figure6_sensitivity', dpi=200,
                  pad=0.03)


if __name__ == '__main__':
    main()
