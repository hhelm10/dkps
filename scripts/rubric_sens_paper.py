"""Field-count sensitivity under the REAL paper pipeline (per HH: the
3-NN sweep made generic look >= qubric; the paper stack is where they
differ, so the figure must use it).

Pipeline: leave-one-FAMILY-out, consensus centering, PKPS kernel over
q20 qvecs, per-draw pooled CV over sigma x {kNN-k5, ridge(16,0.1) with
LOO-honest refs}. Additivity of the kernel Gram terms over fields makes
the subset sweep tractable: per-(field, sigma, draw) M x M terms are
precomputed once; a subset's distance matrix is a slice-sum.

Scope: bank 32, r in {1,2,4,6,8,12,16,24,32}, <=200 subsets per r
(first 200 of the frozen manifest masks; shared across arms),
m in {1, 5, 20} (20 singletons / 30 draws / full panel).
Writes figures/rubric_sens_paper.json + figures/fig_sensitivity_paper.png.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from rubric16 import bank, panel  # noqa: E402
from pillars import vendor_tag  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

FIELDS, SECTIONS, ROOT, PREFIX = bank(32)
F = 32
RS = (1, 2, 4, 6, 8, 12, 16, 24, 32)
MSP = (1, 5, 20)
N_SUB = 200
SIGS = (2, 4, 8)
OUT_JSON = 'figures/rubric_sens_paper.json'
OUT_PNG = 'figures/fig_sensitivity_paper.png'


def main():
    labels, systems, q20 = panel()
    M, Q = len(systems), 20
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    fam = [vendor_tag(labels, s) for s in systems]
    allowed = np.array([[j != i and not (fam[i] and fam[j] == fam[i])
                         for j in range(M)] for i in range(M)])
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

    def estimators(D):
        cand = {}
        # kNN k=5
        errs, tgt = [], np.zeros(M)
        for i in range(M):
            refs = np.where(allowed[i])[0]
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
        order = np.argsort(ew)[::-1][:16]
        Z = ev[:, order] * np.sqrt(np.maximum(ew[order], 1e-12))
        errs, tgt = [], np.zeros(M)
        for i in range(M):
            refs = np.where(allowed[i])[0]
            Zr, yr = Z[refs], y[refs]
            G = Zr.T @ Zr + 0.1 * np.eye(16)
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
                            for nm, v in estimators(D).items():
                                cand_all[(s_, nm)] = v
                        tgt = min(cand_all.values(),
                                  key=lambda v: v[0])[1]
                        tgt_err += np.abs(tgt - y) / len(draws[m])
                    maes.append(float(tgt_err.mean()))
                maes = np.array(maes)
                per_r[str(r)] = dict(
                    mean=float(maes.mean()),
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
                        for nm, v in estimators(D).items():
                            cand_all[(s_, nm)] = v
                    tgt = min(cand_all.values(), key=lambda v: v[0])[1]
                    tgt_err += np.abs(tgt - y) / len(draws[m])
                per_r[cname] = float(tgt_err.mean())
            res[str(m)] = per_r
            print(arm, f'm={m}',
                  {r: round(per_r[str(r)]['mean'], 4) for r in RS},
                  'orig6', round(per_r['orig6'], 4),
                  'all16', round(per_r['all16'], 4),
                  'all32', round(per_r['all32'], 4), flush=True)
        results[arm] = res
        json.dump(results, open(OUT_JSON, 'w'), indent=1)
    render(results)


def render(results):
    import matplotlib
    matplotlib.use('Agg')
    import hv_style
    hv_style.apply()
    import matplotlib.pyplot as plt

    SZ = hv_style.SIZES
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), sharex=True)
    for ax, m0 in zip(axes, ('1', '5', '20')):
        for arm, nice, role in (('generic', 'generic rubric',
                                 'comparator'),
                                ('qspec', 'qubric', 'focus')):
            res = results[arm][m0]
            rs = np.array(RS)
            mean = [res[str(r)]['mean'] for r in RS]
            p10 = [res[str(r)]['p10'] for r in RS]
            p90 = [res[str(r)]['p90'] for r in RS]
            st = hv_style.ROLES[role]
            ax.fill_between(rs, p10, p90, color=st['color'], alpha=.16,
                            lw=0)
            ax.plot(rs, mean, color=st['color'], lw=2.8, marker='o',
                    ms=4, label=nice)
            ax.scatter([6], [res['orig6']], marker='D', s=70,
                       color=st['color'], zorder=5)
        ax.set_title(f'$m = {m0}$', fontsize=SZ['label'],
                     color=hv_style.INK_TITLE)
        ax.set_xlabel('number of rubric fields $r$',
                      fontsize=SZ['subtitle'])
        ax.set_xticks(RS)
        ax.tick_params(labelsize=SZ['tick'])
    axes[0].scatter([], [], marker='D', s=70, color=hv_style.INK_MUTE,
                    label='original six fields')
    axes[0].set_ylabel('MAE$(\\hat{y}, y)$ (paper pipeline)',
                       fontsize=SZ['subtitle'])
    axes[0].legend(fontsize=SZ['legend'] - 1, handlelength=2.6)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches='tight', pad_inches=0.03)
    print('wrote', OUT_PNG)


if __name__ == '__main__':
    main()
