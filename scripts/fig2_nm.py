"""Figure 2 (per HH 2026-09-09, rev 2): system-level embeddings, raw vs
qubric, as (n, m) grows.

  2 x 3 grid, colored by full-benchmark score y:
    row 1  raw trace embeddings (median-centered + L2)
    row 2  qubric embeddings (consensus-centered + L2)
    col 1  small n, small m   (n=25,  m=3)
    col 2  small n, large m   (n=25,  m=100)
    col 3  large n, large m   (n=107, m=100)

Each panel: classical MDS of the PKPS distance matrix over the m tasks
for the n systems (sigma^2 = med/4). Within a row, panels are Procrustes-
aligned to that row's full (107, 100) configuration restricted to the
panel's systems. System/task subsets are nested and seeded.

Writes figures/fig2_nm.png.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import load_panel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

CELLS = [(20, 3, 'fewer ref. systems, fewer probes'),
         (20, 100, 'fewer ref. systems, more probes'),
         (100, 100, 'more ref. systems, more probes')]
SIG = 4


def cmds2(D):
    n = len(D)
    J = np.eye(n) - 1 / n
    Bm = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1][:2]
    return v[:, idx] * np.sqrt(np.maximum(w[idx], 0))


def procrustes(Z, ref):
    Zc = Z - Z.mean(0)
    Rc = ref - ref.mean(0)
    U, S, Vt = np.linalg.svd(Zc.T @ Rc)
    s = S.sum() / (Zc ** 2).sum()
    return s * Zc @ (U @ Vt)


def main():
    systems, q100, y, B, _ = load_panel(panel='data/judge/q100.json')
    M, Q = len(systems), len(q100)
    raw = np.load('data/judge/q100_raw_emb_openai_small.npz')['HT'] \
        .reshape(M, Q, -1)
    raw = raw - np.median(raw, axis=0, keepdims=True)
    raw /= np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-9)
    qub = consensus_center(
        np.load('data/judge/q100_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), M)).reshape(M, Q, -1)
    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    KQfull = np.exp(-D2q / (2 * np.median(D2q) / SIG))

    rng = np.random.default_rng(7)
    sys_order = rng.permutation(M)
    task_order = rng.permutation(Q)

    def pkps_D(X, sub, cols):
        Xs = X[np.ix_(sub, cols)].astype(np.float32)
        KQ = KQfull[np.ix_(cols, cols)]
        W = np.einsum('qp,jpd->jqd', KQ, Xs, optimize=True)
        A = (Xs.reshape(len(sub), -1) @ W.reshape(len(sub), -1).T) / KQ.sum()
        d2 = np.diag(A)[:, None] + np.diag(A)[None] - 2 * A
        return np.sqrt(np.maximum(d2, 0))

    def loo_mae(D, ys, k=3):
        Dm = D.copy()
        np.fill_diagonal(Dm, np.inf)
        nn = np.argsort(Dm, 1)[:, :k]
        w = 1 / (np.take_along_axis(Dm, nn, 1) + 1e-12)
        pred = (w * ys[nn]).sum(1) / w.sum(1)
        return float(np.abs(pred - ys).mean())

    import matplotlib
    matplotlib.use('Agg')
    import hv_style
    hv_style.apply()
    import matplotlib.pyplot as plt

    SZ = hv_style.SIZES
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=float(y.mean()),
                                          vmin=float(y.min()),
                                          vmax=float(y.max()))
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.7))
    fig.subplots_adjust(left=.085, right=.90, top=.80, bottom=.06,
                        wspace=.06, hspace=.10)
    # row references: qubric's full configuration anchors the figure; the
    # raw row's reference is rotated onto it so the rows are comparable
    ref_qub = cmds2(pkps_D(qub, np.arange(M), np.arange(Q)))
    ref_raw = procrustes(cmds2(pkps_D(raw, np.arange(M), np.arange(Q))),
                         ref_qub)
    for r, (rname, X, ref_full) in enumerate([('raw', raw, ref_raw),
                                              ('qubric', qub, ref_qub)]):
        for c, (n, m, cname) in enumerate(CELLS):
            sub = np.sort(sys_order[:n])
            cols = np.sort(task_order[:m])
            D = pkps_D(X, sub, cols)
            Z = procrustes(cmds2(D), ref_full[sub])
            ax = axes[r, c]
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=y[sub],
                            cmap=hv_style.CMAP_DIV, norm=norm, s=26,
                            alpha=.95, edgecolors=hv_style.EDGE, lw=.5)
            ax.text(.035, .96, f'LOO MAE {loo_mae(D, y[sub]):.3f}',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=SZ['annot'], color=hv_style.INK,
                    bbox=dict(boxstyle='round,pad=0.35', fc=hv_style.WASH,
                              ec=hv_style.SPINE, lw=.8, alpha=.95))
            if r == 0:
                ax.set_title(cname.replace(', ', ',\n')
                             + f'\n($n={n}$, $m={m}$)',
                             fontsize=SZ['subtitle'], pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_color(hv_style.SPINE)
            if r == 1:
                ax.set_xlabel('PKPS 1', fontsize=SZ['tick'], labelpad=2)
        axes[r, 0].set_ylabel('PKPS 2', fontsize=SZ['tick'], labelpad=2)
    for r, rname in enumerate(('raw', 'qubric')):
        fig.text(.022, .62 - .385 * r, rname, rotation=90,
                 fontsize=SZ['label'], color=hv_style.INK,
                 va='center', ha='center')
    cb = fig.colorbar(sc, ax=axes, shrink=.85, pad=.025, aspect=28)
    cb.set_label('benchmark score $y$', fontsize=SZ['tick'])
    cb.ax.tick_params(labelsize=SZ['annot'])
    cb.outline.set_edgecolor(hv_style.SPINE)
    fig.savefig('figures/fig2_nm.png', dpi=250, bbox_inches='tight',
                pad_inches=0.02)
    print('wrote figures/fig2_nm.png')


if __name__ == '__main__':
    main()
