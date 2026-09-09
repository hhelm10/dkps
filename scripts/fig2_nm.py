"""Figure 2 (per HH 2026-09-09): effect of the number of reference systems
n and the number of scored tasks m on the system-level qubric embedding.

Grid: rows n in {25, 50, 107} (nested random system subsets, seeded),
cols m in {3, 10, 30, 100} (nested random task subsets). Each panel is
the 2D classical MDS of the PKPS distance matrix over the m tasks for the
n systems, colored by full-benchmark score y. Panels are Procrustes-
aligned (rotation/reflection + scale) to the full (107, 100) configuration
restricted to the panel's systems, so convergence is visible left-to-right
and top-to-bottom. Agentic analogue of Fig. 1 in arXiv:2605.07096.

Writes figures/fig2_nm.png.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import load_panel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

NS = (25, 50, 107)
MS = (3, 10, 30, 100)
SIG = 4  # sigma^2 = med/SIG at the working bandwidth


def cmds2(D):
    n = len(D)
    J = np.eye(n) - 1 / n
    Bm = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1][:2]
    return v[:, idx] * np.sqrt(np.maximum(w[idx], 0))


def procrustes(Z, ref):
    """Rotate/reflect+scale Z (centered) onto ref (centered)."""
    Zc = Z - Z.mean(0)
    Rc = ref - ref.mean(0)
    U, S, Vt = np.linalg.svd(Zc.T @ Rc)
    R = U @ Vt
    s = S.sum() / (Zc ** 2).sum()
    return s * Zc @ R


def main():
    systems, q100, y, B, _ = load_panel(panel='data/judge/q100.json')
    M, Q = len(systems), len(q100)
    X = consensus_center(
        np.load('data/judge/q100_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), M)).reshape(M, Q, -1).astype(np.float32)
    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    KQfull = np.exp(-D2q / (2 * np.median(D2q) / SIG))

    rng = np.random.default_rng(7)
    sys_order = rng.permutation(M)          # nested system subsets
    task_order = rng.permutation(Q)         # nested task subsets

    def pkps_mds(sub, cols):
        Xs = X[np.ix_(sub, cols)]
        KQ = KQfull[np.ix_(cols, cols)]
        W = np.einsum('qp,jpd->jqd', KQ, Xs, optimize=True)
        A = (Xs.reshape(len(sub), -1) @ W.reshape(len(sub), -1).T) / KQ.sum()
        d2 = np.diag(A)[:, None] + np.diag(A)[None] - 2 * A
        return cmds2(np.sqrt(np.maximum(d2, 0)))

    ref_full = pkps_mds(np.arange(M), np.arange(Q))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(NS), len(MS),
                             figsize=(3.0 * len(MS), 2.9 * len(NS)))
    for r, n in enumerate(NS):
        sub = np.sort(sys_order[:n])
        for c, m in enumerate(MS):
            cols = np.sort(task_order[:m])
            Z = pkps_mds(sub, cols)
            Z = procrustes(Z, ref_full[sub])
            ax = axes[r, c]
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=y[sub], cmap='viridis',
                            vmin=y.min(), vmax=y.max(), s=34, alpha=.9,
                            edgecolors='white', lw=.4)
            if r == 0:
                ax.set_title(f'$m = {m}$ tasks', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('.8')
        axes[r, 0].set_ylabel(f'$n = {n}$ systems', fontsize=11)
    cb = fig.colorbar(sc, ax=axes, shrink=.55, pad=.015)
    cb.set_label('benchmark score $y$ (resolve rate)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle('System-level qubric embeddings (PKPS, classical MDS)',
                 fontsize=12)
    fig.savefig('figures/fig2_nm.png', dpi=200, bbox_inches='tight')
    print('wrote figures/fig2_nm.png')


if __name__ == '__main__':
    main()
