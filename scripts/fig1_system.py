"""Figure 1 candidate (per HH: per-trace qubric scatter 'looks like
noise'): SYSTEM-level geometry, one point per system (107), the object
the estimator actually consumes.

Coordinates: classical MDS (2D) of the PKPS distance matrix over the full
q100 panel at the working bandwidth (sigma^2 = med/4), per row:
  row 1  raw trace embeddings (median-centered + L2, as in the raw
         baseline -- its best-case geometry)
  row 2  qubric embeddings (consensus-centered + L2)
Columns colored by: underlying LLM, harness, benchmark score y.

Writes figures/fig1_system.png.
"""
import json
import re
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import load_panel  # noqa: E402
from pillars import harness_tag  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

SIG = 4  # sigma^2 = med / SIG, the pooled working bandwidth at full panel


def pkps_full_D(X, V):
    M, Q, _ = X.shape
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    KQ = np.exp(-D2q / (2 * np.median(D2q) / SIG))
    W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
    A = (X.reshape(M, -1) @ W.reshape(M, -1).T) / KQ.sum()
    d2 = np.diag(A)[:, None] + np.diag(A)[None] - 2 * A
    return np.sqrt(np.maximum(d2, 0))


def cmds2(D):
    n = len(D)
    J = np.eye(n) - 1 / n
    Bm = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1][:2]
    return v[:, idx] * np.sqrt(np.maximum(w[idx], 0))


def main():
    systems, q100, y, B, _ = load_panel(panel='data/judge/q100.json')
    M, Q = len(systems), len(q100)
    labels = json.load(open('data/leaderboard/verified_labels.json'))
    model = []
    for s in systems:
        m = re.search(r'^\s+model_display:\s*(.*)$',
                      labels[s].get('metadata_yaml', ''), re.M)
        model.append(m.group(1).strip() if m else None)
    harness = [harness_tag(s) for s in systems]

    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]

    raw = np.load('data/judge/q100_raw_emb_openai_small.npz')['HT'] \
        .reshape(M, Q, -1)
    raw = raw - np.median(raw, axis=0, keepdims=True)
    raw /= np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-9)
    qub = consensus_center(
        np.load('data/judge/q100_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), M)).reshape(M, Q, -1)

    Zr = cmds2(pkps_full_D(raw.astype(np.float32), V))
    Zq = cmds2(pkps_full_D(qub.astype(np.float32), V))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from collections import Counter

    PAL = ['#2c7fb8', '#e08214', '#c51b7d', '#41ab5d', '#6a51a3',
           '#a63603', '#01665e', '#bdb200']

    def cat_colors(vals):
        cnt = Counter(v for v in vals if v)
        top = [k for k, _ in cnt.most_common(8)]
        cmap = {k: PAL[i] for i, k in enumerate(top)}
        return np.array([cmap.get(v, '#cccccc') for v in vals]), cmap

    mcols, mmap = cat_colors(model)
    hcols, hmap = cat_colors(harness)

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7))
    rows = [('Raw trace embedding', Zr), ('qubric embedding', Zq)]
    for r, (rname, Z) in enumerate(rows):
        specs = [('Colored by model (LLM)', dict(c=mcols)),
                 ('Colored by harness', dict(c=hcols)),
                 ('Colored by benchmark score $y$',
                  dict(c=y, cmap='viridis', vmin=y.min(), vmax=y.max()))]
        for c, (cname, kw) in enumerate(specs):
            ax = axes[r, c]
            sc = ax.scatter(Z[:, 0], Z[:, 1], s=42, alpha=.9,
                            edgecolors='white', lw=.5, **kw)
            if r == 0:
                ax.set_title(cname, fontsize=10.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('.8')
        axes[r, 0].set_ylabel(rname + '\n(PKPS, classical MDS)', fontsize=10)
    cb = fig.colorbar(sc, ax=axes[:, 2], shrink=.6, pad=.02)
    cb.set_label('resolve rate $y$', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    def handles(cmap, extra):
        h = [mlines.Line2D([], [], marker='o', ls='', ms=6, color=v,
                           label=k) for k, v in cmap.items()]
        h.append(mlines.Line2D([], [], marker='o', ls='', ms=6,
                               color='#cccccc', label=extra))
        return h
    fig.legend(handles=handles(mmap, 'other'), loc='lower left',
               bbox_to_anchor=(0.03, -0.005), ncol=3, fontsize=7,
               frameon=False, title='model (LLM)', title_fontsize=7.5)
    fig.legend(handles=handles(hmap, 'untagged'), loc='lower left',
               bbox_to_anchor=(0.42, -0.005), ncol=3, fontsize=7,
               frameon=False, title='harness', title_fontsize=7.5)
    fig.tight_layout(rect=(0, 0.075, 0.98, 1))
    fig.savefig('figures/fig1_system.png', dpi=200)
    print('wrote figures/fig1_system.png')


if __name__ == '__main__':
    main()
