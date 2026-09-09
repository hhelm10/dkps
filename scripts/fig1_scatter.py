"""Figure 1 (per HH 2026-09-09): 2x4 PCA scatter of per-(system,task)
trace embeddings on the SWE-bench q100 panel.

  row 1  raw off-the-shelf embeddings (head+tail 8K-token slices,
         text-embedding-3-small, L2; no centering -- the default a
         practitioner would use)
  row 2  qubric embeddings (query-specific extraction, consensus-centered
         + L2 -- the pipeline output)
  cols   PC1 v PC2 colored by: model (LLM), harness, task, correctness

One PCA per row (all 10,700 traces); the four columns recolor the same
coordinates. Writes figures/fig1_scatter.png.
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

TOPK = 8


def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]
    var = (S ** 2) / (S ** 2).sum()
    return Z, var[:2]


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

    raw = np.load('data/judge/q100_raw_emb_openai_small.npz')['HT'] \
        .reshape(M * Q, -1).astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-9)
    qub = np.load('data/judge/q100_emb_openai_small.npz')['X'] \
        .astype(np.float32)
    qub = consensus_center(qub, np.tile(np.arange(Q), M))
    qub /= np.maximum(np.linalg.norm(qub, axis=1, keepdims=True), 1e-9)

    Zr, vr = pca2(raw)
    Zq, vq = pca2(qub)
    print('explained var: raw', vr.round(3), 'qubric', vq.round(3))

    sys_idx = np.repeat(np.arange(M), Q)
    task_idx = np.tile(np.arange(Q), M)
    corr = B[sys_idx, task_idx]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # fixed categorical palette (colorblind-validated ordering)
    PAL = ['#2c7fb8', '#e08214', '#c51b7d', '#41ab5d', '#6a51a3',
           '#a63603', '#01665e', '#bdb200']

    def cat_colors(vals):
        from collections import Counter
        cnt = Counter(v for v in vals if v)
        top = [k for k, _ in cnt.most_common(TOPK)]
        cmap = {k: PAL[i] for i, k in enumerate(top)}
        cols = np.array([cmap.get(v, '#cccccc') for v in vals])
        return cols, top, cmap

    mcols, mtop, mmap = cat_colors([model[i] for i in sys_idx])
    hcols, htop, hmap = cat_colors([harness[i] for i in sys_idx])
    tcols = plt.get_cmap('hsv')(task_idx / Q)
    ccols = np.where(corr > 0, '#2166ac', '#d95f02')

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8))
    rows = [('Raw trace embedding', Zr, vr), ('qubric embedding', Zq, vq)]
    cols_spec = [('Colored by model (LLM)', lambda: mcols),
                 ('Colored by harness', lambda: hcols),
                 ('Colored by task', lambda: tcols),
                 ('Colored by correctness', lambda: ccols)]
    for r, (rname, Z, var) in enumerate(rows):
        order = np.random.default_rng(0).permutation(len(Z))
        for c, (cname, fcol) in enumerate(cols_spec):
            ax = axes[r, c]
            ax.scatter(Z[order, 0], Z[order, 1], s=2.5,
                       c=np.asarray(fcol())[order], alpha=.55, lw=0,
                       rasterized=True)
            if r == 0:
                ax.set_title(cname, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('.8')
        axes[r, 0].set_ylabel(f'{rname}\nPC2 ({var[1]:.0%})', fontsize=9)
        for c in range(4):
            axes[1, c].set_xlabel('PC1', fontsize=8)
    axes[0, 0].set_xlabel('')
    # compact legends under the grid
    import matplotlib.lines as mlines
    def handles(cmap, extra=None):
        h = [mlines.Line2D([], [], marker='o', ls='', ms=5, color=v,
                           label=k) for k, v in cmap.items()]
        if extra:
            h.append(mlines.Line2D([], [], marker='o', ls='', ms=5,
                                   color='#cccccc', label=extra))
        return h
    fig.legend(handles=handles(mmap, 'other'), loc='lower left',
               bbox_to_anchor=(0.01, -0.01), ncol=3, fontsize=6.5,
               frameon=False, title='model (LLM)', title_fontsize=7)
    fig.legend(handles=handles(hmap, 'untagged'), loc='lower left',
               bbox_to_anchor=(0.47, -0.01), ncol=3, fontsize=6.5,
               frameon=False, title='harness', title_fontsize=7)
    fig.legend(handles=[mlines.Line2D([], [], marker='o', ls='', ms=5,
                                      color='#2166ac', label='resolved'),
                        mlines.Line2D([], [], marker='o', ls='', ms=5,
                                      color='#d95f02', label='unresolved')],
               loc='lower left', bbox_to_anchor=(0.86, -0.01), ncol=1,
               fontsize=6.5, frameon=False, title='correctness',
               title_fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig('figures/fig1_scatter.png', dpi=200)
    print('wrote figures/fig1_scatter.png')


if __name__ == '__main__':
    main()
