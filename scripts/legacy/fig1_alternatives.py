"""Two candidate replacements for the 2x4 PCA scatter (HH: 'not very
interpretable ... rethink').

A. Dumbbell chart of nearest-neighbor retrieval rates (raw -> qubric)
   per axis, with chance ticks. Quantitative headline; numbers from
   figures/radar_all_data_v2.json (canonical embedder text-embedding-3-small).
   -> figures/fig1A_dumbbell.png

B. Single-task scatter: one task's 107 system traces, PCA per row
   (raw / qubric), colored by harness and by correctness. Visual
   mechanism, no overplotting. -> figures/fig1B_onetask.png
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fig_a():
    d = json.load(open('figures/radar_all_data_v2.json'))
    p = d['panels']['text-embedding-3-small']
    ch = d['chance']
    rows = [  # (key, label, group)
        ('prov', 'same submission', 'confounder'),
        ('scaf', 'same harness', 'confounder'),
        ('llm', 'same underlying LLM', 'confounder'),
        ('task', 'same task', 'content'),
        ('outcome', 'same outcome', 'content'),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ys = np.arange(len(rows))[::-1] * 1.0
    ys[3:] -= 0.55  # gap between groups
    for (k, lab, grp), yy in zip(rows, ys):
        r, q, c = p['raw'][k], p['qubric'][k], ch[k]
        ax.plot([r, q], [yy, yy], color='.75', lw=1.6, zorder=1)
        ax.scatter([r], [yy], s=55, color='#555555', zorder=3,
                   label='raw trace' if k == 'prov' else None)
        ax.scatter([q], [yy], s=55, color='#c51b7d', zorder=3,
                   label='qubric' if k == 'prov' else None)
        ax.plot([c, c], [yy - .22, yy + .22], color='.3', lw=1.2, ls=':',
                zorder=2)
        ax.annotate('', xy=(q, yy + 0.16), xytext=(r, yy + 0.16),
                    arrowprops=dict(arrowstyle='->', color='.55', lw=1))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows], fontsize=9)
    ax.text(1.01, ys[1] + 0.5, 'identity\n(confounders)', fontsize=8,
            color='.4', va='center', transform=ax.get_yaxis_transform())
    ax.text(1.01, (ys[3] + ys[4]) / 2, 'content', fontsize=8, color='.4',
            va='center', transform=ax.get_yaxis_transform())
    ax.plot([], [], color='.3', lw=1.2, ls=':', label='chance')
    ax.set_xlim(0, 1)
    ax.set_xlabel('P(nearest neighbor shares attribute)', fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.grid(True, axis='x', color='.93', lw=.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig('figures/fig1A_dumbbell.png', dpi=200)
    print('wrote figures/fig1A_dumbbell.png')


def fig_b():
    import re
    from outcome_baselines import load_panel
    from pillars import harness_tag
    from dkps.traces.qubric import consensus_center

    systems, q100, y, B, _ = load_panel(panel='data/judge/q100.json')
    M, Q = len(systems), len(q100)
    harness = [harness_tag(s) for s in systems]
    ntag = np.array([h is not None for h in harness])

    raw = np.load('data/judge/q100_raw_emb_openai_small.npz')['HT'] \
        .reshape(M, Q, -1).astype(np.float32)
    qub = consensus_center(
        np.load('data/judge/q100_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), M)).reshape(M, Q, -1).astype(np.float32)

    # pick the task with balanced correctness and many tagged harnesses
    bal = -np.abs(B.mean(0) - .5)
    t = int(np.argmax(bal))
    print('task:', q100[t], 'resolve rate', round(float(B[:, t].mean()), 2))

    def pca2(X):
        Xc = X - X.mean(0, keepdims=True)
        Xc /= np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :2] * S[:2]

    Zr, Zq = pca2(raw[:, t]), pca2(qub[:, t])
    PAL = ['#2c7fb8', '#e08214', '#c51b7d', '#41ab5d', '#6a51a3',
           '#a63603', '#01665e', '#bdb200']
    from collections import Counter
    top = [k for k, _ in Counter(h for h in harness if h).most_common(8)]
    hmap = {k: PAL[i] for i, k in enumerate(top)}
    hcols = np.array([hmap.get(h, '#cccccc') for h in harness])
    ccols = np.where(B[:, t] > 0, '#2166ac', '#d95f02')

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.8))
    for r, (name, Z) in enumerate([('Raw trace embedding', Zr),
                                   ('qubric embedding', Zq)]):
        for c, (cname, cols) in enumerate([('by harness', hcols),
                                           ('by correctness', ccols)]):
            ax = axes[r, c]
            ax.scatter(Z[:, 0], Z[:, 1], s=34, c=cols, alpha=.85,
                       edgecolors='white', lw=.4)
            if r == 0:
                ax.set_title(f'Colored {cname}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('.8')
        axes[r, 0].set_ylabel(name, fontsize=10)
    import matplotlib.lines as mlines
    hs = [mlines.Line2D([], [], marker='o', ls='', ms=6, color=v, label=k)
          for k, v in hmap.items()]
    hs.append(mlines.Line2D([], [], marker='o', ls='', ms=6,
                            color='#cccccc', label='untagged'))
    fig.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.03, -0.005),
               ncol=5, fontsize=7, frameon=False)
    fig.legend(handles=[
        mlines.Line2D([], [], marker='o', ls='', ms=6, color='#2166ac',
                      label='resolved'),
        mlines.Line2D([], [], marker='o', ls='', ms=6, color='#d95f02',
                      label='unresolved')],
        loc='lower right', bbox_to_anchor=(0.98, -0.005), ncol=2,
        fontsize=7, frameon=False)
    fig.suptitle(f'One task, 107 systems ({q100[t]})', fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig('figures/fig1B_onetask.png', dpi=200)
    print('wrote figures/fig1B_onetask.png')


if __name__ == '__main__':
    fig_a()
    fig_b()
