"""Figure 5: sensitivity of the Moltbook subnetworks to the analysis choices.

Reference: the reported run (Jan 30 - Feb 9, T = 8, cohort active in >= 6 of 8 steps, nomic-embed-v1.5,
d by the elbow (10), all data); its eight subnetworks are the rounds with p <= 0.05. Each variant
changes one choice and reruns the whole method (real_tstep.py):
  a  embedding function: nomic-embed-v1.5 (reference), all-MiniLM-L6-v2, all-mpnet-base-v2,
     bge-small-en-v1.5, OpenAI text-embedding-3-small
  b  embedding dimension d in {2, 4, 8, 16, 32, 64} (reference: elbow, 10)
  c  time steps T in {4, 6, 8, 10, 12}, on the 1,623 agents active in >= 75% of the steps under every
     one of these binnings (so the cohort is the same for every T)
  d  amount of data: a random 10, 20, 30, 50, 80% of all actions in the window (3 seeds each), cohort
     fixed
(a)-(d): rows 1-8 are the reference subnetworks; a dot in the subnetwork's color, sized and shaded by
the largest Jaccard index between the subnetwork (restricted to the variant's cohort) and any
subnetwork the variant detects (p <= 0.05, at least two agents), averaged over seeds for (d). Row 9
('new'): a check mark where the variant detects a subnetwork that shares no agent with any of the
eight (a set inside one of them counts toward that subnetwork's Jaccard index, not as new); for (d),
shaded by the share of seeds that do. The reference setting is shaded.
(e): share of the non-reference variants of (a)-(d) that recover each subnetwork (Jaccard >= 0.5), and
the share that find a new one.

    python writing/figures/make_fig_sensitivity.py
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
from pipeline.paths import REPO, DERIVED  # noqa: E402
from pipeline.figstyle import apply_print_sizes, below_axes  # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import hv_style as hv  # noqa: E402

warnings.filterwarnings('ignore')
OUT = os.path.join(HERE, 'fig_sensitivity')
D_DIR = os.path.join(DERIVED, 'real_tstep')
BASE = '01-30_02-09_T8_ms6'
COHORT_T = '4-6-8-10-12'
SEEDS = (0, 1, 2)


def load(tag):
    f = os.path.join(D_DIR, f'{tag}.pkl')
    if not os.path.exists(f):
        return None
    r = pickle.load(open(f, 'rb'))
    names = np.array(r['cohort'])
    found = [set(names[g['members']].tolist()) for g in r['res']['groups'] if g['p'] <= 0.05 and g['subnetwork']]
    return set(r['cohort']), found


ref_cohort, REF = load(BASE)
K = len(REF)
jac = lambda a, b: len(a & b) / len(a | b) if a | b else 0.0


def score(variant):
    """(Jaccard per reference subnetwork, whether a new subnetwork is found) for one run, or None."""
    if variant is None:
        return None
    cohort, found = variant
    refs = [S & cohort for S in REF]
    J = np.array([max((jac(S, F) for F in found), default=0.0) if S else np.nan for S in refs])
    allref = set().union(*refs)
    return J, float(any(not (F & allref) for F in found))         # new: shares no agent with the eight


def column(tags):
    """Mean Jaccard, share of runs recovering each subnetwork (Jaccard >= 0.5) and share of runs that
    find a new subnetwork, over the runs that exist (seeds); None if none exist yet."""
    sc = [s for s in (score(load(t)) for t in tags) if s is not None]
    if not sc:
        return None
    Js = np.array([s[0] for s in sc])
    return np.nanmean(Js, 0), np.nanmean(Js >= 0.5, 0), np.mean([s[1] for s in sc])


AXES = [
    ('a) Embedder', [('nomic', [BASE]), ('MiniLM', [f'{BASE}_emb-minilm']), ('mpnet', [f'{BASE}_emb-mpnet']),
                               ('bge', [f'{BASE}_emb-bge-small']), ('OpenAI', [f'{BASE}_emb-openai-small'])],
     'nomic'),
    ('b) Dimension (d)', [(str(d), [f'{BASE}_d{d}']) for d in (2, 4, 8)] + [('10', [BASE])]
     + [(str(d), [f'{BASE}_d{d}']) for d in (16, 32, 64)], '10'),
    ('c) Time steps (T)', [(str(T), [f'01-30_02-09_T{T}_cT{COHORT_T}']) for T in (4, 6, 8, 10, 12)], '8'),
    ('d) Data (%)', [(f'{int(f * 100)}', [f'{BASE}_frac{f:g}s{s}' for s in SEEDS]) for f in (0.1, 0.2, 0.3, 0.5, 0.8)]
     + [('100', [BASE])], '100'),
]

hv.apply()
apply_print_sizes(hv, 20)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']   # Figure 4
C0, INK = hv.ROLES['baseline_pale']['color'], hv.INK_TITLE
lab = dict(fontsize=hv.SIZES['annot'])
size = lambda j: 40 + 520 * j
rows = K + 1                                                                # 8 subnetworks + 'new'
wr = [len(cols) for _, cols, _ in AXES] + [3.2]
fig, axes = plt.subplots(1, 5, figsize=(20, 6.0), gridspec_kw=dict(width_ratios=wr, wspace=0.3))
recov, newf = [], []
for c, (ax, (title, cols, ref_label)) in enumerate(zip(axes[:4], AXES)):
    n = len(cols)
    jr = [l for l, _ in cols].index(ref_label)
    ax.axvspan(jr - 0.5, jr + 0.5, color=hv.WASH, zorder=0)
    ax.axhline(K - 0.5, color=hv.EDGE, lw=1.0)
    for j, (_, tags) in enumerate(cols):
        r = column(tags)
        if r is None:                                                       # not finished yet
            ax.scatter([j] * rows, range(rows), s=18, facecolors='none', edgecolors=hv.EDGE, lw=0.8)
            continue
        J, R_, nw = r
        for k in range(K):
            if np.isfinite(J[k]):
                ax.scatter(j, k, s=size(J[k]), color=PAL[k], alpha=0.25 + 0.75 * J[k], edgecolors='none')
        if nw > 0:
            ax.scatter(j, K, marker='$\\checkmark$', s=260, color=INK, alpha=0.3 + 0.7 * nw, edgecolors='none')
        if j != jr:
            recov.append(R_); newf.append(nw)
    ax.set_xticks(range(n)); ax.set_xticklabels([l for l, _ in cols], rotation=45 if c == 0 else 0,
                                                ha='right' if c == 0 else 'center', rotation_mode='anchor')
    ax.set_yticks(range(rows)); ax.set_yticklabels([str(k + 1) for k in range(K)] + ['new'] if c == 0 else [])
    ax.set_ylim(rows - 0.5, -0.5); ax.set_xlim(-0.6, n - 0.4); ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc='left', fontsize=hv.SIZES['subtitle'])
axes[0].set_ylabel('subnetwork', **lab)
ax = axes[4]
if recov:
    sh = np.r_[np.nanmean(np.array(recov), 0), np.mean(newf)]
    ax.barh(range(rows), sh, color=PAL + [C0], height=0.62)
ax.axhline(K - 0.5, color=hv.EDGE, lw=1.0)
ax.set_ylim(rows - 0.5, -0.5); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([])
ax.set_xlabel('share of settings', **lab); ax.grid(axis='y', visible=False)
ax.set_title('e) Overall robustness', loc='left', fontsize=hv.SIZES['subtitle'])
# legends
kw = dict(fontsize=hv.SIZES['annot'], frameon=False, handlelength=1.4, handletextpad=0.4, columnspacing=1.0,
          title_fontsize=hv.SIZES['annot'])
dot = lambda col, l: Line2D([], [], color=col, ls='None', marker='o', ms=14, label=l)
groups = [('subnetwork', [dot(PAL[k], str(k + 1)) for k in range(K)]),
          ('Jaccard index', [Line2D([], [], color=hv.INK_MUTE, ls='None', marker='o', ms=np.sqrt(size(v)), label=f'{v:g}')
                             for v in (0, 0.5, 1)]),
          ('', [Patch(color=hv.WASH, label='reference analysis')])]
Y = below_axes(fig)
legs = [fig.legend(handles=h, title=t, loc='upper left', ncol=len(h), bbox_to_anchor=(0, Y), **kw) for t, h in groups]
fig.canvas.draw(); inv = fig.transFigure.inverted()
w = [np.ptp(inv.transform(l.get_window_extent())[:, 0]) for l in legs]
x = 0.5 - (sum(w) + 0.03 * (len(w) - 1)) / 2
for l, ww in zip(legs, w):
    l.set_bbox_to_anchor((x, Y), transform=fig.transFigure); x += ww + 0.03
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', dpi=200, bbox_inches='tight')
print('saved', OUT)
