#!/usr/bin/env python
"""SECONDARY matrix-completion conditional figure (4 ridgeline panels). Per-missing-cell win
Delta = MC MAE - ensemble MAE (>0 = the embedding-augmented ensemble beats low-rank completion),
stacked by dataset, by cohort, by task coverage, and by query depth. Each panel varies one lever
and holds the others at the MEDIAN of their sweep (n=40, p_task=0.5, p_query=0.5). The win is
modest and TAIL-driven (completion is MC's home turf on this low-rank suite), so the mean marker
(v) is the honest aggregate. Reads the consistent per-cell dump from experiments/dump_completion_conditional.py."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({                              # clean modern style: white, subtle grid, despined
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': '#afbec6',
    'axes.linewidth': 0.9, 'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#ececec',
    'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'xtick.color': '#486884', 'ytick.color': '#486884',
    'axes.labelcolor': '#213c66', 'axes.titlecolor': '#0a2245', 'text.color': '#213c66', 'font.size': 11})
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

XL = (-0.2, 0.34)
XLAB = 'LRMC $-$ Ensemble (per missing cell)'
DS = [('MATH', 'math'), ('MedQA', 'med_qa'), ('LegalB', 'legalbench'), ('WMT', 'wmt_14')]


def ridge(ax, groups, title):
    xs = np.linspace(*XL, 300)
    for i, (lab, v) in enumerate(groups[::-1]):
        v = np.asarray(v, float); v = v[np.isfinite(v)]; off = i * 1.0
        if len(v) > 10:
            dd = gaussian_kde(v)(xs); dd = dd / dd.max() * 0.82
            ax.fill_between(xs, off, off + dd, where=xs >= 0, color='#3596ff', alpha=0.6, lw=0)
            ax.fill_between(xs, off, off + dd, where=xs < 0, color='#bbb', alpha=0.55, lw=0)
            ax.plot(xs, off + dd, color='#444', lw=0.7)
            ax.plot([v.mean()], [off + 0.02], 'v', color='#0a2245', ms=5)
        ax.text(XL[0] - 0.02 * (XL[1] - XL[0]), off + 0.12, lab, fontsize=14, ha='right', va='bottom')
    ax.axvline(0, color='#888', lw=0.9); ax.set_yticks([]); ax.set_xlim(*XL)
    ax.set_xticks([-0.2, 0.0, 0.2]); ax.tick_params(labelsize=13)
    ax.spines['left'].set_visible(False)
    ax.set_title(title, fontsize=12.8)   # 3-param 2nd line is wide; keep it inside the panel


d = pd.read_csv('results-pkps-unified/comp_cond_cells.csv')
d['d'] = d['mc_err'] - d['ens_err']
cen = d[(d.sweep == 'cohort') & (d.n_models == 40)]             # median center (n=40, p_task=0.5, p_q=0.5)
coh, cov, pq = d[d.sweep == 'cohort'], d[d.sweep == 'coverage'], d[d.sweep == 'querydepth']

fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.9))
ridge(ax[0], [(n, cen[cen.dataset == dd]['d'].values) for n, dd in DS],
      '(e) by dataset\n$n{=}40,\\ p_\\mathrm{task}{=}0.5,\\ p_\\mathrm{query}{=}0.5$')
ridge(ax[1], [(f'$n{{=}}{n}$', coh[coh.n_models == n]['d'].values) for n in [10, 40, 93]],
      '(f) by cohort\n$p_\\mathrm{task}{=}0.5,\\ p_\\mathrm{query}{=}0.5$')
ridge(ax[2], [(f'$p_\\mathrm{{task}}{{=}}{q}$', cov[cov.p_task == q]['d'].values) for q in [0.2, 0.5, 0.9]],
      '(g) by task coverage\n$n{=}40,\\ p_\\mathrm{query}{=}0.5$')
ridge(ax[3], [(f'$p_\\mathrm{{query}}{{=}}{q}$', pq[pq.p_query == q]['d'].values) for q in [0.25, 0.5, 1.0]],
      '(h) by query depth\n$n{=}40,\\ p_\\mathrm{task}{=}0.5$')
fig.supxlabel(XLAB, fontsize=15, y=0.175)
fig.tight_layout(rect=[0, 0.10, 1, 0.99])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-unified/fig_completion_conditional.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-unified/fig_completion_conditional.png')
