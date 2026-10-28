#!/usr/bin/env python
"""SECONDARY query-efficiency conditional figure (4 ridgeline panels). Per-cell denoising win
Delta = sample MAE - ensemble MAE (>0 = the embedding-augmented ensemble beats the raw sample),
stacked by dataset, by budget, by cohort, and by task coverage. Each panel varies one lever and
holds the others at the MEDIAN of their sweep (m=4, n=40, p_task=0.5). Mean marker per row.
Reads the consistent per-cell dump from experiments/dump_qe_conditional.py."""
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

XL = (-0.12, 0.34)
XLAB = 'Sample score $-$ Ensemble (per cell)'
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
    ax.set_xticks([0.0, 0.2]); ax.tick_params(labelsize=13)
    ax.spines['left'].set_visible(False)
    ax.set_title(title, fontsize=12.8)   # match the completion-conditional sibling


d = pd.read_csv('results-pkps-rd1/qe_cond_cells.csv')
p = d.pivot_table(index=['sweep', 'm', 'n_models', 'p_task', 'seed', 'model', 'task', 'dataset'],
                  columns='method', values='abs_err').reset_index()
p['d'] = p['sample'] - p['ens']
cen = p[(p.sweep == 'cohort') & (p.n_models == 40)]              # median center (m=4, n=40, p_task=0.5)
bud, coh, cov = p[p.sweep == 'budget'], p[p.sweep == 'cohort'], p[p.sweep == 'coverage']

fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.9))
ridge(ax[0], [(n, cen[cen.dataset == dd]['d'].values) for n, dd in DS],
      '(d) by dataset\n$m{=}4,\\ n{=}40,\\ p_\\mathrm{task}{=}0.5$')
ridge(ax[1], [(f'$m{{=}}{m}$', bud[bud.m == m]['d'].values) for m in [1, 2, 4, 8, 16]],
      '(e) by budget\n$n{=}40,\\ p_\\mathrm{task}{=}0.5$')
ridge(ax[2], [(f'$n{{=}}{n}$', coh[coh.n_models == n]['d'].values) for n in [10, 40, 93]],
      '(f) by cohort\n$m{=}4,\\ p_\\mathrm{task}{=}0.5$')
ridge(ax[3], [(f'$p_\\mathrm{{task}}{{=}}{q}$', cov[cov.p_task == q]['d'].values) for q in [0.2, 0.5, 0.9]],
      '(g) by task coverage\n$m{=}4,\\ n{=}40$')
fig.supxlabel(XLAB, fontsize=15, y=0.175)
fig.tight_layout(rect=[0, 0.10, 1, 0.99])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_qe_conditional.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_qe_conditional.png')
