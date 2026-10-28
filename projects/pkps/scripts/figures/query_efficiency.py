#!/usr/bin/env python
"""Main real-data figure 1 -- query efficiency on the heterogeneous 18-task suite.
Each cell is observed with m noisy queries; predict its true (full-budget) score. Sample =
the m-query mean; DKPS/PKPS denoise via the perspective; IRT is the item-response baseline;
ensemble = sample (+) PKPS. Three observation levers; MAE over observed cells, +/- 1 SEM."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({                              # clean modern style: white, subtle grid, despined
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#afbec6', 'axes.linewidth': 0.9,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#e5e7eb', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': '#486884', 'ytick.color': '#486884',
    'axes.labelcolor': '#213c66', 'axes.titlecolor': '#0a2245', 'text.color': '#213c66', 'font.size': 11,
})
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument('--suite', choices=['helm', 'eee'], default='helm')
args = ap.parse_args()
# suite-specific: results dir, full cohort size n, output stem
CFG = {'helm': dict(dir='results-pkps-rd1', n=93, stem='fig_suite_query_efficiency'),
       'eee':  dict(dir='results-eee-rd1',  n=45, stem='fig_eee_query_efficiency')}[args.suite]
D = Path(CFG['dir'])
# colour = method; line style = score-only (dashed: sample, IRT) vs response embedding
# (solid: DKPS, PKPS); the ensemble is solid and thicker.
STYLE = {'sample': dict(color='#9ca3af', ls='--', label='Sample score'),
         'irt':    dict(color='#afbec6', ls='--', label='IRT'),
         'dkps':   dict(color='#93aacc', ls='-',  label='DKPS'),
         'pkps':   dict(color='#3596ff', ls='-',  label='PKPS'),
         'ens':    dict(color='#114471', ls='-',  label='Ensemble', lw=3.6)}
ORDER = ['sample', 'irt', 'dkps', 'pkps', 'ens']
N = CFG['n']
PANELS = [('budget',   'm',       r'queries per cell $m$',           rf'$n{{=}}{N},\ p_{{\mathrm{{task}}}}{{=}}1$', True),
          ('n_models', 'n_models', r'number of models $n$',          r'$m{=}2,\ p_{\mathrm{task}}{=}1$',   False),
          ('coverage', 'p_task',  r'task coverage $p_{\mathrm{task}}$', rf'$m{{=}}2,\ n{{=}}{N}$',              False)]


def panel(ax, sweep, xcol, xlabel, fixed, logx, skip=()):
    df = pd.read_csv(D / f'rd1_suite_{sweep}.csv')
    # mean over tasks per (x, seed, method), then mean +/- SEM over seeds
    per_seed = df.groupby([xcol, 'seed', 'method'])['mae'].mean().reset_index()
    g = per_seed.groupby([xcol, 'method'])['mae'].agg(['mean', 'sem'])
    for m in ORDER:
        if m not in df['method'].values or m in skip:
            continue
        st = STYLE[m]
        sub = g.xs(m, level='method').sort_index()
        ax.errorbar(sub.index.values, sub['mean'], yerr=sub['sem'], marker='o', ms=4,
                    lw=st.get('lw', 2.6), color=st['color'], ls=st['ls'], label=st['label'],
                    capsize=2, elinewidth=0.8)
    if logx:
        ax.set_xscale('log', base=2)
    ax.set_xlabel(xlabel, fontsize=15.6)
    ax.set_title(fixed, fontsize=13.8)
    return g


# panel (a) keeps the full range (the budget crossover needs it); panels (b, c) zoom to
# the working band with IRT annotated off-scale -- on a shared axis its ~0.34 line pins
# the scale and flattens the real PKPS trends vs cohort and coverage.
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0), sharey=False)
gs = {}
for ax, (sweep, xcol, xlabel, fixed, logx), letter in zip(axes, PANELS, 'abc'):
    gs[sweep] = panel(ax, sweep, xcol, xlabel, fixed, logx,
                      skip=() if sweep == 'budget' else ('irt',))
    ax.set_title(f'({letter})', loc='left', fontweight='bold', fontsize=16.2)
    ax.tick_params(labelsize=13.1)
lo = min(gs[s].xs(m, level='method')['mean'].min()
         for s in ('n_models', 'coverage') for m in ('pkps', 'ens'))
hi = max(gs[s].xs('sample', level='method')['mean'].max() for s in ('n_models', 'coverage'))
for ax, sweep in zip(axes[1:], ('n_models', 'coverage')):
    ax.set_ylim(lo * 0.88, hi * 1.14)
    irt = gs[sweep].xs('irt', level='method')['mean'].mean()
    tag = 'off-scale' if irt > hi * 1.14 else 'not shown'
    ax.text(0.97, 0.965, f'IRT {tag} ({irt:.2f})', transform=ax.transAxes, ha='right',
            va='top', fontsize=11.5, color=STYLE['irt']['color'], fontweight='bold')
axes[0].set_ylabel(r'MAE$(\hat{y}, y)$', fontsize=15.6)
handles = [Line2D([0], [0], color=STYLE[m]['color'], ls=STYLE[m]['ls'],
                  lw=STYLE[m].get('lw', 2.6), marker='o', ms=4, label=STYLE[m]['label']) for m in ORDER]
fig.tight_layout()
# legend flush above the tallest panel title, suptitle flush above the legend, both
# centred on the panel span (canvas centre is skewed left by the shared ylabel)
fig.canvas.draw()
r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
xc = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
ytop = max(ax.get_tightbbox(r).transformed(inv).y1 for ax in axes)
lg = fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
                bbox_to_anchor=(xc, ytop + 0.012), fontsize=13.8)
fig.canvas.draw()
fig.suptitle('Query-efficiency ($m \\geq 1$)', fontsize=17.5, fontweight='bold',
             x=xc, y=lg.get_window_extent(r).transformed(inv).y1 + 0.015, va='bottom')
for ext in ('png', 'pdf'):
    fig.savefig(D / f'{CFG["stem"]}.{ext}', dpi=200, bbox_inches='tight')
print(f'wrote {D}/{CFG["stem"]}.png')
