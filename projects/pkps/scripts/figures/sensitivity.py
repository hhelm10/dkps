#!/usr/bin/env python
"""Appendix sensitivity figure: (a, b) query-efficiency budget curves at MDS dimension
d in {4, 8, 12, 24} on both suites (d=12 is the paper's a-priori choice, read from the
main results); (c) MAE at m=2 for each FIXED query-bandwidth multiplier on the CV grid
(5 = the delta/DKPS limit), with the CV-selected bandwidth's MAE as a reference line.
Reads results-sens/ (see run_sens) and the main result dirs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#afbec6', 'axes.linewidth': 0.9,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#e5e7eb', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': '#486884', 'ytick.color': '#486884',
    'axes.labelcolor': '#213c66', 'axes.titlecolor': '#0a2245', 'text.color': '#213c66', 'font.size': 11,
})
import numpy as np
import pandas as pd

MAIN = {'helm': 'results-pkps-rd1', 'eee': 'results-eee-rd1'}
DIMS = [4, 8, 24]
DCOL = {4: '#6baed6', 8: '#3596ff', 24: '#2171b5'}   # d=8 (paper) in red
SIG_LABELS = ['0.03', '0.1', '0.3', '1', '3', r'$\delta$']
SUITE_LBL = {'helm': 'HELM', 'eee': 'EEE'}


def budget_curve(path, method='pkps'):
    df = pd.read_csv(Path(path) / 'rd1_suite_budget.csv')
    per_seed = df[df.method == method].groupby(['m', 'seed'])['mae'].mean().reset_index()
    return per_seed.groupby('m')['mae'].agg(['mean', 'sem'])


fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

# (a, b) MDS dimension
for ax, suite in zip(axes[:2], ['helm', 'eee']):
    for d in DIMS:
        path = MAIN[suite] if d == 8 else f'results-sens/{suite}-d{d}'
        g = budget_curve(path)
        ax.errorbar(g.index.values, g['mean'], yerr=g['sem'], marker='o', ms=3.5,
                    lw=2.4 if d == 8 else 1.8, color=DCOL[d],
                    label=f'$d{{=}}{d}$' + (' (paper)' if d == 8 else ''))
    s = budget_curve(MAIN[suite], 'sample')
    ax.errorbar(s.index.values, s['mean'], yerr=s['sem'], color='#999', ls='--', lw=1.8,
                marker='o', ms=3, label='Sample score')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('queries per cell $m$', fontsize=14.5)
    ax.legend(frameon=False, fontsize=10.5)
axes[0].set_ylabel(r'PKPS MAE$(\hat{y}, y)$', fontsize=14.5)

# (c) fixed bandwidth vs CV
ax = axes[2]
for suite, col in [('helm', '#486884'), ('eee', '#114471')]:
    ys, es = [], []
    for i in range(6):
        g = budget_curve(f'results-sens/{suite}-sig{i}')
        ys.append(g.loc[2, 'mean']); es.append(g.loc[2, 'sem'])
    ax.errorbar(range(6), ys, yerr=es, marker='o', ms=4, lw=2.2, color=col,
                label=f'{SUITE_LBL[suite]} (fixed $\\sigma$)')
    cv = budget_curve(MAIN[suite]).loc[2, 'mean']
    ax.axhline(cv, color=col, ls=':', lw=1.6)
    ax.text(5.15, cv, 'CV', color=col, fontsize=10, va='center')
ax.set_xticks(range(6)); ax.set_xticklabels(SIG_LABELS, fontsize=12)
ax.set_xlabel(r'fixed bandwidth ($\times$ median) at $m{=}2$', fontsize=14.5)
ax.legend(frameon=False, fontsize=10.5)

for ax, letter, title in zip(axes, 'abc',
                             ['MDS dimension (HELM)', 'MDS dimension (EEE)',
                              'query bandwidth']):
    ax.set_title(f'$\\bf{{({letter})}}$  {title}', loc='left', fontsize=13.5)
    ax.tick_params(labelsize=12.5)

fig.tight_layout()
Path('results-sens').mkdir(exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(f'results-sens/fig_sensitivity.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-sens/fig_sensitivity.png')
