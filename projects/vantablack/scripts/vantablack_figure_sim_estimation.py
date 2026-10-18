"""Figure: membership estimation simulations, 2x4 panels. Columns as in the detection
figure (swarm size, amount of content, subnetwork size, time); rows are the precision
|S_hat & S| / |S_hat| and recall |S_hat & S| / |S| of the estimate S_hat, one line per
effect size rho > 0. All replicates count, rejected or not; precision is averaged over
replicates with a non-empty S_hat (an empty estimate has recall 0).

    python vantablack_figure_sim_estimation.py [--stage pilot]

Reads data/results/sim_detection_<stage>.csv (sim_detection.py); writes
artifacts/vantablack_figure_sim_estimation[_<stage>].{png,pdf}.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pipeline.paths import RESULTS, ART, REPO
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import hv_style as hv  # noqa: E402

from sim_detection import BASE, SWEEPS  # noqa: E402
from vantablack_figure_sim_detection import PANELS  # noqa: E402

ROWS = [('precision', 'precision'), ('recall', 'recall')]
Q = 0.05   # Benjamini-Hochberg level of the membership step (pipeline/tstep.estimate)


def cell_means(df, sweep, col):
    d = df[(df.sweep == sweep) | (df.sweep == 'base')]
    d = d.assign(x=d[sweep]).dropna(subset=[col])
    g = d.groupby(['rho', 'x'])[col].agg(['mean', 'std', 'count']).reset_index()
    g['se'] = g['std'].fillna(0) / np.sqrt(g['count'])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='pilot')
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(RESULTS, f'sim_detection_{a.stage}.csv'))
    df = df[df.rho > 0]
    hv.apply()
    rhos = sorted(df.rho.unique())
    cols = {r: hv.CMAP_BLUE(v) for r, v in zip(rhos, np.linspace(0.35, 1.0, len(rhos)))}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4), sharey=True, sharex='col')
    for i, (col, ylabel) in enumerate(ROWS):
        for j, (sweep, title, xlabel) in enumerate(PANELS):
            ax = axes[i, j]
            g = cell_means(df, sweep, col)
            for rho in rhos:
                h = g[g.rho == rho].sort_values('x')
                ax.plot(h.x, h['mean'], marker='o', ms=5, color=cols[rho], lw=2.4, label=f'ρ = {rho:g}')
                ax.fill_between(h.x, h['mean'] - h.se, h['mean'] + h.se, color=cols[rho], alpha=0.15, lw=0)
            if col == 'precision':
                ax.axhline(1 - Q, color=hv.REFLINE, ls=':', lw=1.2)
            ax.axvline(BASE[sweep], color=hv.EDGE, lw=6, zorder=0)
            ax.set_xscale('log')
            ax.set_xticks(SWEEPS[sweep]); ax.set_xticklabels([str(v) for v in SWEEPS[sweep]])
            ax.minorticks_off()
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(title, loc='left', fontsize=hv.SIZES['subtitle'])
            else:
                ax.set_xlabel(xlabel, fontsize=hv.SIZES['annot'])
        axes[i, 0].set_ylabel(ylabel, fontsize=hv.SIZES['annot'])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), fontsize=hv.SIZES['annot'], bbox_to_anchor=(0.5, -0.04))
    reps = int(df.groupby(['sweep', 'value', 'rho']).size().median())
    fig.text(0.995, 0.005, f'{a.stage}: ~{reps} replicates per point, ±1 SE; dotted = 1 − q; shaded bar = base value',
             ha='right', va='bottom', fontsize=10, color=hv.INK_MUTE)
    fig.tight_layout()
    name = 'vantablack_figure_sim_estimation' + ('' if a.stage == 'full' else f'_{a.stage}')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ART, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    print(f'wrote artifacts/{name}.png/.pdf')


if __name__ == '__main__':
    main()
